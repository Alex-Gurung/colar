from .prompt_utils import generate_next_chapter_messages

with open("/mnt/disk/new_nrl_ncp/prompt_to_datapoint_with_baseline_ppl_qwen7B_cpu_new.pkl", "rb") as f:
    prompt_to_datapoint_with_baseline_ppl = pickle.load(f)
    

USE_PPL = True
USE_BASELINE = True
SUBTRACT_RANDOM_BASELINE = True
RANDOM_BASELINE_WEIGHTING = 0.5
PREDEFINED_RANDOM_CHOICES = False

def find_index_of_last_system_message(
    input_ids, special_token, offset_after_token=4, end_offset=4
):
    # find the index of the last system message in the input_ids
    # offset is to avoid encoding the special tokens from tokenization
    for i in range(len(input_ids) - end_offset - 1, 0, -1):
        if input_ids[i] == special_token:
            return i + offset_after_token
    print("DIDNT FIND IT, RETURNING -1")
    return -1

def make_message_data_from_datapoint_and_model_response(datapoint, model_response, tokenizer):
    next_chapter_messages = generate_next_chapter_messages(
        datapoint,
        [[model_response, ""]],

    )

    next_chapter_tokens = tokenizer.apply_chat_template(
        next_chapter_messages, tokenize=True, return_tensors="pt"
    )

    start_of_system_message = find_index_of_last_system_message(
        next_chapter_tokens[0],
        tokenizer.eos_token_id,
        offset_after_token=5,
    )
    labels = next_chapter_tokens.clone()
    labels[:, :start_of_system_message] = -100

    return next_chapter_tokens, labels

def find_datapoint_from_sequence(question_string, tokenizer):
    # print(f"sequence: {sequence}")
    decoded_sequence = question_string
    
    split_term = "### Next Chapter Synopsis: ###"
    prior_story_text = (
        # decoded_sequence.split("### Next Chapter Synopsis: ###")[1]
        decoded_sequence.split(split_term)[1]
        .split("###")[0]
        .strip()
    )
    # prior_story_text = decoded_sequence.split("### Prior Story: ###")[1].split("### Next Chapter Synopsis: ###")[0].strip()
    prior_story_text = tokenizer.decode(tokenizer.encode(prior_story_text))

    datapoint = prompt_to_datapoint_with_baseline_ppl[prior_story_text]
    return datapoint, prior_story_text

def get_next_chapter_messages_from_sequence(question_string, original_model_response, datapoint, tokenizer):
    # print(f"question_string: {question_string}")
    decoded_sequence = question_string
    
    model_response = original_model_response.split("In summary:")[-1].strip()
    model_response = model_response.split("In summary,")[-1].strip()
    model_response = model_response.split("Detailed Plan:")[-1].strip()
    baseline_ppl = datapoint["baseline_ppl"].detach().to("cpu").item()
    if not USE_PPL:
        baseline_ppl = -baseline_ppl

    next_chapter_tokens, labels = make_message_data_from_datapoint_and_model_response(datapoint, model_response, tokenizer)

    return next_chapter_tokens, labels, baseline_ppl, model_response, original_model_response

def randomly_select_datapoint(cur_datapoint):
    # subselect datapoints to have a different story_id
    cur_story_id = cur_datapoint["story_id"]
    datapoints_from_different_story = [datapoint for datapoint in prompt_to_datapoint_with_baseline_ppl.values() if datapoint["story_id"] != cur_story_id]
    random_datapoint = random.choice(datapoints_from_different_story)
    return random_datapoint

def convert_loss_to_reward(loss: float, baseline_ppl: Optional[float] = None):
    # note baseline_ppl could be either ppl or loss
    reward = -loss
    if USE_PPL:
        ppl = torch.exp(loss).item()
        reward = ppl
    if USE_BASELINE and baseline_ppl is not None:
        reward = (baseline_ppl - reward) / baseline_ppl * 100
    return reward

@torch.no_grad()
def calculate_reward(question_string, original_model_response, tokenizer, remote_reward_model, random_datapoint):
    datapoint, prior_story_text = find_datapoint_from_sequence(question_string, tokenizer)
    print(f"sanity check: {datapoint['story_id']}")
    next_chapter_tokens, labels, baseline_ppl, model_response, original_model_response = get_next_chapter_messages_from_sequence(question_string, original_model_response, datapoint, tokenizer)

    attention_mask = next_chapter_tokens.ne(tokenizer.pad_token_id)
    num_actions = next_chapter_tokens.size(-1)
    with torch.no_grad():
        outputs = remote_reward_model(
            input_ids=next_chapter_tokens.to(remote_reward_model.device),
            attention_mask=attention_mask.to(remote_reward_model.device),
            labels=labels.to(remote_reward_model.device),
        )
        loss = outputs.loss.cpu()
    reward = convert_loss_to_reward(loss, baseline_ppl)
    print(f"reward from loss: {reward}")

    if SUBTRACT_RANDOM_BASELINE:
        random_baseline_ppl = random_datapoint["baseline_ppl"].detach().to("cpu").item()
        if not USE_PPL:
            # random_baseline_ppl is loss (NLL), so negate it so higher is better
            random_baseline_ppl = -random_baseline_ppl
        # get loss from random datapoint
        random_next_chapter_tokens, random_labels = make_message_data_from_datapoint_and_model_response(random_datapoint, model_response, tokenizer)
        random_attention_mask = random_next_chapter_tokens.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            outputs = remote_reward_model(
                input_ids=random_next_chapter_tokens.to(remote_reward_model.device),
                attention_mask=random_attention_mask.to(remote_reward_model.device),
                labels=random_labels.to(remote_reward_model.device),
            )
        random_loss = outputs.loss.cpu()
        random_reward = convert_loss_to_reward(random_loss, random_baseline_ppl)
        print(f"reward from random datapoint: {random_reward}")
        reward -= random_reward * RANDOM_BASELINE_WEIGHTING
    print(f"final reward: {reward}")

    return reward

# VERY HACKY BUT WE WILL IDENTIFY A RANDOM OTHER DATAPOINT FOR EVERY ELEMENT IN PROMPT TO DATAPOINT
# that way, no matter what datapoint we see during training, we always are consistent within a group
# the downside to this is the model may learn (for this prompt, don't mention X, Y, Z)
if PREDEFINED_RANDOM_CHOICES:
    prior_story_to_random_datapoint = {}
    prior_story_to_group_idx = {}
    for prior_story, datapoint in prompt_to_datapoint_with_baseline_ppl.items():
        if prior_story not in prior_story_to_random_datapoint:
            prior_story_to_random_datapoint[prior_story] = randomly_select_datapoint(datapoint)
            prior_story_to_group_idx[prior_story] = len(prior_story_to_random_datapoint)
else:
    prior_story_to_random_datapoint = None
    prior_story_to_group_idx = None

def get_r_refs(question_strings, pred_answers, remote_reward_model, tokenizer, n_samples_per_prompt):
    epsilon = 1e-10
    rewards = []

    # Calculate number of groups
    # group will be based on the prior_story_text
    if not PREDEFINED_RANDOM_CHOICES:
        prior_story_to_random_datapoint = {}
        prior_story_to_group_idx = {}
        question_string = question_strings[0]

        for i, model_response in enumerate(pred_answers):
            question_string = question_strings[i]
            datapoint, prior_story_text = find_datapoint_from_sequence(question_string, tokenizer)
            if prior_story_text not in prior_story_to_random_datapoint:
                prior_story_to_random_datapoint[prior_story_text] = randomly_select_datapoint(datapoint)
                prior_story_to_group_idx[prior_story_text] = len(prior_story_to_random_datapoint)

    num_groups = len(prior_story_to_group_idx)
    # breakpoint()
    print(f"num_groups: {num_groups}; n_samples_per_prompt: {n_samples_per_prompt}; len(question_strings): {len(question_strings)}")
    # group_to_random = {}
    # num_groups = len(sequences_cpu) // n_samples_per_prompt
    # print(f"num_groups: {num_groups}; n_samples_per_prompt: {n_samples_per_prompt}; len(sequences_cpu): {len(sequences_cpu)}")
    # group_to_random = {}

    # Pre-select random datapoint for each group
    # for group_idx in range(num_groups):
    #     first_seq_idx = group_idx * n_samples_per_prompt
    #     datapoint, prior_story_text = find_datapoint_from_sequence(sequences_cpu[first_seq_idx][0], tokenizer)
    #     group_to_random[group_idx] = randomly_select_datapoint(datapoint)
    #     print(f"Group {group_idx} random datapoint: {group_to_random[group_idx]['story_id']}")
    count_seen_from_group = {}
    overall_start_time = time.time()
    question_string = question_strings[0]
    # assert len(question_strings) == 1
    for i, model_response in enumerate(pred_answers):
        # group_idx = i // n_samples_per_prompt
        # random_datapoint = group_to_random[group_idx]
        question_string = question_strings[i]
        datapoint, prior_story_text = find_datapoint_from_sequence(question_string, tokenizer)
        group_idx = prior_story_to_group_idx[prior_story_text]
        count_seen_from_group[group_idx] = count_seen_from_group.get(group_idx, 0) + 1
        random_datapoint = prior_story_to_random_datapoint[prior_story_text]
        s = time.time()
        reward = calculate_reward(question_string, model_response, tokenizer, remote_reward_model, random_datapoint)
        e = time.time()
        print(f"Group {group_idx}, Sample {count_seen_from_group[group_idx]}: reward: {reward}; time taken: {e - s} seconds")
        if reward == 0:
            reward += epsilon  # avoid log(0)
        rewards.append(torch.tensor([reward]))

    e = time.time()
    print(f"Time taken to get all ({len(rewards)}) losses: {e - overall_start_time} seconds")
    for group_idx, count in count_seen_from_group.items():
        print(f"Group {group_idx} seen {count} times")

    return rewards