#################
# Prompt Utility Functions
#################
# structure of task is the following:
# inputs:
# query (the prompt to generate the reasoning step)
# response (the reasoning step)
# continuation (the next chapter)
# outputs:
# reward (the ppl of the continuation - baseline_ppl)

ROLE_PROMPT = "You are a helpful and expert writing assistant."

REASONING_ROLE_PROMPT = "You are a helpful and expert writing assistant."

GENERIC_PROMPT_PREFIX = """{instructions_0}

{story_information}

{next_chapter_summary_text}
{next_chapter_header_text}
{prior_reasoning_text}### Instructions: ###
{instructions_1}"""

GENERIC_PROMPT_PREFIX_CHUNKED = """{instructions_0}

{story_information}

{next_chapter_summary_text}
{prior_reasoning_text}### Instructions: ###
{instructions_1}"""

STORY_INFORMATION_TEMPLATE = """{high_level_summary_text}{prior_summary_text}{character_sheet_text}{last_n_chapters_text}"""

# INSTRUCTION_PREAMBLE_TEMPLATE = """Instructions: You will be given {given_text}. You will first reason about the given story and about what should come next. Next, you will write the next chapter of the story."""
INSTRUCTION_PREAMBLE_TEMPLATE = """Instructions: You will be given {given_text}. You will first reason about the given story and about what should come next. Next, you will write the next chapter of the story."""
INSTRUCTION_PREAMBLE_CHUNKED_TEMPLATE = """Instructions: You will be given {given_text}. You will first reason about the given story and about what should come next. Next, you will write the next couple hundred words of the story."""
INSTRUCTION_GIVEN_NOTHING_TEMPLATE = """Instructions: You will first reason about the given story and about what should come next. Next, you will write the next chapter of the story."""
INSTRUCTION_GIVEN_NOTHING_CHUNKED_TEMPLATE = """Instructions: You will first reason about the given story and about what should come next. Next, you will write the next couple hundred words of the story."""
BASELINE_INSTRUCTION_TEMPLATE = """Instructions: Use all of the information provided to write the next chapter. Your response should begin with the chapter header."""
BASELINE_INSTRUCTION_TEMPLATE_CHUNKED = """Instructions: Use all of the information provided to write the next couple hundred words of the story."""
BASELINE_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION = """Instructions: Use all of the information provided to write the next chapter (at least {approximate_chapter_length} words). End the chapter with <END_OF_CHAPTER>. Your response should begin with the chapter header."""
BASELINE_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION_CHUNKED = """Instructions: Use all of the information provided to write the next small chunk of the story (around {approximate_chapter_length} words). End your chunk with <END_OF_CHUNK>."""

USING_REASONING_INSTRUCTION_TEMPLATE = """Instructions: Use all of the preliminary thoughts and information provided to write the next chapter. Your response should begin with the chapter header."""
USING_REASONING_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION = """Instructions: Use all of the preliminary thoughts and information provided to write the next chapter (at least {approximate_chapter_length} words long). End the chapter with <END_OF_CHAPTER>."""
USING_CHUNKED_REASONING_INSTRUCTION_TEMPLATE = """Instructions: Use all of the preliminary thoughts and information provided to write the next couple hundred words of the story."""
USING_CHUNKED_REASONING_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION = """Instructions: Use all of the preliminary thoughts and information provided to write the next small chunk of the story (around {approximate_chapter_length} words long). End your chunk with <END_OF_CHUNK>."""


# TRYING TO GET THE REASONING TO BE MORE SPECIFIC
GENERAL_REASONING_FOR_PLAN_INSTRUCTION_TEMPLATE = """Instructions: Based on the next chapter's synopsis and header, please reason step by step to come up with a more detailed plan for the next chapter. Format your reasoning with "<citation>source A says X</citation>, therefore <reasoning>reasoning</reasoning>" pairs, where the sources can be the character sheets, the high-level story plan, the previous-chapters summary, the next chapter synopsis, and the previous few chapters. Add and modify your conclusions as you remember more information. End your response with a detailed paragraph explaining your reasoning as to how next chapter will unfold (including plot and character points), beginning this paragraph with "In summary: "."""
# GENERAL_REASONING_FOR_PLAN_INSTRUCTION_TEMPLATE_CHUNKED = """Instructions: Based on the next chapter's synopsis and all information provided, please reason step by step to come up with a more detailed plan for the next couple hundred words of the story. Format your reasoning with citation, reasoning pairs (e.g. <citation>source A says X</citation>, therefore <reasoning>reasoning</reasoning>), where the sources can be the character sheets, the high-level story plan, the previous-chapters summary, the next chapter synopsis, and the previous story text. Add and modify your conclusions as you remember more information. End your response with a detailed paragraph explaining your reasoning as to how the next couple paragraphs will unfold, beginning this paragraph with "In summary: "."""
GENERAL_REASONING_FOR_PLAN_INSTRUCTION_TEMPLATE_CHUNKED = """Instructions: Based on all of the information provided, please reason step by step about how the next few sentences of the story will unfold. Format your reasoning with citation, reasoning pairs (e.g. <citation>source A says X</citation>, therefore <reasoning>reasoning</reasoning>), where the sources can be the character sheets, the high-level story plan, the previous-chapters summary, the next chapter synopsis, and the previous story text. Add and modify your conclusions as you remember more information. End your response with a paragraph explaining your reasoning as to how the next few sentences will unfold, beginning this paragraph with "In summary: "."""

PRIOR_PLOT_SUMMARY_TEMPLATE = (
    """\n### Summary of Already Written Chapters: ###\n{prior_plot_summary}\n"""
)
HIGH_LEVEL_SUMMARY_TEMPLATE = (
    """\n### High-Level Story Summary/Plan: ###\n{high_level_plot_summary}\n"""
)
CHARACTER_SHEET_TEMPLATE = (
    """## Character Sheet ({character_name}): ##\n{character_sheet}\n"""
)
ALL_CHARACTER_SHEETS_TEMPLATE = """\n### Character Sheets: ###\n{character_sheets}\n"""

NEXT_CHAPTER_SUMMARY_TEMPLATE = (
    """\n### Next Chapter Synopsis: ###\n{next_chapter_summary}\n"""
)
NEXT_CHAPTER_SUMMARY_TEMPLATE_CHUNKED = (
    """\n### This Chapter Synopsis: ###\n{next_chapter_summary}\n"""
)

NEXT_CHAPTER_HEADER_TEMPLATE = (
    """\n### Next Chapter Header: ###\n{next_chapter_header}\n"""
)

PRIOR_REASONING_TEMPLATE = (
    """\n### Thoughts About the Next Chapter ###\n{prior_reasoning}\n\n"""
)

PRIOR_REASONING_TEMPLATE_CHUNKED = (
    # """\n### Thoughts About the Next couple Hundred Words ###\n{prior_reasoning}\n\n"""
    """\n### Thoughts About the Next Few Sentences ###\n{prior_reasoning}\n\n"""
)

PRIOR_REASONING_PLAN_TEMPLATE = (
    """\n### Detailed Plan for the Next Chapter ###\n{prior_reasoning}\n\n"""
)

PRIOR_REASONING_PLAN_TEMPLATE_CHUNKED = (
    # """\n### Detailed Plan for the Next couple Hundred Words ###\n{prior_reasoning}\n\n"""
    """\n### Detailed Plan for the Next Few Sentences ###\n{prior_reasoning}\n\n"""
)


def get_instruction_preamble(
    includes_high_level_summary,
    includes_prior_plot_summary,
    includes_character_sheet,
    includes_next_chapter_summary,
    last_n_chapters=-1,
    for_chunk=False,
):
    if not for_chunk:
        if last_n_chapters is None:
            what_will_be_given = []
        elif last_n_chapters == 1:
            what_will_be_given = ["the most recent chapter of the story"]
        elif last_n_chapters > 1:
            what_will_be_given = [f"the last {last_n_chapters} chapters of the story"]
        else:
            what_will_be_given = ["the entire story"]
    else:
        if for_chunk:
            what_will_be_given = ["the previous couple hundred words of the story (potentially including part of the current chapter)"]
        else:
            what_will_be_given = ["the previous couple thousand words of the story"]

    if includes_high_level_summary:
        what_will_be_given.append("a high-level plan of the entire story")
    if includes_prior_plot_summary:
        what_will_be_given.append("a summary of the previously written chapters")
    if includes_character_sheet:
        what_will_be_given.append("character sheets for the three main characters")
    if includes_next_chapter_summary:
        if for_chunk:
            what_will_be_given.append(
                "a brief synopsis of what should happen in this chapter (not just the next few sentences)"
            )
        else:
            what_will_be_given.append(
                "a brief synopsis of what should happen in the next chapter"
            )

    if len(what_will_be_given) == 0:
        if for_chunk:
            return INSTRUCTION_GIVEN_NOTHING_CHUNKED_TEMPLATE
        else:
            return INSTRUCTION_GIVEN_NOTHING_TEMPLATE
    elif len(what_will_be_given) == 1:
        given_text = what_will_be_given[0]
    else:
        given_text = (
            ", ".join(what_will_be_given[:-1]) + " and " + what_will_be_given[-1]
        )
    if for_chunk:
        given_text = (
            given_text
            + ". The previous story text will also contain the current chapter header, containing the chapter's title and any other epigraph-type text"
        )

    if for_chunk:
        return INSTRUCTION_PREAMBLE_CHUNKED_TEMPLATE.format(given_text=given_text)
    else:
        return INSTRUCTION_PREAMBLE_TEMPLATE.format(given_text=given_text)


def get_story_section(
    story_text, last_n_chapters=-1, include_packed_story_text=False, for_chunk=False
):
    prefix = "### Prior Story: ###\n"
    if not for_chunk:
        if include_packed_story_text:
            prefix = "### Prior Story (Previous Chapter + This Chapter): ###\n"

        if last_n_chapters == 1 and not include_packed_story_text:
            prefix = "### Previous Chapter: ###\n"
        elif last_n_chapters > 1 and not include_packed_story_text:
            prefix = f"### Previous {last_n_chapters} Chapters: ###\n"
    return prefix + story_text


########################
# get_prompt
########################
def compile_prompt_components(
    story_text,
    prior_plot_summary,
    high_level_plot_summary,
    character_sheets,
    next_chapter_synopsis,
    next_chapter_header,
    last_n_chapters=-1,
    include_packed_story_text=False,
    for_chunk=False,
):
    includes_high_level_summary = high_level_plot_summary is not None
    includes_prior_plot_summary = prior_plot_summary is not None
    includes_character_sheet = character_sheets is not None
    includes_next_chapter_summary = next_chapter_synopsis is not None

    last_n_chapters_text = (
        get_story_section(
            story_text,
            last_n_chapters,
            include_packed_story_text=include_packed_story_text,
            for_chunk=for_chunk,
        )
        if last_n_chapters is not None
        else ""
    )

    high_level_summary_text = (
        HIGH_LEVEL_SUMMARY_TEMPLATE.format(
            high_level_plot_summary=high_level_plot_summary
        )
        if includes_high_level_summary
        else ""
    )
    prior_plot_summary_text = (
        PRIOR_PLOT_SUMMARY_TEMPLATE.format(prior_plot_summary=prior_plot_summary)
        if includes_prior_plot_summary
        else ""
    )

    if includes_character_sheet:
        formatted_csheets = []
        for character_name, character_sheet in character_sheets.items():
            formatted_csheets.append(
                CHARACTER_SHEET_TEMPLATE.format(
                    character_name=character_name, character_sheet=character_sheet
                )
            )
        character_sheet_text = ALL_CHARACTER_SHEETS_TEMPLATE.format(
            character_sheets="\n".join(formatted_csheets)
        )
    else:
        character_sheet_text = ""

    next_chapter_summary_text = (
        NEXT_CHAPTER_SUMMARY_TEMPLATE.format(next_chapter_summary=next_chapter_synopsis)
        if includes_next_chapter_summary
        else ""
    )
    
    # we don't include the header in the chunked prompt
    next_chapter_header_text = ""
    if not for_chunk:
        next_chapter_header_text = NEXT_CHAPTER_HEADER_TEMPLATE.format(
            next_chapter_header=next_chapter_header.strip()
        )


    instruction_preamble = get_instruction_preamble(
        includes_high_level_summary,
        includes_prior_plot_summary,
        includes_character_sheet,
        includes_next_chapter_summary,
        last_n_chapters=last_n_chapters,
        for_chunk=for_chunk,
    )

    story_information = STORY_INFORMATION_TEMPLATE.format(
        high_level_summary_text=high_level_summary_text,
        prior_summary_text=prior_plot_summary_text,
        last_n_chapters_text=(
            "\n" + last_n_chapters_text + "\n" if last_n_chapters_text != "" else ""
        ),
        character_sheet_text=character_sheet_text,
    ).strip()
    return {
        "instruction_preamble": instruction_preamble,
        "story_information": story_information,
        "next_chapter_summary_text": next_chapter_summary_text,
        "next_chapter_header_text": next_chapter_header_text,
    }


def get_baseline_prompt(
    story_text,
    prior_plot_summary,
    high_level_plot_summary,
    character_sheets,
    next_chapter_synopsis,
    next_chapter_header,
    last_n_chapters=-1,
    for_actual_generation=False,
    approximate_chapter_length=None,
    for_chunk=False,
):
    if for_actual_generation:
        baseline_instruction = (
            BASELINE_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION.format(
                approximate_chapter_length=approximate_chapter_length
            )
            if not for_chunk
            else BASELINE_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION_CHUNKED.format(
                approximate_chapter_length=approximate_chapter_length
            )
        )
    else:
        baseline_instruction = (
            BASELINE_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION.format(
                approximate_chapter_length=approximate_chapter_length
            )
            if for_chunk
            else BASELINE_INSTRUCTION_TEMPLATE
        )

    prompt_components = compile_prompt_components(
        story_text,
        prior_plot_summary,
        high_level_plot_summary,
        character_sheets,
        next_chapter_synopsis,
        next_chapter_header,
        last_n_chapters=last_n_chapters,
        for_chunk=for_chunk,
    )

    if not for_chunk:
        prompt = GENERIC_PROMPT_PREFIX.format(
            instructions_0=prompt_components["instruction_preamble"],
            prior_reasoning_text="",  # baseline prompt has no prior reasoning text
            instructions_1=baseline_instruction,
            story_information=prompt_components["story_information"],
            next_chapter_summary_text=prompt_components["next_chapter_summary_text"],
            next_chapter_header_text=prompt_components["next_chapter_header_text"],
        )
    else:
        prompt = GENERIC_PROMPT_PREFIX_CHUNKED.format(
            instructions_0=prompt_components["instruction_preamble"],
            prior_reasoning_text="",  # baseline prompt has no prior reasoning text
            instructions_1=baseline_instruction,
            story_information=prompt_components["story_information"],
            next_chapter_summary_text=prompt_components["next_chapter_summary_text"],
            # no next chapter header text for chunked baseline prompt
        )
    return prompt


def get_reasoning_from_story_prompt(
    story_text,
    prior_plot_summary,
    high_level_plot_summary,
    character_sheets,
    next_chapter_synopsis,
    next_chapter_header,
    last_n_chapters=-1,
    prior_reasoning="",
    for_chunk=False,
):
    if for_chunk:
        give_me_some_general_reasoning = (
            GENERAL_REASONING_FOR_PLAN_INSTRUCTION_TEMPLATE_CHUNKED
        )
    else:
        give_me_some_general_reasoning = GENERAL_REASONING_FOR_PLAN_INSTRUCTION_TEMPLATE

    prompt_components = compile_prompt_components(
        story_text,
        prior_plot_summary,
        high_level_plot_summary,
        character_sheets,
        next_chapter_synopsis,
        next_chapter_header,
        last_n_chapters=last_n_chapters,
        for_chunk=for_chunk,
    )

    if len(prior_reasoning) > 0:
        if for_chunk:
            reasoning_text = PRIOR_REASONING_TEMPLATE_CHUNKED.format(
                prior_reasoning=prior_reasoning
            )
        else:
            reasoning_text = PRIOR_REASONING_TEMPLATE.format(
                prior_reasoning=prior_reasoning
            )
    else:
        reasoning_text = ""

    if not for_chunk:
        prompt = GENERIC_PROMPT_PREFIX.format(
            instructions_0=prompt_components["instruction_preamble"],
            prior_reasoning_text=reasoning_text,
            instructions_1=give_me_some_general_reasoning,
            story_information=prompt_components["story_information"],
            next_chapter_summary_text=prompt_components["next_chapter_summary_text"],
            next_chapter_header_text=prompt_components["next_chapter_header_text"],
        )
    else:
        prompt = GENERIC_PROMPT_PREFIX_CHUNKED.format(
            instructions_0=prompt_components["instruction_preamble"],
            prior_reasoning_text=reasoning_text,
            instructions_1=give_me_some_general_reasoning,
            story_information=prompt_components["story_information"],
            next_chapter_summary_text=prompt_components["next_chapter_summary_text"],
            # no next chapter header text for chunked reasoning prompt
        )
    return prompt


def get_next_chapter_with_reasoning_prompt(
    story_text,
    prior_plot_summary,
    high_level_plot_summary,
    character_sheets,
    next_chapter_synopsis,
    next_chapter_header,
    last_n_chapters=-1,
    prior_reasoning="",
    for_actual_generation=False,
    approximate_chapter_length=None,
    for_chunk=False,
):
    if for_actual_generation:
        use_prior_reasoning_instruction = (
            USING_REASONING_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION.format(
                approximate_chapter_length=approximate_chapter_length
            )
            if not for_chunk
            else USING_CHUNKED_REASONING_INSTRUCTION_TEMPLATE_FOR_ACTUAL_GENERATION.format(
                approximate_chapter_length=approximate_chapter_length
            )
        )
    else:
        use_prior_reasoning_instruction = (
            USING_CHUNKED_REASONING_INSTRUCTION_TEMPLATE.format(
                approximate_chapter_length=approximate_chapter_length
            )
            if for_chunk
            else USING_REASONING_INSTRUCTION_TEMPLATE
        )

    prompt_components = compile_prompt_components(
        story_text,
        prior_plot_summary,
        high_level_plot_summary,
        character_sheets,
        next_chapter_synopsis,
        next_chapter_header,
        last_n_chapters=last_n_chapters,
        for_chunk=for_chunk,
    )

    if len(prior_reasoning) > 0:
        if for_chunk:
            reasoning_text = PRIOR_REASONING_PLAN_TEMPLATE_CHUNKED.format(
                prior_reasoning=prior_reasoning
            )
        else:
            reasoning_text = PRIOR_REASONING_PLAN_TEMPLATE.format(
                prior_reasoning=prior_reasoning
            )
    else:
        reasoning_text = ""

    if not for_chunk:
        prompt = GENERIC_PROMPT_PREFIX.format(
            instructions_0=prompt_components["instruction_preamble"],
            prior_reasoning_text=reasoning_text,
            instructions_1=use_prior_reasoning_instruction,
            story_information=prompt_components["story_information"],
            next_chapter_summary_text=prompt_components["next_chapter_summary_text"],
            next_chapter_header_text=prompt_components["next_chapter_header_text"],
        )
    else:
        prompt = GENERIC_PROMPT_PREFIX_CHUNKED.format(
            instructions_0=prompt_components["instruction_preamble"],
            prior_reasoning_text=reasoning_text,
            instructions_1=use_prior_reasoning_instruction,
            story_information=prompt_components["story_information"],
            next_chapter_summary_text=prompt_components["next_chapter_summary_text"],
            # no next chapter header text for chunked reasoning prompt
        )

    return prompt


#########################################
# message utils
#########################################


def generate_reasoning_from_story_messages(
    datapoint,
    prior_question_answer_pairs,
    USE_SYSTEM_ROLE=True,
    include_prior_plot_summary=True,
    include_high_level_plot_summary=True,
    include_character_sheets=True,
    include_next_chapter_synopsis=True,
    include_last_n_chapters=True,
):
    story_text = datapoint["story_text"]
    prior_plot_summary = (
        datapoint["prior_plot_summary"] if include_prior_plot_summary else None
    )
    high_level_plot_summary = (
        datapoint["high_level_plot_summary"]
        if include_high_level_plot_summary
        else None
    )
    character_sheets = (
        datapoint["character_sheets"] if include_character_sheets else None
    )
    next_chapter_synopsis = (
        datapoint["next_chapter_synopsis"] if include_next_chapter_synopsis else None
    )
    next_chapter_header = datapoint["next_chapter_header"]
    last_n_chapters = datapoint["last_n_chapters"] if include_last_n_chapters else None

    if USE_SYSTEM_ROLE:
        role_message = {"role": "system", "content": REASONING_ROLE_PROMPT}
    else:
        role_message = None

    lines = [
        (question + "\n" + answer).strip()
        for question, answer in prior_question_answer_pairs
    ]
    prior_reasoning_text = "\n".join(lines)
    
    for_chunk = "chunked" in datapoint and datapoint["chunked"]

    user_message = {
        "role": "user",
        "content": get_reasoning_from_story_prompt(
            story_text,
            prior_plot_summary,
            high_level_plot_summary,
            character_sheets,
            next_chapter_synopsis,
            next_chapter_header,
            last_n_chapters=last_n_chapters,
            prior_reasoning=prior_reasoning_text,
            for_chunk=for_chunk,
        ),
    }

    if USE_SYSTEM_ROLE:
        return [role_message, user_message]
    else:
        return [user_message]


def generate_next_chapter_messages(
    datapoint,
    prior_question_answer_pairs,
    USE_SYSTEM_ROLE=True,
    include_prior_plot_summary=True,
    include_high_level_plot_summary=True,
    include_character_sheets=True,
    include_next_chapter_synopsis=True,
    include_last_n_chapters=True,
    for_actual_generation=False,
    approximate_chapter_length=None,
):
    story_text = datapoint["story_text"]
    for_chunk = "chunked" in datapoint and datapoint["chunked"]
    prior_plot_summary = (
        datapoint["prior_plot_summary"] if include_prior_plot_summary else None
    )
    high_level_plot_summary = (
        datapoint["high_level_plot_summary"]
        if include_high_level_plot_summary
        else None
    )
    character_sheets = (
        datapoint["character_sheets"] if include_character_sheets else None
    )
    next_chapter_synopsis = (
        datapoint["next_chapter_synopsis"] if include_next_chapter_synopsis else None
    )
    last_n_chapters = datapoint["last_n_chapters"] if include_last_n_chapters else None
    next_chapter_header = datapoint["next_chapter_header"]

    if USE_SYSTEM_ROLE:
        role_message = {"role": "system", "content": ROLE_PROMPT}
    else:
        role_message = None

    use_baseline_prompt = len(prior_question_answer_pairs) == 0
    if use_baseline_prompt:
        user_message = {
            "role": "user",
            "content": get_baseline_prompt(
                story_text,
                prior_plot_summary,
                high_level_plot_summary,
                character_sheets,
                next_chapter_synopsis,
                next_chapter_header,
                last_n_chapters,
                for_actual_generation=for_actual_generation,
                approximate_chapter_length=approximate_chapter_length,
                for_chunk=for_chunk,
            ),
        }
    else:
        lines = [
            (question + "\n" + answer).strip()
            for question, answer in prior_question_answer_pairs
        ]
        prior_reasoning_text = "\n".join(lines)

        user_message = {
            "role": "user",
            "content": get_next_chapter_with_reasoning_prompt(
                story_text,
                prior_plot_summary,
                high_level_plot_summary,
                character_sheets,
                next_chapter_synopsis,
                next_chapter_header,
                last_n_chapters=last_n_chapters,
                prior_reasoning=prior_reasoning_text,
                for_actual_generation=for_actual_generation,
                approximate_chapter_length=approximate_chapter_length,
                for_chunk=for_chunk,
            ),
        }

    # add next chapter message
    next_chapter_message = {
        "role": "assistant",
        "content": datapoint["next_chapter"],
    }
    if USE_SYSTEM_ROLE:
        return [role_message, user_message, next_chapter_message]
    else:
        return [user_message, next_chapter_message]