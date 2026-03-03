import json
import copy
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class ConvertedRLDataset(Dataset):
    """Dataset for pre-converted RL data in JSONL format (question/steps/answer)."""

    def __init__(self, data: List[dict]):
        super().__init__()
        self.data = {}
        for idx, d in enumerate(data):
            self.data[idx] = {
                "idx": d.get("idx", idx),
                "question": d["question"],
                "answer": d["answer"],
                "steps": d["steps"],
                "n_steps": len(d["steps"].split('\n\n')) if d["steps"] else 0,
            }
        self.all_indices = list(self.data.keys())
        self.indices = copy.deepcopy(self.all_indices)

    def get_all_indices(self):
        return self.all_indices

    def set_indices(self, indices: List[int]):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        data_idx = self.indices[idx]
        return self.data[data_idx]


class ConvertedRLDataModule(pl.LightningDataModule):
    """DataModule for pre-converted RL data. Expects {split}.jsonl files."""

    def __init__(self, dataset_name, tiny_dataset=False, epoch_scaling=1, all_config=None, dataset_dir=None):
        super().__init__()
        self.dataset_name = dataset_name
        if dataset_dir is None:
            raise ValueError("dataset_dir must be specified in the config (e.g. dataset_dir: /path/to/data)")
        self.dataset_dir = Path(dataset_dir)
        self.tiny_dataset = tiny_dataset
        self.epoch_scaling = epoch_scaling
        self.all_config = all_config
        self.batch_size = all_config.dataloader.batch_size

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str = None):
        def load_split(split: str):
            file_path = self.dataset_dir / f"{split}.jsonl"
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            if self.tiny_dataset:
                data = data[:32]
            return data

        if stage == "fit":
            self.train_set = ConvertedRLDataset(load_split("train"))
            self.val_set = ConvertedRLDataset(load_split("val"))
        elif stage == "test":
            self.train_set = ConvertedRLDataset(load_split("train"))
            self.test_set = ConvertedRLDataset(load_split("test"))

    def get_all_train_indices(self):
        return self.train_set.get_all_indices()

    def set_train_indices(self, indices):
        self.train_set.set_indices(indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.all_config.dataloader.num_workers,
            pin_memory=self.all_config.dataloader.pin_memory,
            persistent_workers=self.all_config.dataloader.get("persistent_workers", False),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.all_config.dataloader.val_batch_size,
            shuffle=False,
            num_workers=self.all_config.dataloader.num_workers,
            pin_memory=self.all_config.dataloader.pin_memory,
            persistent_workers=self.all_config.dataloader.get("persistent_workers", False),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.all_config.dataloader.val_batch_size,
            shuffle=False,
            num_workers=self.all_config.dataloader.num_workers,
            pin_memory=self.all_config.dataloader.pin_memory,
            persistent_workers=self.all_config.dataloader.get("persistent_workers", False),
        )
