import json
import copy
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


class ConvertedRLDataset(Dataset):
    def __init__(self, data: List[dict]):
        super().__init__()
        self.data = {}
        for idx, d in enumerate(data):
            self.data[idx] = {
                "idx": d.get("idx", idx),
                "question": d["question"],
                "answer": d["answer"],
                "steps": d["steps"],
                "n_steps": len(d["steps"].split('\n')) if d["steps"] else 0,
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
    def __init__(self, dataset_name, tiny_dataset=False, epoch_scaling=1, all_config=None):
        super().__init__()
        self.dataset_name = dataset_name
        # self.dataset_dir = Path("/mnt/disk/baseline_colar/colar_rl_data")
        # self.dataset_dir = Path("/mnt/disk/baseline_colar/musr_rl_data")
        self.dataset_dir = Path("/mnt/disk/baseline_colar/ff_rl_data")
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

        # Initialize datasets
        if stage == "fit":
            train_data = load_split("train")
            val_data = load_split("val")

            self.train_set = self._create_dataset(train_data, "train")
            self.val_set = self._create_dataset(val_data, "val")

            # Dataset verification box
            print("\n" + "="*80)
            print("📊 DATASET LOADING VERIFICATION")
            print("="*80)
            print(f"🗂️  Dataset Directory: {self.dataset_dir}")
            print(f"🔬 Tiny Dataset Mode: {self.tiny_dataset}")
            print(f"📈 Epoch Scaling: {self.epoch_scaling}")
            print(f"🚂 Train samples: {len(self.train_set):,}")
            print(f"✅ Val samples: {len(self.val_set):,}")
            print(f"📦 Batch size per GPU: {self.batch_size}")

            # Sample data inspection
            if len(self.train_set) > 0:
                sample = self.train_set[0]
                print("\n📋 SAMPLE TRAINING EXAMPLE:")
                print("-" * 60)
                print(f"Question: {sample['question'][:100]}{'...' if len(sample['question']) > 100 else ''}")
                print(f"Steps length: {len(sample['steps'])} chars")
                print(f"Answer: {sample['answer']}")
                print(f"N_steps: {sample['n_steps']}")
            print("="*80 + "\n")

        elif stage == "test":
            train_data = load_split("train")
            self.train_set = self._create_dataset(train_data, "train")

            test_data = load_split("test")
            self.test_set = self._create_dataset(test_data, "test")

            print("\n" + "="*60)
            print("🧪 TEST DATASET LOADED")
            print("="*60)
            print(f"🔍 Test samples: {len(self.test_set):,}")
            print("="*60 + "\n")

    def _create_dataset(self, raw_data: List[dict], mode: str) -> ConvertedRLDataset:
        return ConvertedRLDataset(raw_data)

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