from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.data.components.attn.datagen import Attn_generate

# from components.attn.datagen import Attn_generate # For single file debug


class AttnDataModule(LightningDataModule):
    """Example of LightningDataModule for Attn dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        seq_length: int = 100,
        input_dim: int = 1,
        output_dim: int = 1,
        Gaussian_input: bool = False,
        train_val_test_split: Tuple[int, int, int] = (12_800, 1_280, 1_280),
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_size = train_val_test_split[0]
        self.val_size = train_val_test_split[1]
        self.test_size = train_val_test_split[2]

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Gaussian_input = Gaussian_input
        self.seed = seed

    def prepare_data(self):
        """Generate data if needed.

        Do not use it to assign state (self.x = y).
        """

        Attn_generate(
            self.hparams.data_dir,
            self.train_size + self.val_size + self.test_size,
            self.seq_length,
            self.input_dim,
            self.Gaussian_input,
        )

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            input_file_path = (
                self.hparams.data_dir
                + f"attn_T{self.seq_length}_N{self.train_size + self.val_size + self.test_size}_D{self.input_dim}_inputs.npy"
            )
            output_file_path = (
                self.hparams.data_dir
                + f"attn_T{self.seq_length}_N{self.train_size + self.val_size + self.test_size}_D{self.input_dim}_outputs.npy"
            )

            inputs = np.load(input_file_path, "r")
            outputs = np.load(output_file_path, "r")

            inputs_tensor = torch.from_numpy(inputs.copy()).float()
            outputs_tensor = torch.from_numpy(outputs.copy()).float()
            dataset = torch.utils.data.TensorDataset(inputs_tensor, outputs_tensor)

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(self.seed),
            )

    def train_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = AttnDataModule()
