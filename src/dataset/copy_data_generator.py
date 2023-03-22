from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CopyDataset(Dataset):

    def __init__(self, num_samples, copy_seq_len, zeros_len, pad_len,  data_seed=0, data_seed_offset=0, simple=False):
        # input_data = [B,S], output_data = [B,S]
        self.input_data, self.output_data = self.generate_copy_sequence_batch(num_samples, copy_seq_len,
                                                                              zeros_len, pad_len, data_seed,
                                                                              data_seed_offset, simple)

    def __len__(self):
        # index on batch dim
        return self.input_data.shape[0]
        # return self.input_data.shape[1]

    def __getitem__(self, item):
        # index on batch dim
        return self.input_data[item], self.output_data[item]
        # return self.input_data[:, item], self.output_data[:, item]

    def generate_copy_sequence_batch(self, num_samples: int, copy_digits_seq_len: int = 10, copy_zeros_len: int = 50,
                                     pad_len: int = 60, data_seed: int = 0, data_seed_offset: int = 0, simple: bool = False,
                                     state: str = 'train') -> \
            Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Generates the input and output batches used in the copy task.
        A single input data sample consists of a sequence of digits of size (copy_seq_len + zeros_len + '9' + pad_len).
        The copy sequence are digits from [1,8].
        The zero sequence are a sequence of 0's which represent the dormant phase.
        The digit 9 is used to identify the end of the copy and 0 seq and start of padding (using 0's)

        A output sample will only contain the copy sequence.
        Example input:
        [1 4 2 0 0 0 0 9 0 0 ] # seq len(=10) = copy(3) + zeros(4) + end token(1) + padding(2)
        Example output:
        [0 0 0 0 0 0 0 1 4 2 ]  # seq len(=10) = padding(2) + zeros(5) + copy(3)


        :param num_samples: number of data samples to generate i.e. full batch size
        :param copy_digits_seq_len: number of digits the model should learn to copy
        :param copy_zeros_len: number of null digits the model should learn to ignore
        :param pad_len: number of padding tokens to use (i.e. after the copying tokens occur to fill out the remaning space).
        :param data_seed: Seed for data generation
        :param data_seed_offset: offset added to dataseed to avoid the 'train', 'val' and 'test' data generating the same copy numbers
        :param simple: set all copy digits to be 1. (Used for sanity checks).
        :return: Batch of data as (X, Y) where the shape of X and Y is [B=num_samples, S=seq len]
        """
        timestep_dim = 1

        # create the generator using the data_seed. Independent from the run seed which is used from the model.
        data_seed += data_seed_offset
        print(f"Generator created from data seed: {data_seed}")
        generator = torch.Generator()
        generator.manual_seed(data_seed)

        # simple mode will just generate ones for the copy digits.
        if simple:
            copy_digits_seq = torch.ones(size=(num_samples, copy_digits_seq_len)).type(torch.FloatTensor)
        else:
            copy_digits_seq = torch.randint(1, 9, size=(num_samples, copy_digits_seq_len), generator=generator).type(torch.FloatTensor)
        copy_zeros_seq_X = torch.zeros((num_samples, copy_zeros_len - 1)).type(torch.FloatTensor)  # will contain '9'
        copy_zeros_seq_Y = torch.zeros((num_samples, copy_zeros_len)).type(torch.FloatTensor)  # will not contain '9'
        end_indicator = torch.full((num_samples, 1), 9).type(torch.FloatTensor)  # indicates copy sequence has ended
        pad_seq = torch.zeros((num_samples, pad_len)).type(torch.FloatTensor)
        X_seq = torch.cat((copy_digits_seq, copy_zeros_seq_X, end_indicator, pad_seq), dim=timestep_dim).type(torch.LongTensor)
        Y_seq = torch.cat((pad_seq, copy_zeros_seq_Y, copy_digits_seq), dim=timestep_dim).type(torch.LongTensor)

        # since dataset's get_item returns a group of items where the batch dim is 0, just return [B,S] and deal with reshaping later on
        # X_seq = X_seq.t()
        # Y_seq = Y_seq.t()

        return X_seq, Y_seq  # ([B, S], [B,S])


class CopyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 0, seed: int = 0, data_seed: int = 0, simple: bool = False,
                 train_num_samples: int = 50000, train_copy_len: int = 10, train_zeros_len: int = 50, train_pad_len: int = 10,
                 val_num_samples: int = 20000, val_copy_len: int = 10, val_zeros_len: int = 50, val_pad_len: int = 10,
                 test_num_samples: int = 20000, test_copy_len: int = 10, test_zeros_len: int = 50, test_pad_len: int = 10):

        super(CopyDataModule, self).__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset_args = {'data_seed_offset': 0, 'data_seed': data_seed, 'simple': simple,
                                   'num_samples': train_num_samples, 'copy_seq_len': train_copy_len,
                                   'zeros_len': train_zeros_len, 'pad_len': train_pad_len}
        self.val_dataset_args = {'data_seed_offset': 1, 'data_seed': data_seed, 'simple': simple,
                                 'num_samples': val_num_samples, 'copy_seq_len': val_copy_len,
                                 'zeros_len': val_zeros_len, 'pad_len': val_pad_len}
        self.test_dataset_args = {'data_seed_offset': 2, 'data_seed': data_seed, 'simple': simple,
                                  'num_samples': test_num_samples, 'copy_seq_len': test_copy_len,
                                  'zeros_len': test_zeros_len, 'pad_len': test_pad_len}

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CopyDataset(**self.train_dataset_args)
            self.val_dataset = CopyDataset(**self.val_dataset_args)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = CopyDataset(**self.test_dataset_args)

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          generator=g, pin_memory=True)

    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          generator=g, pin_memory=True)

    def test_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          generator=g, pin_memory=True)


if __name__ == '__main__':
    # Quick check of data generation
    batch_size = 2
    copy_len = 3
    null_tokens_len = 4
    pad_len = copy_len

    ds = CopyDataset(batch_size, copy_len, null_tokens_len, pad_len, data_seed=42, simple=False)
    in_data, out_data = ds.input_data, ds.output_data
    assert (in_data.shape == torch.empty(batch_size, (copy_len + null_tokens_len + pad_len)).shape), \
        "Input Tensor shape is incorrect."
    assert (out_data.shape == in_data.shape), "Output Tensor shape is incorrect."

    samples = 600
    dataset = CopyDataset(samples, copy_seq_len=copy_len, zeros_len=null_tokens_len, pad_len=pad_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X, Y = next(iter(dataloader))
    print(X)  # X
    print(Y)  # Y
    print(X.shape)
    print(f"seq {copy_len + null_tokens_len + pad_len}, batch {batch_size}")
