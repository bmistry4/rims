import os.path as path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from src.dataset.copy_data_generator import CopyDataModule
from src.flag_sanity_checks import copying_sanity_checks
from src.models.pl_rims import LitRIM
from src.parser_args import create_base_parser
from src.utils import setup_wandb_logger


class Copying:
    def __init__(self):
        # Parse arguments
        parser = create_base_parser()
        args = parser.parse_args()
        args.device = torch.device("cuda" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu")

        # print parser args
        print(' '.join(f'{k}: {v}\n' for k, v in vars(args).items()))
        ###############################################################################################################
        # Sanity checks for sizes which should match
        copying_sanity_checks(args)

        ###############################################################################################################
        # REPRODUCIBILITY
        # https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#reproducibility
        seed_everything(args.seed, workers=True)
        torch.cuda.manual_seed(args.seed)

        # Set reproducability flags - see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np_rng = np.random.RandomState(args.seed)
        torch.set_default_dtype(torch.float32 if args.pytorch_precision == 32 else torch.float64)
        ################################################################################################################
        data_module = CopyDataModule(args.batch_size, args.num_workers, args.seed, args.data_seed, False,
                                     args.train_num_samples, args.train_copy_len, args.train_zeros_len,
                                     args.train_pad_len,
                                     args.val_num_samples, args.val_copy_len, args.val_zeros_len, args.val_pad_len,
                                     args.test_num_samples, args.test_copy_len, args.test_zeros_len, args.test_pad_len)
        model = LitRIM(args)
        ################################################################################################################
        # print param counts
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        for name, p in model.named_parameters():
            print(name, p.numel(), p.shape)
        print(f"Total trainable params: {total_params}\n")

        # for n, p in model.named_parameters():
        #     print(n)
        #     print(p.data)
        ################################################################################################################
        # setup logger
        run_log_path = path.join(args.log_path, args.name_prefix)
        logger = setup_wandb_logger(run_log_path, args)
        ################################################################################################################
        # TODO - remove logger=False, enable_checkpointing=False when ready to do logging
        # TODO -> overfit_batches flag and fast_dev_run
        trainer = pl.Trainer(gpus=args.num_gpus, max_epochs=args.max_epochs, default_root_dir=run_log_path,
                             deterministic=True, gradient_clip_val=args.grad_clip_value,
                             logger=logger, enable_checkpointing=not args.no_save, overfit_batches=False,
                             enable_progress_bar=args.progress_bar, num_sanity_val_steps=0, profiler=None)
        data_module.setup()
        # the val_dataloaders contains datasets for interpolation (validation) and extrapolation (testing)
        trainer.fit(model,
                    train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=[data_module.val_dataloader(), data_module.test_dataloader()]
                    )
        trainer.test(model, data_module)


if __name__ == '__main__':
    Copying()
