import argparse


def create_base_parser():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Runs regression')

    # experiment args
    parser.add_argument('--id',
                        action='store',
                        default=-1,
                        type=int,
                        help='Unique id assigned for each experiment. (The same model with different seeds will have '
                             'the same ID value.)'),
    parser.add_argument('--logger_off',
                        action='store_true',
                        default=False,
                        help='Switch off data logging.')
    parser.add_argument('--seed',
                        action='store',
                        default=0,
                        type=int),
    parser.add_argument('--data_seed',
                        action='store',
                        default=2792445,    # generated using https://www.random.org/
                        type=int,
                        help='seed to use to generate the data.'),
    parser.add_argument('--cell',
                        action='store',
                        default='VanillaLSTMCell',
                        type=str,
                        help='class name of the type of Cell that wants to be utilised.'),
    parser.add_argument('--use_vanilla_lstm_layer',
                        action='store_true',
                        default=False,
                        help='Use a plain lstm layer (instead of a RIMS cell.')

    parser.add_argument('--log_path',
                        action='store',
                        default="../wandb_logs/",
                        type=str,
                        help='Path to save logger outputs for experiments. (Relative to the experiments folder)')
    parser.add_argument('--name_prefix',
                        action='store',
                        default='copying',
                        type=str,
                        help='Type of task. Name of where the data (tensorboard/checkpoints/etc.) should be stored.'
                             'Compatible with nesting e.g., simple/copying')
    parser.add_argument('--name_postfix',
                        action='store',
                        default='',
                        type=str,
                        help='Additional info to add to experiment name')
    parser.add_argument('--wandb_project',
                        action='store',
                        default='rims',
                        type=str,
                        help='Name of project to save experiments in. (Highest level)')
    parser.add_argument('--wandb_notes',
                        action='store',
                        default='',
                        type=str,
                        help='Notes to save about a run. (Can be just useful misc info)')

    parser.add_argument('--log_last_n',
                        action='store',
                        default=0,
                        type=int,
                        help='logs loss for last n timesteps. If 0 nothing will be logged'),

    parser.add_argument('--learning_rate',
                        action='store',
                        default=1e-3,
                        type=float,
                        help='Specify the learning-rate')
    parser.add_argument('--max_epochs',     # todo - ablations only use 100 epochs
                        action='store',
                        default=150,
                        type=int),
    parser.add_argument('--batch_size',
                        action='store',
                        default=64,
                        type=int),
    parser.add_argument('--batch_first',
                        action='store_true',
                        default=False,
                        help='Controls if the batch is in dim 0 (True) or 1 (False). If False the timestep dim is first')
    parser.add_argument('--num_gpus',
                        action='store',
                        default=1,
                        type=int),
    parser.add_argument('--pytorch_precision',
                        type=int,
                        default=32,
                        help='Precision for pytorch to work in')

    parser.add_argument('--dropout',
                        action='store',
                        default=0.5,
                        type=float,
                        help='Dropout value. Applied to the Cell output of all layers. Set to 0 to switch it off')
    parser.add_argument('--grad_clip_value',        # todo - used in copying task for original code
                        action='store',
                        default=1,
                        type=float,
                        help='Clips gradient global norm be within the given value. (Set to 0 to switch it off).')

    parser.add_argument('--use_inactive_rims',
                        action='store_true',
                        default=False,
                        help='lets the inactive rims be used as part of the output')
    parser.add_argument('--block_grads',
                        action='store_true',
                        default=False,
                        help='blocks the gradients of the inactive rims contributing in the back pass')

    # trainer/ lit related args
    parser.add_argument('--no_save',
                        action='store_true',
                        default=False,
                        help='switch off checkpointing')
    parser.add_argument('--progress_bar',
                        action='store_true',
                        default=False,
                        help='switch on progress bar')

    # size args
    parser.add_argument('--num_tokens',
                        action='store',
                        default=20,         # todo - why 20 not 10???
                        type=int,
                        help=''),
    parser.add_argument('--d_out',          # can be as low as 9 but original code uses 20
                        action='store',
                        default=20,
                        type=int,
                        help='Specify the output units of the model. Assumes you can model integers 0-8.'),
    parser.add_argument('--d_enc_inp',
                        action='store',
                        default=600,
                        type=int,
                        help='Dimensionality the inputs should be encoded to.'),

    # rim args
    parser.add_argument('--d_hid_all_rims_per_layer',
                        action='store',
                        nargs='+',
                        default=[600],
                        type=int,
                        help='Equal to value of h_size in the paper'),
    parser.add_argument('--num_rims_per_layer',
                        action='store',
                        nargs='+',
                        default=[6],
                        type=int,
                        help='Equal to value of k_T (total RIMs for a layer) in the paper'),
    parser.add_argument('--num_active_rims_per_layer',
                        action='store',
                        nargs='+',
                        default=[4],
                        type=int,
                        help='Equal to value of k_A (active rims for a layer) in the paper'),
    parser.add_argument('--num_modules_read_input',
                        action='store',
                        default=2,
                        type=int,
                        help='Number of input keys and values'),

    # attention args
    parser.add_argument('--use_comm_attn',
                        action='store_true',
                        default=False,      # todo rims default should be true
                        help='use communication attention')
    parser.add_argument('--in_num_head',
                        action='store',
                        default=1,
                        type=int,
                        help=''),
    parser.add_argument('--comm_num_head',
                        action='store',
                        default=4,
                        type=int,
                        help=''),

    parser.add_argument('--in_d_k',
                        action='store',
                        default=64,
                        type=int,
                        help=''),
    parser.add_argument('--in_d_v',
                        action='store',
                        default=400,        # from appendix |d_rim|*4
                        type=int,
                        help=''),
    parser.add_argument('--comm_d_k',
                        action='store',
                        default=16,         # fixme - table 3 mentions 32 but code uses 16
                        type=int,
                        help=''),
    parser.add_argument('--comm_d_v',
                        action='store',
                        default=16,         # fixme - table 3 mentions 32 but code uses 16
                        type=int,
                        help=''),
    parser.add_argument('--in_residual',
                        action='store_true',
                        default=False,
                        help='Use gated residual in the input attention')
    parser.add_argument('--comm_residual',
                        action='store_true',
                        default=False,  # todo: default is True
                        help='Use gated residual in the communication attention')

    # dataset args
    parser.add_argument('--num_workers',
                        action='store',
                        default=0,
                        type=int),

    parser.add_argument('--train_num_samples',
                        type=int,
                        default=200*64,     # = 12,800; see org code num_batches=200
                        help='')
    parser.add_argument('--train_copy_len',
                        type=int,
                        default=10,
                        help='')
    parser.add_argument('--train_zeros_len',
                        type=int,
                        default=50,
                        help='')
    parser.add_argument('--train_pad_len',
                        type=int,
                        default=10,
                        help='')

    parser.add_argument('--val_num_samples',
                        type=int,
                        default=200*64,
                        help='')
    parser.add_argument('--val_copy_len',
                        type=int,
                        default=10,
                        help='')
    parser.add_argument('--val_zeros_len',
                        type=int,
                        default=50,
                        help='')
    parser.add_argument('--val_pad_len',
                        type=int,
                        default=10,
                        help='')

    parser.add_argument('--test_num_samples',
                        type=int,
                        default=200*64,
                        help='')
    parser.add_argument('--test_copy_len',
                        type=int,
                        default=10,
                        help='')
    parser.add_argument('--test_zeros_len',
                        type=int,
                        default=200,
                        help='')
    parser.add_argument('--test_pad_len',
                        type=int,
                        default=10,
                        help='')
    return parser
