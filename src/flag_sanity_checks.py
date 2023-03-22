"""
Asserts to check before the exp is run. More of a sanity check to make sure user hasn't set anything contradictory.
"""


def copying_sanity_checks(args):

    if args.cell == 'VanillaLSTMCell':
        for n_rim, n_active_rim in zip(args.num_rims_per_layer, args.num_active_rims_per_layer):
            assert n_rim == 1
            assert n_active_rim == 1
    else:
        # projected dim of values for inp attn must be same as the dim of a single rim module times 4
        assert args.in_d_v == (args.d_hid_all_rims_per_layer[0] // args.num_rims_per_layer[0]) * 4
