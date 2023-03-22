import sys

from ..cells.vanilla_lstm_cell import VanillaLSTMCell
from ..cells.batched_cell import BatchCellGRU, BatchCellLSTM
from ..cells.blocked_cell import BlockCellGRU, BlockCellLSTM

def str_to_class(class_name: str):
    """
    Convert sting into class an uninstantiated class object
    :param class_name: name of class to instantiate
    :return: Class object
    """
    return getattr(sys.modules[__name__], class_name)
