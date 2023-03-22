from typing import Sequence, Union

import torch
import torch.nn as nn
from torchtyping import TensorType

from ..attention import MultiHeadAttention
from ..utils import BlockedGradients


class RIMsCell(nn.Module):

    def __init__(self, cell, d_enc_inp, d_hid_all_rims, num_rims, num_active_rims, use_comm_attn,
                 num_modules_read_input,
                 in_num_head=1, comm_num_head=4,
                 in_d_k=64, in_d_v=400,
                 comm_d_k=16, comm_d_v=16,
                 in_residual=False, comm_residual=True,
                 in_learn_head_weights=False, comm_learn_head_weights=False,
                 in_sparsifier=None, comm_sparsifier=None,
                 device='cpu'
                 ):
        """
        Naming convention of variables:
            - 'in_' refers to input attention params
            - 'comm_' refers to comm attention params
            - 'd_' refers to dimension size
            - 'num_' refers to number
            - '_in' refers to an input obj for the attn module
            - '_out' refers to an output obj of the attn module

        :param cell: Recurrent cell instance
        :param d_enc_inp: Size of the encoded input todo: was 'ninp' in original impl
        :param d_hid_all_rims:  total hidden units for all RIMs todo: was 'nhid' in original impl
        :param num_rims: Number of RIMs in this cell. todo: was 'num_blocks_out' in original impl
        :param num_active_rims: number of input queries (RIMs) to select todo: was 'topkval' in original impl
        :param use_comm_attn: if to use a communication step
        :param num_modules_read_input: todo: rename or just replace with in_num_k_or_v?
        :param in_num_head:
        :param comm_num_head:
        :param in_d_k:
        :param in_d_v:
        :param comm_d_k:
        :param comm_d_v:
        :param in_residual:
        :param comm_residual:
        :param in_learn_head_weights: if the reduction of heads should take the avg (False) or weighted average (True)
        :param comm_learn_head_weights: if the reduction of heads should take the avg (False) or weighted average (True)
        :param in_sparsifier:
        :param comm_sparsifier:
        """

        super(RIMsCell, self).__init__()
        """
        TODO - REMOVE AFTER - JUST JOTTINGS
            in_sparsifier = TopkSparsification(topk=num_blocks_in+1) -> k=1+1 = 2
            comm_sparsifier = TopkSparsification(topk=num_total_rims) -> k=6
            
            block_size_out = hidden size for a single RIM
            
            num_blocks_in
                inp attn: topk (+1)
                    --> NOTE: topk will select the top k queries
                    
            num_blocks_out 
                inp attn: num_q
                comm attn: num_q/k/v, topk
                cell inp dim: att_out * num_blocks_out 
                
            block_size_out
                nhid // num_blocks_out = 600/6 = 100
                inp attn: d_q_in 
                comm attn: d_q/k/v_in or d_mha_out
                
            att_out = block_size_out * 4(?)
            
            topkval = 4
                used in masking
        """
        self.num_active_rims = num_active_rims
        self.use_comm_attn = use_comm_attn
        self.num_rims = num_rims
        d_rim = d_hid_all_rims // num_rims
        self.d_rim = d_rim
        self.d_in_attn_out = in_d_v                 # output dim for the input attention
        in_d_q_in, comm_d_q_in = d_rim, d_rim
        in_d_k_in, comm_d_k_in = d_enc_inp, d_rim
        in_d_v_in, comm_d_v_in = d_enc_inp, d_rim
        in_num_q, comm_num_q = num_rims, num_rims
        in_num_k_or_v, comm_num_k_or_v = num_modules_read_input, num_rims
        d_comm_mha_out = d_rim

        self.d_enc_inp = d_enc_inp
        self.in_num_k_or_v = in_num_k_or_v
        self.in_num_head = in_num_head
        self.in_num_q = in_num_q

        # note - d_mha_out isn't utilised for input attention
        self.inp_attn = MultiHeadAttention(n_head=in_num_head, d_q_in=in_d_q_in, d_k_in=in_d_k_in, d_v_in=in_d_v_in,
                                           d_mha_out=self.d_in_attn_out, d_k=in_d_k, d_v=in_d_v,
                                           num_q=in_num_q, num_k_or_v=in_num_k_or_v, residual=in_residual,
                                           learn_weighting_heads=in_learn_head_weights, sparsifier=in_sparsifier)
        self.cell = cell
        self.comm_attn = MultiHeadAttention(n_head=comm_num_head, d_q_in=comm_d_q_in, d_k_in=comm_d_k_in, d_v_in=comm_d_v_in,
                                            d_mha_out=d_comm_mha_out, d_k=comm_d_k, d_v=comm_d_v,
                                            num_q=comm_num_q, num_k_or_v=comm_num_k_or_v, residual=comm_residual,
                                            learn_weighting_heads=comm_learn_head_weights, sparsifier=comm_sparsifier)
        self.device = device

    def forward(self, input: TensorType["batch", "num_rims", "d_rim"], states: Union[Sequence[torch.Tensor], torch.Tensor, None])\
            -> (Union[Sequence[torch.Tensor], torch.Tensor], torch.Tensor):
        """
        Forward pass on a single timestep
        :param input: single timestep input for cell. The input can also be 2D i.e., [B, d_enc] which is used in the
                        single layer case. If 3D case the shape is [B, n_rim, d_rim].
        :param states: cell states e.g. hx, cx
        :return: new states and mask (of inactive cells)
        """
        bsz = input.shape[0]
        inp_use = input

        # initialise the states for timestep 0
        if states is None:
            states = self.cell.init_states(bsz=bsz)

        """
        Input attention
        Given a set of RIMs as queries and the input as keys and values, inp attn will use MHA to get attention scores
        for RIMs focus on the null and input features. 
        There is NO need to use sparse attention here. 
        """
        # add 'number' dimension to the input (which will represent the number of keys/values)
        inp_use = inp_use.reshape((inp_use.shape[0], 1, self.d_enc_inp))    # [B,d_enc_in] -> [B,n_blk_in=1, d_enc_inp]
        inp_use = inp_use.repeat(1, self.in_num_k_or_v - 1, 1)  # [B, (2-1)* 1, d_enc_inp] = [B, 1, d_enc_inp] = [B, 1, 600]

        # concat the null input
        inp_use = torch.cat([torch.zeros_like(inp_use[:, 0:1, :]), inp_use], dim=1) # [B, 2, 600] = cat([B, 1, d_enc_inp], [B, 1, d_enc_inp] )
        in_attn_q = self.cell.get_hidden_state(states)    # [B, num_rims=6, d_hid=100]
        inp_use, iatt = self.inp_attn(in_attn_q, inp_use, inp_use)  # inp_use = [B, num_q, d_v], iatt= [B, H, num_q, num_k]

        # average over heads
        iatt = iatt.mean(dim=1)       # [B, num_q, num_k]
        ###############################################################################################################
        """
        Create a mask where the top K_A RIMS (queries) have 1s and all other queries have 0s.
        The top K_A is determined by the RIMs (queries) which have the LEAST attention on the null input. 
        The query-key attention matrix (iatt) is used to determine the mask which represents the amount of attention 
        each RIM places on each of the input elements (i.e. the null and input feature). 
        
        From the paper: 
            "Based on the softmax values in (2), we select the top k_A RIMs (out of the total K RIMs) to be activated 
            for each step, which have the least attention on the null input (and thus put the highest attention on 
            the input)"
        
        The top k_A rims will therefore refer to the set of queries not selected by bottomk_indicies.
        """
        new_mask = torch.ones_like(iatt[:, :, 0])    # [B, n_q]
        # get index of the queries where most their attention goes to the null input (hence the bottom k queries)
        bottomk_indices = torch.topk(iatt[:, :, 0], dim=1, sorted=True, largest=True,
                                     k=self.num_rims - self.num_active_rims).indices              # [B, top_k]
        # update the fresh mask of 1's with the 0s representing the RIMs to ignore e.g., new_mask[bottomk_indices] = 0
        # [B, num_q] = index_put_(([B,1], [B, top_k]), [top_k])
        new_mask.index_put_((torch.arange(bottomk_indices.shape[0], device=self.device).unsqueeze(1), bottomk_indices),
                            torch.zeros_like(bottomk_indices, dtype=new_mask.dtype))
        mask = new_mask

        # assert (torch.mean(torch.sum(mask, dim=1)).item() == self.num_active_rims)

        # extend mask to over the projected dimension size
        # [B, n_q] -> [B,n_r, 1] -> [B,n_r, d_r] ; note: num_q == num_total_rims
        mask = mask.reshape((bsz, self.num_rims, 1)).repeat((1, 1, self.d_rim)).detach()
        ###############################################################################################################
        """
        Independent dynamics (i.e. fp on the cell)
        """
        # inp_use [B, n_r, d_in]; each state (and new_state) is in cell specific shape [B, n_r*d_h] or [B*n_r, d_h]
        new_states = self.cell(inp_use, states)
        ###############################################################################################################
        """
        Communication attention
        Block the gradients on the cell states so there's no gradients for the inactive RIMs during backprop. 
        Do MHA using the block grad hx as the query, key and value. 
        The MHA acts as way for the RIMs to attend to all other RIMs. (Imagine a fully connected RIM topology) 
        However, as sparse attention is used, it means that only the top-k RIMs will be able to act as queries and all
        other RIMs will only be able to emit keys and values for the top-k RIMs to read from.
        Apply the residual connection which adds the hidden state for the current timestep to the output of the comm 
        attention
        """
        if self.use_comm_attn:
            hx_new = self.cell.get_hidden_state(new_states)             # [B, n_r, d_rim] (canonical form)
            hx_new_grad_mask = BlockedGradients.apply(hx_new, mask)     # [B, n_r, d_rim]

            # hx_new_att = [B, num_q, d_comm_mha_out], _= [B, num_q, num_k]
            hx_new_att, _ = self.comm_attn(hx_new_grad_mask, hx_new_grad_mask, hx_new_grad_mask)
            # [B, n_rims, d_rim] + [B, num_q, d_comm_mha_out]
            hx_new = hx_new + hx_new_att
            # canonical -> cell specific form
            new_states = self.cell.set_hidden_state(new_states, hx_new)

        """
        Masking of states:
        Use mask from above to determine if the final states of a RIM for the current timestep should use states from 
        the previous timestep (i.e. no changes), or the states post communication attention.  
        This masking stops the cell weights corresponding to the inactive RIMs being updated.
        """
        # states and new_states are in cell specific (2D tensor) form, mask is in canonical (3D tensor) form
        new_states = self.cell.apply_mask(states, new_states, mask)

        # mask = [B, n_r, d_r]
        return new_states, mask
