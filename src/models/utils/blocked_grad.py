import torch


class BlockedGradients(torch.autograd.Function):
    """
    Let forward pass just return the input (unmasked) but during back pass block out the grads using the mask.
    """

    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mask = ctx.saved_tensors
        return grad_output * mask, mask * 0.0