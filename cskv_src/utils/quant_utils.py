import torch

class KVQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, n_bits=8, q_group_size=-1):
        
        # some params, hard-coded here for simplicity
        zero_point=True
        per_tensor=False
        inplace=False

        org_tensor_shape = tensor.shape # [bsz, num_heads, seq_len, compressed_dim]
        if q_group_size > 0:
            assert org_tensor_shape[-1] % q_group_size == 0
            tensor = tensor.reshape(-1, q_group_size)   # [bsz * num_heads * seq_len, compressed_dim]
        if per_tensor:
            tensor = tensor.reshape(1, -1)
        assert tensor.dim() == 2
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)  # [bsz * num_heads * seq_len, 1]
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            scales = max_val / max_int
            zeros = 0
        
        min_value_clipped = scales * (min_int - zeros)  # [bsz * num_heads * seq_len, 1]
        max_value_clipped = scales * (max_int - zeros)
        ctx.q_group_size = q_group_size
        ctx.save_for_backward(tensor, min_value_clipped, max_value_clipped)   

        if inplace:
            (
                (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
            ).mul_(scales)
        else:
            tensor = (
                torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
            ) * scales

        assert torch.isnan(tensor).sum() == 0

        tensor = tensor.reshape(org_tensor_shape)

        # return the quantized tonsor, the scaling factor and the zero point value
        # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
        return tensor   # [bsz, num_heads, seq_len, compressed_dim]

    @staticmethod
    def backward(ctx, grad_output): # [bsz, num_heads, seq_len, compressed_dim]
        input_tensor, min_value_clipped, max_value_clipped = ctx.saved_tensors  # input_tensor.shape = [bsz * num_heads * seq_len, compressed_dim]; min_value_clipped.shape = [bsz * num_heads * seq_len, 1]
        q_group_size = ctx.q_group_size
        grad_input = grad_output.clone().reshape(-1, q_group_size)  # [bsz * num_heads * seq_len, compressed_dim]
        grad_input[input_tensor.ge(max_value_clipped)] = 0
        grad_input[input_tensor.le(min_value_clipped)] = 0
        grad_input = grad_input.reshape(grad_output.shape)
        return grad_input, None, None


class KVQuantizerChannel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, n_bits=8, q_window_size=32):
        
        # some params, hard-coded here for simplicity
        zero_point=True
        inplace=False

        # tensor.shape == [bsz, num_heads, seq_len, compressed_dim]
        assert tensor.dim() == 4    # only apply to kv cache

        seq_len = tensor.shape[2] 
        assert seq_len % q_window_size == 0, f"The quantized seq_len ({seq_len}) must be a multiple of q_window_size ({q_window_size})"  # must be a multiple of q_window_size

        tensor = tensor.transpose(2, 3) # [bsz, num_heads, compressed_dim, seq_len]
        org_tensor_shape_transposed = tensor.shape
        tensor = tensor.reshape(-1, q_window_size)  # [n, q_window_size]

        assert tensor.dim() == 2
        if zero_point:
            max_val = tensor.amax(dim=1, keepdim=True)  
            min_val = tensor.amin(dim=1, keepdim=True)
            max_int = 2**n_bits - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        else:
            max_val = tensor.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (n_bits - 1) - 1
            min_int = -(2 ** (n_bits - 1))
            scales = max_val / max_int
            zeros = 0
        
        min_value_clipped = scales * (min_int - zeros)  
        max_value_clipped = scales * (max_int - zeros)
        ctx.q_window_size = q_window_size
        ctx.save_for_backward(tensor, min_value_clipped, max_value_clipped)   

        if inplace:
            (
                (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
            ).mul_(scales)
        else:
            tensor = (
                torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
            ) * scales

        assert torch.isnan(tensor).sum() == 0

        tensor = tensor.reshape(org_tensor_shape_transposed)    # [bsz, num_heads, compressed_dim, seq_len]
        tensor = tensor.transpose(2, 3) # [bsz, num_heads, seq_len, compressed_dim]

        # return the quantized tonsor, the scaling factor and the zero point value
        # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
        return tensor   # [bsz, num_heads, seq_len, compressed_dim]

    @staticmethod
    def backward(ctx, grad_output): # [bsz, num_heads, seq_len, compressed_dim]
        input_tensor, min_value_clipped, max_value_clipped = ctx.saved_tensors  # input_tensor.shape = [bsz * num_heads * seq_len, compressed_dim]; min_value_clipped.shape = [bsz * num_heads * seq_len, 1]
        q_window_size = ctx.q_window_size
        grad_input = grad_output.clone().transpose(2, 3)    # [bsz, num_heads, compressed_dim, seq_len]
        org_tensor_shape_transposed = grad_input.shape
        grad_input = grad_input.reshape(-1, q_window_size) # [n, window_size] 
        grad_input[input_tensor.ge(max_value_clipped)] = 0
        grad_input[input_tensor.le(min_value_clipped)] = 0
        grad_input = grad_input.reshape(org_tensor_shape_transposed).transpose(2, 3)
        return grad_input, None, None

