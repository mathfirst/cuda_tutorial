import math, torch, time, os, sys
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
# from helixon_ops import swiglu_cuda
import swiglu_cuda, geglu_cuda
# from swiglu_linear_cpp import SwiGLULinearCpp

'''
https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb
'''


def seed_torch(seed=2023):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # The randomness of hash is not allowed for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        assert d_ff % 2 == 0, "d_ff / 2 is supposed to be an even number."
        self.fc2 = nn.Linear(d_ff // 2, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.fc1(x)
        y1, y2 = y.chunk(2, dim=-1)
        z = self.act(y1) * y2
        return self.fc2(z)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() / d_model * math.log(10000.0))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, customFFN=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        if customFFN:
            self.feed_forward = CustomPositionWiseFeedForward(d_model, d_ff, imp=customFFN)
        else:
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attns = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attns))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, customFFN=False):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        if customFFN:
            self.feed_forward = CustomPositionWiseFeedForward(d_model, d_ff, imp=customFFN)
        else:
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class CustomSwiGLULinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, b1=None, b2=None, bias=None):        
        input_1, input_2 = inputs.chunk(2, dim=-1)
        if b1 is not None and b2 is not None:
            y1 = input_1 + b1
            y2 = input_2 + b2
        else:
            y1 = input_1
            y2 = input_2
        
        output = y1 * F.sigmoid(1.702*y1) * y2 # F.silu(y1) * y2
        output = output.matmul(weight.t())
        if bias is not None:
            output = output + bias
        ctx.save_for_backward(input_1, input_2, weight, b1, b2, bias)
        return output


    @staticmethod
    def backward(ctx, grad_outputs):
        input_1, input_2, weight, b1, b2, bias = ctx.saved_tensors
        x1_shape = input_1.shape
        grad_inputs = torch.empty((*x1_shape[:-1], x1_shape[-1]*2), dtype=input_1.dtype, device=input_1.device)
        grad_input_1, grad_input_2 = grad_inputs.chunk(2, dim=-1)
        sigmoid_input1 = F.sigmoid(1.702 * input_1) # F.sigmoid(input_1)
        grad_output_mm_w, silu_input1 = grad_outputs.matmul(weight), input_1 * sigmoid_input1
        grad_input_1[:] = grad_output_mm_w * sigmoid_input1 * (1.0 + 1.702*(input_1 - silu_input1)) * input_2
        grad_input_2[:] = grad_output_mm_w * silu_input1
        # grad_inputs = torch.cat([grad_input_1, grad_input_2], dim=-1)
        if b1 is not None and b2 is not None:
            grad_b1 = grad_input_1.sum(0)
            grad_b2 = grad_input_2.sum(0)
        else:
            grad_b1 = grad_b2 = None
        grad_w = grad_outputs.transpose(-2, -1).matmul(silu_input1 * input_2)
        if bias is not None:
            grad_b = grad_outputs.sum(0)
        else:
            grad_b = None

        return grad_inputs, grad_w, grad_b1, grad_b2, grad_b
    

class CustomGEGLULinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, b1=None, b2=None, bias=None):        
        input_1, input_2 = inputs.chunk(2, dim=-1)
        if b1 is not None and b2 is not None:
            y1 = input_1 + b1
            y2 = input_2 + b2
        else:
            y1 = input_1
            y2 = input_2
        
        output = F.gelu(y1, approximate='tanh') * y2
        output = output.matmul(weight.t())
        if bias is not None:
            output = output + bias
        ctx.save_for_backward(input_1, input_2, weight, b1, b2, bias)
        return output


    @staticmethod
    def backward(ctx, grad_outputs):
        input_1, input_2, weight, b1, b2, bias = ctx.saved_tensors
        x1_shape = input_1.shape
        grad_inputs = torch.empty((*x1_shape[:-1], x1_shape[-1]*2), dtype=input_1.dtype, device=input_1.device)
        coef = np.sqrt(2.0/np.pi)
        squared_x1 = input_1**2
        d_input_of_tanh = coef * (1 + 0.134145 * squared_x1)  # 
        tanh_polynomial_x1 = F.tanh(coef * input_1 * (1 + 0.044715 * squared_x1))  # tanh( sqrt(2/pi) * (x+0.044715x^3) )
        # d(GELU(x1))/dx1 = 0.5 * (1 + GELU(x1) + (1-GELU(x1)^2) * sqrt(2/pi)*x1*(1+0.14145x1^2))
        one_plus_thah_poly_x1 = 1 + tanh_polynomial_x1
        d_GELU_dx1 = 0.5 * (one_plus_thah_poly_x1 + (1 - tanh_polynomial_x1**2) * d_input_of_tanh * input_1)
        grad_input_1, grad_input_2 = grad_inputs.chunk(2, dim=-1)
        grad_output_mm_w, gelu_out = grad_outputs.matmul(weight), 0.5 * input_1 * one_plus_thah_poly_x1
        grad_input_1[:] = grad_output_mm_w * input_2 * d_GELU_dx1  
        grad_input_2[:] = grad_output_mm_w * gelu_out
        # grad_inputs = torch.cat([grad_input_1, grad_input_2], dim=-1)
        if b1 is not None and b2 is not None:
            grad_b1 = grad_input_1.sum(0)
            grad_b2 = grad_input_2.sum(0)
        else:
            grad_b1 = grad_b2 = None
        grad_w = grad_outputs.transpose(-2, -1).matmul(gelu_out * input_2)
        if bias is not None:
            grad_b = grad_outputs.sum(0)
        else:
            grad_b = None

        return grad_inputs, grad_w, grad_b1, grad_b2, grad_b
    

class CustomSwiGLULinear2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, w, b1=None, b2=None, bias=None):
        '''
        SwiGLU: 
            inputs: x1, x2, b1, b2, 
            outputs: y = Silu(x1 + b1) \odot (x2 + b2)
        linear2: 
            inputs: w, bias
            outputs: z = yw^T + bias
        '''
        x1, x2 = inputs.chunk(2, dim=-1)
        output = torch.empty_like(x1)
        inputs_ = x1, x2, b1, b2, output
        swiglu_cuda.apply_swiglu_fwd(*inputs_)
        output = output.matmul(w.t())
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(x1, x2, b1, b2, w, bias)
        return output


    @staticmethod
    def backward(ctx, grad_outputs):
        x1, x2, b1, b2, w, bias = ctx.saved_tensors
        # x1_silu = F.silu(x1)
        output_activation = torch.empty_like(x1)
        grad_x = torch.empty((*(x1.shape[:-1]), x1.shape[-1]*2), dtype=x1.dtype, device=x1.device)
        dx1, dx2 = grad_x.chunk(2, dim=-1)
        grad_output_mm_w = grad_outputs.matmul(w)
        swiglu_cuda.apply_swiglu_fwd(x1, x2, b1, b2, output_activation)
        swiglu_cuda.apply_swiglu_bwd(x1, x2, b1, b2, grad_output_mm_w, dx1, dx2)
        # grad_x = torch.cat([dx1, dx2], dim=-1)
        if b1 is not None and b2 is not None:
            grad_b1 = dx1.sum(0)
            grad_b2 = dx2.sum(0)
        else:
            grad_b1 = grad_b2 = None
        grad_w = grad_outputs.transpose(-2, -1).matmul(output_activation)
        if bias is not None:
            grad_bias = grad_outputs.sum(0)
        else:
            grad_bias = None
        
        return grad_x, grad_w, grad_b1, grad_b2, grad_bias
    

class CustomSwiGLULinear3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, w, b1=None, b2=None, bias=None):
        '''
        SwiGLU: 
            inputs: x1, x2, b1, b2, 
            outputs: y = Silu(x1 + b1) \odot (x2 + b2)
        linear2: 
            inputs: w, bias
            outputs: z = yw^T + bias
        '''
        x1, x2 = inputs.chunk(2, dim=-1)
        output = torch.empty_like(x1)
        inputs_ = x1, x2, b1, b2, output
        swiglu_cuda.apply_swiglu_fwd(*inputs_)
        output = output.matmul(w.t())
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(x1, x2, b1, b2, w, bias)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x1, x2, b1, b2, w, bias = ctx.saved_tensors
        # x1_silu = F.silu(x1)
        # output_activation = torch.empty_like(x1)
        grad_x = torch.empty((*(x1.shape[:-1]), x1.shape[-1]*2), dtype=x1.dtype, device=x1.device)
        dx1, dx2 = grad_x.chunk(2, dim=-1)
        grad_output_mm_w = grad_outputs.matmul(w)
        x4 = torch.empty_like(x1)
        swiglu_cuda.apply_swiglu_effi_bwd(x1, x2, b1, b2, grad_output_mm_w, dx1, dx2, x4)
        # grad_x = torch.cat([dx1, dx2], dim=-1)
        if b1 is not None and b2 is not None:
            grad_b1 = dx1.sum(0)
            grad_b2 = dx2.sum(0)
        else:
            grad_b1 = grad_b2 = None
        grad_w = grad_outputs.transpose(-2, -1).matmul(x4)
        if bias is not None:
            grad_bias = grad_outputs.sum(0)
        else:
            grad_bias = None
        
        return grad_x, grad_w, grad_b1, grad_b2, grad_bias
    

class CustomGEGLULinear3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, w, b1=None, b2=None, bias=None):
        '''
        GEGLU: 
            inputs: x1, x2, b1, b2, 
            outputs: y = GELU(x1 + b1) \odot (x2 + b2)
        linear2: 
            inputs: w, bias
            outputs: z = yw^T + bias
        '''
        x1, x2 = inputs.chunk(2, dim=-1)
        output = torch.empty_like(x1)
        inputs_ = x1, x2, b1, b2, output
        geglu_cuda.apply_geglu_fwd(*inputs_)
        output = output.matmul(w.t())
        if bias is not None:
            output = output + bias

        ctx.save_for_backward(x1, x2, b1, b2, w, bias)
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        x1, x2, b1, b2, w, bias = ctx.saved_tensors
        # x1_silu = F.silu(x1)
        # output_activation = torch.empty_like(x1)
        grad_x = torch.empty((*(x1.shape[:-1]), x1.shape[-1]*2), dtype=x1.dtype, device=x1.device)
        dx1, dx2 = grad_x.chunk(2, dim=-1)
        grad_output_mm_w = grad_outputs.matmul(w)
        x4 = torch.empty_like(x1)
        geglu_cuda.apply_geglu_effi_bwd(x1, x2, b1, b2, grad_output_mm_w, dx1, dx2, x4)
        # grad_x = torch.cat([dx1, dx2], dim=-1)
        if b1 is not None and b2 is not None:
            grad_b1 = dx1.sum(0)
            grad_b2 = dx2.sum(0)
        else:
            grad_b1 = grad_b2 = None
        # geglu_cuda.apply_geglu_fwd(x1, x2, b1, b2, x4)
        grad_w = grad_outputs.transpose(-2, -1).matmul(x4)
        if bias is not None:
            grad_bias = grad_outputs.sum(0)
        else:
            grad_bias = None
        
        return grad_x, grad_w, grad_b1, grad_b2, grad_bias


# from torch.autograd import gradcheck

# # gradcheck takes a tuple of tensors as input, check if your gradient
# # evaluated with these tensors are close enough to numerical
# # approximations and returns True if they all verify this condition.
# device = 'cuda'
# inputs = (torch.randn(20,20,dtype=torch.double,requires_grad=True).to(device), torch.randn(30,10,dtype=torch.double,requires_grad=True).to(device))
# linear = CustomSwiGLULinear2.apply#(inputs[0], inputs[1])
# test = gradcheck(linear, inputs, eps=1e-6, atol=1e-4)
# print(test)


from torch.nn import Module, Parameter, init
from torch import Tensor

class SwiGLULinearCuda(Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b1 = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b2 = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.b1, -bound, bound)
            init.uniform_(self.b2, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return CustomSwiGLULinear3.apply(input, self.weight, self.b1, self.b2, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    

class GEGLULinearCuda(Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b1 = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b2 = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.b1, -bound, bound)
            init.uniform_(self.b2, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return CustomGEGLULinear3.apply(input, self.weight, self.b1, self.b2, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    

class SwiGLULinear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b1 = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b2 = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return CustomSwiGLULinear.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
class GEGLULinear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b1 = Parameter(torch.empty(out_features, **factory_kwargs))
            self.b2 = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('b1', None)
            self.register_parameter('b2', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return CustomGEGLULinear.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class CustomPositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, bias=False, imp='custom'):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        if imp == 'custom':
            self.fc2 = SwiGLULinear(d_ff // 2, d_model, bias=bias)
        elif imp == 'custom-GEGLU':
            self.fc2 = GEGLULinear(d_ff // 2, d_model, bias=bias)
        elif imp == 'cuda':
            self.fc2 = SwiGLULinearCuda(d_ff // 2, d_model, bias=bias)
        elif imp == 'GEGLU-cuda':
            self.fc2 = GEGLULinearCuda(d_ff // 2, d_model, bias=bias)  
        elif imp == 'cpp':
            self.fc2 = SwiGLULinearCpp(d_ff // 2, d_model, bias=bias)
        else:
            raise NotImplementedError(f"Unknown implementation: {customFFN}")

    def forward(self, x):
        return self.fc2(self.fc1(x))


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_layers, d_ff, max_seq_length, dropout, customFFN=False):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout, customFFN) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout, customFFN) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_length = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(1, seq_length, seq_length, device=src.device), diagonal=0).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.position_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.position_encoding(self.decoder_embedding(tgt)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output



class Test(nn.Module):
    '''
    What is under the hood of nn.Linear: input * W^T + b
    W: [out_d, in_d]
    b: [out_d, ]
    '''

    def __init__(self, in_d, out_d, bias=False):
        super().__init__()
        self.bias = bias
        self.linear = nn.Linear(in_d, out_d, bias=bias)

    def forward(self, x):
        w = self.linear.weight.data
        print('w', w)
        if self.bias:
            b = self.linear.bias.data
            print('b', b)
            xWT = torch.mm(x, w.t()) + b
        else:
            xWT = torch.mm(x, w.t())
        y = self.linear(x)
        print('y', y)
        print(torch.allclose(xWT, y), xWT)
        return y
    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        device = 'cpu'
    # device = 'cuda:7'
    print("device", device)
    seed_torch(2023)
    src_vocab_size = 50000
    tgt_vocab_size = 50000
    d_model = 768 * 2
    num_heads = 8
    num_layers = 6

    max_seq_length = 12
    dropout = 0.1
    customFFN = 'custom' # False # 'F' #'custom' # 'GEGLU-cuda' # 'custom-GEGLU' #  'cuda'
    d_ff = d_model * 4
    if customFFN == 'custom':
        # print(f"We use custom approximate GELU(1.702) Linear without custom cuda code. d_model: {d_model}, d_FFN: {d_ff}")        
        print(f"We use custom SwiGLU Linear without custom cuda code. d_model: {d_model}, d_FFN: {d_ff}")
        assert d_ff % 2 == 0, f"d_ff ({d_ff}) is supposed to be an even number."
    elif customFFN == 'cuda':
        print('We are using custom cuda code written by Rui Wang.')
    elif customFFN == 'custom-GEGLU':
        print('We use custom-GEGLU code.')
    elif customFFN == 'GEGLU-cuda':
        print('We use GEGLU-cuda code.')
    elif customFFN == 'cpp':
        print('We use cpp code.')
    else:
        # d_ff = d_model * 4
        print(f"We do not use custom SwiGLU Linear. d_model: {d_model}, d_FFN: {d_ff}")

    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers,
                              d_ff, max_seq_length, dropout, customFFN).to(device)#.to(torch.bfloat16)
    bs = 48
    src_data = torch.randint(1, src_vocab_size, (bs, max_seq_length)).to(device)
    tgt_data = torch.randint(1, tgt_vocab_size, (bs, max_seq_length)).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
    transformer.train()
    t0 = time.time()
    for epoch in range(30):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1])
        
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        # print(epoch, loss.item())
        loss.backward()
        optimizer.step()
        if epoch % 10 == 9:
            print(f"Epoch: {epoch}, loss: {loss.item():.2f}, time consumed: {time.time() - t0:.2f}")
            print(f"max_memory_allocated: {torch.cuda.max_memory_allocated(device=device)/(1024**3):.2f} GiB")
            # for k, v in torch.cuda.memory_stats(device).items():
            #     print(k, v)

    print(f'time consumed: {time.time() - t0:.2f}')
