import math
import torch
import torch.nn as nn
'''
From Uchida Takumi https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
extended torch.nn module which cusmize connection.
This code base on https://pytorch.org/docs/stable/notes/extending.html

'''
class CustomizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        # If a mask is provided, apply it to the weight
        if mask is not None:
            weight = weight * mask
        
        # Perform matrix multiplication: input * weight.transpose()
        output = input.mm(weight.t())
        
        # If bias is provided, add it to the output
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, weight, bias, mask)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # Compute gradient w.r.t. input if required
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        
        # Compute gradient w.r.t. weight if required
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        
        # Compute gradient w.r.t. bias if required
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask

class iGTPLinear(nn.Module):
    def __init__(self, args, mask, bias=None):
        super(iGTPLinear, self).__init__()
        # Set input and output features based on mask shape
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        
        # Prepare and register the mask as a non-trainable parameter
        self.mask = nn.Parameter(self._prepare_mask(mask), requires_grad=False)

        # Initialize and register weight as a trainable parameter
        self.weight = nn.Parameter(self._initialize_weight(args))
        
        # Initialize and register bias if required
        self.bias = nn.Parameter(self._initialize_bias(args)) if bias else None

        # Apply mask to weight
        self.weight.data = self.weight.data * self.mask

    def _prepare_mask(self, mask):
        # Convert mask to float tensor and transpose
        if isinstance(mask, torch.Tensor):
            return mask.type(torch.float).t()
        return torch.tensor(mask, dtype=torch.float).t()

    def _initialize_weight(self, args):
        # Create empty weight tensor
        w = torch.empty(self.output_features, self.input_features)
        stdv = 1. / math.sqrt(w.size(1))

        # Initialize weight based on specified initialization type
        if args['init_type'] == 'normal':
            return nn.init.normal_(w, mean=0, std=stdv)
        elif args['init_type'] == 'uniform':
            return nn.init.uniform_(w, a=-stdv, b=stdv)
        elif args['init_type'] == 'pos_normal':
            return nn.init.trunc_normal_(w, mean=0, std=stdv, a=0, b=2*stdv)
        elif args['init_type'] == 'pos_uniform':
            return nn.init.uniform_(w, a=0, b=stdv)
        else:
            raise ValueError(f"Unknown initialization type: {args['init_type']}")

    def _initialize_bias(self, args):
        # Create empty bias tensor
        b = torch.empty(self.output_features)
        stdv = 1. / math.sqrt(self.weight.size(1))

        # Initialize bias based on specified initialization type
        if args['init_type'] in ['normal', 'pos_normal']:
            return nn.init.normal_(b, mean=0, std=stdv)
        elif args['init_type'] in ['uniform', 'pos_uniform']:
            return nn.init.uniform_(b, a=-stdv, b=stdv)
        else:
            raise ValueError(f"Unknown initialization type: {args['init_type']}")

    def forward(self, input):
        output = []
        # Iterate through each item in the batch
        for i in range(input.size()[0]):
            # Extract and squeeze the input
            temp_input = input[i, :, :].squeeze()
            # Apply the custom linear function
            temp_output = CustomizedLinearFunction.apply(temp_input, self.weight, self.bias, self.mask)
            output.append(temp_output)
        # Stack the outputs along the first dimension
        return torch.stack(output, dim=0)

    def extra_repr(self):
        # Provide extra information about the module
        return f'input_features={self.input_features}, output_features={self.output_features}, bias={self.bias is not None}'
