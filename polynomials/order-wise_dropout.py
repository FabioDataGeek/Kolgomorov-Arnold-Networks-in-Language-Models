from kan import *
from torch.nn.functional import dropout
import torch
from torch import Tensor
from typing import Union


def expand_tensor(tensor, repeats, dimension):
    """
    Expand a tensor along a specific dimension.

    Args:
        tensor (torch.Tensor): The input tensor to be expanded.
        repeats (int): The number of times to repeat along the specified dimension.
        dimension (int): The dimension along which to repeat.

    Returns:
        torch.Tensor: The expanded tensor.
    """
    
    if repeats == 1:
        return tensor
    
    elif repeats < 1:
        raise ValueError("Repeats must be greater than or equal to 1.")
    
    # Get the shape of the tensor
    shape = list(tensor.shape)
    
    # Create a repeat factors list with 1s for all dimensions
    repeat_factors = [1] * len(shape)
    
    # Set the repeat factor for the specified dimension
    repeat_factors[dimension] = repeats
    
    # Repeat the tensor
    return tensor.repeat(*repeat_factors)


def dynamic_mask(boolean_tensor, int_tensor, limit):
    """
    Efficiently generates a mask based on a float32 tensor with 0.0s and 1.0s (boolean_tensor)
    and an integer tensor. Both tensors must have the same shape. The mask will
    have an additional dimension with a size equal to the limit. The mask will
    be True up to the integer value in the corresponding position of the int_tensor
    and False after that, with special behavior based on 0.0s and 1.0s in the boolean_tensor.

    Args:
        boolean_tensor (torch.Tensor): A tensor of float32 values (0.0 or 1.0).
        int_tensor (torch.Tensor): A tensor of integers, same shape as boolean_tensor.
        limit (int): The limit for the new dimension size.

    Returns:
        torch.Tensor: A mask tensor with an additional dimension.
    """
    # Ensure boolean_tensor is of float32 dtype
    assert boolean_tensor.dtype == torch.float32, "boolean_tensor must be a float32 tensor with values 0.0 and 1.0"
    
    # Create a range tensor for the new dimension
    range_tensor = torch.arange(limit, device=boolean_tensor.device).view(1, 1, -1)
    
    # Convert float32 boolean_tensor to a boolean tensor
    boolean_tensor = (boolean_tensor == 1.0).to(torch.bool)  # Ensure boolean dtype
    #boolean_tensor_expanded = boolean_tensor.unsqueeze(-1)  # Expand along the new dimension
    #int_tensor_expanded = int_tensor.unsqueeze(-1)          # Expand int_tensor for broadcasting
    
    # Generate the mask
    mask = torch.where(
        boolean_tensor,                              # Boolean tensor condition
        torch.ones_like(range_tensor, dtype=torch.bool),      # Fill all 1s if True
        range_tensor < int_tensor                   # Fill 1s up to int value, 0s after
    )
    return mask



def order_wise_dropout(input: Tensor, p: float, dimension: int = 0, minimum_order: Union[bool, int] = False, dynamic = False, training: bool = True) -> Tensor:
    """
    Applies dropout along a specified dimension while considering other dimensions.
    
    Args:
        input (Tensor): Input tensor.
        p (float): Dropout probability.
        dimension (int): The dimension along which dropout is applied.
        training (bool): Apply dropout only in training mode.
        
    Returns:
        Tensor: The tensor with dropout applied along the specified dimension.
    """
    number_of_dimensions = input.dim()
    
    if dimension >= number_of_dimensions:
        raise ValueError(
            f"Dimension out of range: input has {number_of_dimensions} dimensions, "
            f"please select a dimension between 0 and {number_of_dimensions - 1}."
        )
    
    if not 0 <= p <= 1:
        raise ValueError(f"Dropout probability must be between 0 and 1, but got {p}.")
    
    if not training or p == 0.0:
        return input  # Return input as-is if not training or p=0
    
    mask_size = [size for i, size in enumerate(input.size()) if i != dimension]
    mask = (torch.rand(mask_size, device=input.device) > p).to(input.dtype)
    mask = mask.unsqueeze(dimension)
    
    # If minimum_order is True, the mask will be applied from the minimum order to the maximum order,
    # otherwise, the mask will be applied from the first order to the maximum order.
    if minimum_order is not False:
        order = input.size(dimension)
        if minimum_order >= order:
            raise ValueError(
                f"Minimum order out of range: input has {input.size(dimension)} orders, "
                f"please select a minimum order between 0 and {input.size(dimension) - 1}."
            )
        
        else:
            # If dynamic is True, the mask will be applied dynamically considering the minimum order, 
            # i.e. if minimum order is 2, the mask will be applied in a random position after second order.
            if dynamic:
                if minimum_order == 0:
                    mask = dynamic_mask(mask, torch.randint(0, order, mask_size, device=input.device).unsqueeze(dimension), order)
                else:
                    mask = torch.cat(
                        (expand_tensor(torch.ones_like(mask), minimum_order, dimension), 
                        dynamic_mask(mask, torch.randint(0, order-minimum_order, mask_size, device=input.device).unsqueeze(dimension), order-minimum_order)), 
                        dim=dimension) 
            else:
                mask = torch.cat((
                    expand_tensor(torch.ones_like(mask), minimum_order, dimension), 
                    expand_tensor(mask, order-minimum_order, dimension)), 
                    dim=dimension
                    )
            
    output = input * mask / (1.0 - p)
    
    return output


# Example usage during training and evaluation
model = torch.nn.Linear(10, 5)  # Example model
input_tensor = torch.randn(10, 768, 5)

# Training phase
model.train()  # Switch to training mode
output_train = order_wise_dropout(input_tensor, p=0.5, dimension=2, training=model.training, minimum_order=3, dynamic=True)

# Evaluation phase
model.eval()  # Switch to evaluation mode
output_eval = order_wise_dropout(input_tensor, p=0.5, dimension=2, training=model.training, minimum_order=3, dynamic=True)

print("Training:", output_train.shape)
print("Evaluation:", output_eval.shape)


print('done!')



