import torch

x = torch.empty(2, 3) # 2 by 3 matrix, only 2D and empty

print(x)

example_list = [2, 3, 1]
x = torch.tensor(example_list) # create a tensor from a list
print(x)