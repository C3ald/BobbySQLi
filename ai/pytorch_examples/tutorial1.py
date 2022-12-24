import torch

example_tensor = torch.rand(3)

print(example_tensor)

#see if cuda is available
cuda = torch.cuda.is_available()
print(cuda)