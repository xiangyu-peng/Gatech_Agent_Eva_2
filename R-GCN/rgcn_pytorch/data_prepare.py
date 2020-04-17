import torch
a = torch.Tensor([1,2,3])
b = []
b.append(a)
b.append(a)
print(torch.cat(b,dim=-1))