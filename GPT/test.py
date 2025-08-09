import torch
import torch.nn as nn
import torch.nn.functional as F

token = 6
# # # batch, token, head, hidden
a = torch.rand((1, token, 5, 3)) 
# # b = torch.zeros((1, 6, 5, 3))
# # c = torch.zeros((1, 6, 5, 3))

# # x = torch.matmul(a, b.transpose(2, 3))
# # x = F.softmax(x, dim=1)
# # x = torch.matmul(x, c)

# # x = x.view(1, 6, -1)
# # print(x.shape)

# mask = torch.ones((token, token))
# mask = torch.triu(mask, diagonal=1)
# mask = mask == 0
# mask = mask *1

# print(mask)
hidden = 512
eps = 1e-6

scale = nn.Parameter(torch.ones(hidden))

a = scale * a * torch.rsqrt(torch.mean(a**2, dim=-1, keepdim=True) + eps)
print(ascii)