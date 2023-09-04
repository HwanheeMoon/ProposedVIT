from torch import nn
import torch
from einops import einsum
from einops.layers.torch import Rearrange
import numpy as np

scale = 64 ** -0.5
num_patches=12

scale = nn.Parameter(scale*torch.ones(12))    
mask = torch.eye(num_patches+1, num_patches+1)
#print(mask)

mask = torch.nonzero((mask == 1), as_tuple=False)
tmp = torch.nonzero((mask == 1), as_tuple=False)

#print(tmp)
np.set_printoptions(suppress=True)
torch.constant_pad_nd
temp_k = np.array([[10,0,0],
                   [0,10,0],
                   [0,0,10],
                   [0,0,10]], dtype=float)  # (4, 3)

temp_v = np.array([[   1,0],
                   [  10,0],
                   [ 100,5],
                   [1000,6]], dtype=float)  # (4, 2)
temp_q = np.array([[0, 10, 0]], dtype=float) 

temp_k = torch.FloatTensor(temp_k)
temp_v = torch.FloatTensor(temp_v)
temp_q = torch.FloatTensor(temp_q)

b = 1
h = 12


dots = torch.mul(temp_q, temp_k),scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1))


print(dots)
#dots[:, :, mask[:, 0], mask[:, 1]] = -987654321

#print(dots)


"""attn = self.attend(dots)
out = einsum('b h i j, b h j d -> b h i d', attn, v) 
out = rearrange(out, 'b h n d -> b n (h d)')"""