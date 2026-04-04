# modules/mlp_heads.py
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn
from typing import Optional
class SimpleMLP(nn.Module):
 def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
 super().__init__()
 self.net = nn.Sequential(
 nn.Linear(d_in, d_hidden),
 nn.GELU(),
 nn.Dropout(dropout),
 nn.Linear(d_hidden, d_out),
 )
 self.reset_parameters()

 def reset_parameters(self):
 for m in self.net:
 if isinstance(m, nn.Linear):
 nn.init.xavier_uniform_(m.weight)
 if m.bias is not None: nn.init.zeros_(m.bias)

 def forward(self, x):
 return self.net(x)

# ===== version B diffusion training MLP =====
class RMSNorm(nn.Module):
 def __init__(self, d, eps=1e-6):
 super().__init__()
 self.weight = nn.Parameter(torch.ones(d))
 self.eps = eps
 def forward(self, x):
 return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class DiffusionStyleMLP(nn.Module):
 """
 x -> RMSNorm -> Linear -> SiLU -> Dropout -> Linear(zero-init) -> +residual
 - layerweight/initialize EDM/DiT/ControlNet head
 - residualoriginal
 """
 def __init__(self, d_in, d_hidden, d_out, dropout=0.0, zero_init=True, residual=True):
 super().__init__()
 self.norm = RMSNorm(d_in)
 self.fc1 = nn.Linear(d_in, d_hidden)
 self.act = nn.SiLU()
 self.drop = nn.Dropout(dropout)
 self.fc2 = nn.Linear(d_hidden, d_out)
 self.residual = residual and (d_in == d_out)

 nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
 if zero_init:
 nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
 else:
 nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

 def forward(self, x):
 y = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
 if self.residual:
 y = x + y
 return y

# === headconsistent ===
class DiffusionMLP(nn.Module):
 """ LLM d_model -> diffusion conditiondimension"""
 def __init__(self, d_in, d_hidden, d_out, variant="diffusion", dropout=0.0):
 super().__init__()
 if variant == "simple":
 self.proj = SimpleMLP(d_in, d_hidden, d_out, dropout)
 else:
 self.proj = DiffusionStyleMLP(d_in, d_hidden, d_out, dropout, zero_init=True, residual=False)
 def forward(self, x): return self.proj(x)

class GNNMLP(nn.Module):
 """ GNN vector -> LLM d_model"""
 def __init__(self, d_in, d_hidden, d_out, variant="diffusion", dropout=0.0):
 super().__init__()
 if variant == "simple":
 self.proj = SimpleMLP(d_in, d_hidden, d_out, dropout)
 else:
 self.proj = DiffusionStyleMLP(d_in, d_hidden, d_out, dropout, zero_init=False, residual=False)
 def forward(self, x): return self.proj(x)

class DiffusionAdapter(nn.Module):
 def __init__(self, in_dim, out_dim):
 super().__init__()
 self.fc = nn.Sequential(
 nn.Linear(in_dim, out_dim),
 nn.LayerNorm(out_dim),
 nn.ReLU(),
 nn.Linear(out_dim, out_dim),
 nn.LayerNorm(out_dim)
 )
 # optionaltraining init 
 for m in self.fc:
 if isinstance(m, nn.Linear):
 nn.init.xavier_uniform_(m.weight)
 nn.init.constant_(m.bias, 0)

 def forward(self, x):
 return self.fc(x)

class MLPAdapter(nn.Module):
 """
 fordifferentdimensionembeddingemptymappinglayerMLPadapter
 
 e.g. GNN moleculeembeddingemptymapping LLM embeddingempty
 """
 def __init__(
 self,
 input_dim: int,
 output_dim: int,
 hidden_dim: Optional[int] = None,
 num_layers: int = 2,
 ):
 """
 Args:
 input_dim (int): inputembeddingdimension
 output_dim (int): outputembeddingdimension
 hidden_dim (Optional[int]): hiddenlayerdimensionif Nonedefault output_dim
 num_layers (int): MLP hiddenlayercountincludinginputoutputlayer
 defaultvalue 2 1
 """
 super().__init__()
 
 if num_layers < 1:
 raise ValueError("`num_layers` must 1")

 if hidden_dim is None:
 hidden_dim = output_dim

 layers = []
 # layerinputdimensionhiddendimension
 layers.append(nn.Linear(input_dim, hidden_dim))
 layers.append(nn.ReLU())

 # hiddenlayer
 for _ in range(num_layers - 1):
 layers.append(nn.Linear(hidden_dim, hidden_dim))
 layers.append(nn.ReLU())
 
 # outputlayerhiddendimensionoutputdimension
 layers.append(nn.Linear(hidden_dim, output_dim))

 self.model = nn.Sequential(*layers)

 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """
 

 Args:
 x (torch.Tensor): input (..., input_dim)

 Returns:
 torch.Tensor: mappingoutput (..., output_dim)
 """
 return self.model(x)
