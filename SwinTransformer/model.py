from turtle import forward
from regex import R
from scipy.fftpack import shift
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement
    
    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1,2))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)
    
    
def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zero(window_size**2, window_size**2)
    
    if upper_lower:
        mask[-displacement * window_size:, :-displacement*window_size] = float('-inf')
        mask[ :-displacement * window_size, -displacement*window_size:] = float('-inf')


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super.__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels*downscaling_factor ** 2, out_channels)
    
    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x



        
class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.patch_partition = PatchMerging(in_channels = in_channels, out_channels = hidden_dimension,
                                            downscaling_factor=downscaling_factor)
        
        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension*4,
                          shifted = False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension*4,
                          shifted = True, window_size=window_size, relative_pos_embedding=relative_pos_embedding)
            ]))        
        
    def forward(self,x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        
        return x.permute(0, 3, 1, 2)
    
class SwinTransformer(nn.Module):
    def __init(self, *, hidden_dim, layers, heads, channels=3, num_classses=1000, head_dim=32, 
               window_size=7, downscaling_factors=(4,2,2,2), relative_pos_embedding=True):
        super().__init__()
        
        self.stage0 = StageModule(in_channels = channels, hidden_dimension = hidden_dim, layers=layers[0],
                                  downscaling_factors=downscaling_factors[0], num_heads = heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        
        self.stage1 = StageModule(in_channels = hidden_dim, hidden_dimension = hidden_dim*2, layers=layers[1],
                        downscaling_factors=downscaling_factors[1], num_heads = heads[1], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        
        self.stage2 = StageModule(in_channels = hidden_dim*2, hidden_dimension = hidden_dim*4, layers=layers[2],
                            downscaling_factors=downscaling_factors[2], num_heads = heads[2], head_dim=head_dim,
                            window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        
        self.stage3 = StageModule(in_channels = channels*4,hidden_dimension = hidden_dim*8, layers=layers[3],
                    downscaling_factors=downscaling_factors[3], num_heads = heads[3], head_dim=head_dim,
                    window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim*8),
            nn.Linear(hidden_dim*8, num_classses)
        )
        
        def forward(self,x):
            x = self.stag0(x)
            x = self.stag1(x)
            x = self.stag2(x)
            x = self.stag3(x)
            
            x = x.mean(dim=[2,3])
            return self.mlp_head(x)
            
    