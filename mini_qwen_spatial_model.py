import math
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

class SpatialVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        state = torch.load(
            "mini_qwen_spatial/models/resnet18_spatial.pt",
            map_location=torch.device('cpu'),
            weights_only=True
        )
        self.encoder.load_state_dict(state)

        for p in self.encoder.parameters():
            p.requires_grad = False

        self.encoder.eval()


    def forward(self, x):
        x = self.encoder(x) #B,512,7,7
        x = x.flatten(start_dim=2, end_dim=3).transpose(1, 2) #B,49,512
        return x

class SpatialProjector(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.GELU(),
            nn.Linear(768, 1536),
        )#B,49,1536

    def forward(self, x):
        x = self.projection(x)
        return x


class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear#128.64
        self.r = r
        self.scaling = alpha / r  #y = Wx + b + (alpha/r)*BAx

        in_features = base_linear.in_features#64,128
        out_features = base_linear.out_features

        for p in self.base.parameters():
            p.requires_grad = False

        self.A = nn.Parameter(
            torch.randn(
                r, in_features,
                device=base_linear.weight.device,
                dtype=torch.float32
            ) * 0.01
        )#8,128

        self.B = nn.Parameter(
            torch.zeros(
                out_features, r,
                device=base_linear.weight.device,
                dtype=torch.float32
            )
        )#64,8


    def forward(self, x):#N, 128
        base_out = self.base(x)#Wx+b

        x_lora = x.to(torch.float32)

        lora_out = (x_lora @ self.A.T) @ self.B.T
        lora_out = lora_out * self.scaling
        return base_out + lora_out.to(base_out.dtype) #N,64




class AttentionProjector(nn.Module):
    def __init__(self, embed_dim=512, out_dim=1536, num_heads=1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(embed_dim, out_dim)
        self.coords_proj = nn.Linear(2, embed_dim)
        self.key_proj = nn.Linear(embed_dim + 2, embed_dim)

        axis = torch.linspace(0, 1, 7)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        position = torch.stack(
            [xx.flatten(), yy.flatten()],
            dim=-1)#49,2

        self.register_buffer("positions", position)



    def forward(self, img_features, coords):#img_fea=(B,49,512), B,2
        B, H, D = img_features.shape
        assert H == 49 and D == 512
        coords = coords.to(img_features.device, img_features.dtype)

        q = self.coords_proj(coords).unsqueeze(1)#B,1,512
        pos = self.positions.unsqueeze(0).expand(B, -1, -1)#49,2--B,49,2

        k = self.key_proj(
            torch.cat([img_features, pos], dim=-1)#B,49,514 ---B,49,512
        )


        out, weights = self.attn(
            q, k, img_features,
            need_weights=True
        )
        out = self.proj(out)

        return out, weights

class AttentionBiasProjector(nn.Module):
    def __init__(self, embed_dim=512, out_dim=1536, num_heads=1, spatial_lambda=10):
        super().__init__()

        self.spatial_lambda = spatial_lambda
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True,
        )
        self.proj = nn.Linear(embed_dim, out_dim)
        self.coords_proj = nn.Linear(2, embed_dim)
        self.key_proj = nn.Linear(embed_dim + 2, embed_dim)

        axis = torch.linspace(0, 1, 7)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        position = torch.stack(
            [xx.flatten(), yy.flatten()],
            dim=-1)#49,2

        self.register_buffer("positions", position)



    def forward(self, img_features, coords):#img_fea=(B,49,512), B,2
        B, H, D = img_features.shape
        assert H == 49 and D == 512
        coords = coords.to(img_features.device, img_features.dtype)

        q = self.coords_proj(coords).unsqueeze(1)#B,1,512
        pos = self.positions.unsqueeze(0).expand(B, -1, -1)#49,2--B,49,2

        k = self.key_proj(
            torch.cat([img_features, pos], dim=-1)#B,49,514 ---B,49,512
        )
        v = img_features

        wq, wk, wv = self.attn.in_proj_weight.chunk(3, dim=0)#512*3,512
        if self.attn.in_proj_bias is not None:
            bq, bk, bv = self.attn.in_proj_bias.chunk(3, dim=0)
        else:
            bq =  bk = bv = None

        Q = F.linear(q, wq, bq)
        K = F.linear(k, wk, bk)
        V = F.linear(v, wv, bv)

        scores = torch.matmul(
            Q, K.transpose(-2, -1),
        ) / math.sqrt(D)

        dist = ((pos - coords.unsqueeze(1)) ** 2).sum(dim=-1).unsqueeze(1)
        #(B,49,2--B,1,2)=B,49,2--B,49--B,1,49
        score = scores - self.spatial_lambda * dist

        weights = torch.softmax(score, dim=-1)
        out = torch.matmul(weights, V)
        out = self.attn.out_proj(out)
        out = self.proj(out)

        return out, weights #B,1,512