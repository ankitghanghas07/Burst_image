import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.transforms as transforms
import math

from torchvision.ops import DeformConv2d
from pytorch_lightning import seed_everything
from einops import rearrange

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader

from datasets.burstsr_dataset import BurstSRDataset

############################################################################################

# Hyperparameters
num_epochs = 300
learning_rate = 3e-4
batch_size = 1

############################################################################################

seed_everything(13)

#####################################

class LayerNorm(pl.LightningModule):
    def __init__(self):
        super(LayerNorm, self).__init__()

    #normalize across the feature channels for BFA
    def forward(self, x): 
        b, f, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c' )
        variance = torch.var(x, dim=-1, keepdim=True)
        x = x/torch.sqrt(variance + 1e-5)
        x = rearrange(x, 'b (h w) c -> b c h w', h = h, w = w)
        return x

####################################
# MDTA : Multi-dconv Head Transposed Attention
class Attention(pl.LightningModule):
    def __init__(self, dim, num_heads, stride, bias):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        self.stride = stride

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.q_dwConv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups= dim, bias = bias)
        self.k_dwConv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups= dim, bias = bias)
        self.v_dwConv = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups= dim, bias = bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size= 1, bias= bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.q_dwConv(self.q_conv(x))
        k = self.k_dwConv(self.k_conv(x))
        v = self.v_dwConv(self.v_conv(x))

        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head = self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head = self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head = self.num_heads)

        q = F.normalize(q, dim = -1)
        k = F.normalize(k, dim = -1)

        attn_map = torch.matmul(q, k.transpose(-2, -1))
        attn_map = attn_map.softmax(dim = -1)

        out = torch.matmul(attn_map, v)

        # out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

################################################
# GDFN

class FeedForward(pl.LightningModule):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_size = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_size, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_size, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x2 = x
        x = F.gelu(x)
        x = self.project_out(x2 * x)
        return x
    
###############################################


# BFA - Burst Feature Attention
class BFA(pl.LightningModule):
    def __init__(self, dim, num_heads, ffn_expansion_factor, stride, bias):
        super(BFA, self).__init__() 
        self.layer_norm = LayerNorm()
        self.attn = Attention(dim = dim, num_heads=num_heads, stride = stride, bias = bias)
        self.ffn = FeedForward(dim = dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        
    
    def forward(self, x):
        x = x + self.attn(self.layer_norm.forward(x))
        x = x + self.ffn(self.layer_norm.forward(x))

        return x

#################################################

# Feature Alignment
class FeatureAlignment(pl.LightningModule):
    def __init__(self, dim, num_features = 48, stride = 1, prev = False):
        super(FeatureAlignment, self).__init__() 

        deform_groups = 8
        kernel_size = 3
        padding = kernel_size//2
        out_channels = deform_groups * 3 * (kernel_size**2)

        act = nn.GELU()

        self.offset_conv = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=padding, bias=False)
        self.deform = DeformConv2d(dim, dim, kernel_size = 3, padding = 2, groups = deform_groups, dilation=2)
        self.feat_enrich = FeatureEnrichment(dim, stride = 1)

        self.bottleneck = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias=False), act)
        if prev:
            self.bottleneck_o = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size = 3, padding = 1, bias=False), act)

    def offset_generation(self, x):
        x_offset, y_offset, mask = torch.chunk(x, 3, dim = 1)
        offset = torch.cat((x_offset, y_offset), dim = 1)
        mask = torch.sigmoid(mask)

        return offset, mask

    def forward(self, x, prev_offset = None):
        b, f, h, w = x.size()
        ref = x[0].unsqueeze(0) # adds a new dim at the specified position

        ref = torch.repeat_interleave(ref, b, dim=0) 

        offset_feat = self.bottleneck(torch.cat([ref, x], dim=1))

        if not prev_offset == None:
            offset_feat = self.bottleneck_o(torch.cat([prev_offset, offset_feat], dim = 1))
        
        offset, mask = self.offset_generation(self.offset_conv(offset_feat))

        # x = x.type(float)
        # offset = offset.type(float)
        # mask = mask.type(float)

        aligned_features = self.deform(x, offset, mask)
        aligned_features = self.feat_enrich(aligned_features)

        return aligned_features, offset_feat


##################################################
# RBFE - reference based feature enrichment
class FeatureEnrichment(pl.LightningModule):
    def __init__(self, dim, stride):
        super(FeatureEnrichment, self).__init__()

        bias = False

        self.encoder = nn.Sequential(*[BFA(dim=dim*2, num_heads=1, stride=stride, ffn_expansion_factor=2.66, bias=bias) for i in range(2)])
        self.feat_squeez = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1), nn.GELU())
        self.feat_expand = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1), nn.GELU())
        self.diff_fusion = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1), nn.GELU())

    def forward(self, x):
        b, f, h, w = x.shape

        ref = x[0].unsqueeze(0)
        ref = torch.repeat_interleave(ref, b, dim=0)

        feat = self.encoder(torch.cat([ref, x], dim=1))
        fused_feat = self.feat_squeez(feat)
        expanded_feat = self.feat_expand(fused_feat)

        diff = feat - expanded_feat
        diff = self.diff_fusion(diff)

        fused_feat = fused_feat + diff

        return fused_feat

####################################################
# Enhanced Deformable Alignment (EDA) 
class EDA(pl.LightningModule):
    def __init__(self, in_channels):
        super(EDA, self).__init__()
        heads = [1,2]
        bias = False
        self.encoder_level1 = nn.Sequential(*[BFA(dim=in_channels, num_heads=1, stride=1, ffn_expansion_factor=2.66, bias=bias) for i in range(2)])
        self.encoder_level2 = nn.Sequential(*[BFA(dim=in_channels, num_heads=1, stride=1, ffn_expansion_factor=2.66, bias=bias) for i in range(2)])

        # downsamples, H -> H/2, W -> W/2 
        self.down1 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)        
        self.down2 = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)

        self.alignment0 = FeatureAlignment(in_channels, prev=True)
        self.alignment1 = FeatureAlignment(in_channels, prev=True)
        self.alignment2 = FeatureAlignment(in_channels)
        self.final_alignment = FeatureAlignment(in_channels, prev=True)

        # upsamples, H -> 2*H, W -> 2*W
        self.offset_up1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        self.offset_up2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)

        self.up1 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)        
        self.up2 = nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)


    def forward(self, x):

        # print("inside eda   " , x.shape)
        enc1 = self.encoder_level1(x)
        # print("bfa1   " , enc1.shape)

        enc1 = self.down1(x)

        enc2 = self.encoder_level2(enc1)
        # print("bfa2    ", enc2.shape)
        enc2 = self.down2(enc2)

        enc2 , offset_feat_enc2 = self.alignment2(enc2)

        dec1 = self.up2(enc2)
        dec1_offset_feat = self.offset_up2(offset_feat_enc2) * 2

        aligned_ecn1, offset_feat_enc1 = self.alignment1(enc1, dec1_offset_feat)
        dec1 = dec1 + aligned_ecn1

        dec0 = self.up1(dec1)
        dec0_offset_feat = self.offset_up1(offset_feat_enc1) * 2

        aligned_x, offset_feat_x = self.alignment0(x, dec0_offset_feat)
        dec0 = dec0 + aligned_x

        aligned_feat, _ = self.final_alignment(dec0, offset_feat_x)

        return aligned_feat

###########################################################
# Burstormer

class Burstormer(pl.LightningModule):
    def __init__(self, in_channels = 4,  mode = 'color', num_features = 48, bias = False):
        super(Burstormer, self).__init__()

        if mode == 'color':
            out_channels = 3
        else: out_channels = 1
        
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=bias)
        self.align = EDA(num_features)

        self.merge = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.up = nn.PixelShuffle(4)

        self.train_loss = nn.L1Loss()

    def forward(self, burst):
        # print("burstormer burst shape  ", burst.shape)
        burst = burst[0]
        burst_size = burst.shape[0]
        burst_feat = self.conv1(burst)
        # print("after first conv   " , burst_feat.shape)
        aligned_burst_feat = self.align(burst_feat)
        # enhanced_img = self.merge(aligned_burst_feat)
        enhanced_img = self.up(aligned_burst_feat)

        enhanced_img = (torch.sum(enhanced_img, dim = 0))/burst_size

        return enhanced_img
    
    def training_step(self, train_batch, batch_idx):
        x, y, flow_vectors, meta_info = train_batch
        pred = self.forward(x)
        # downsample the gt or upsample the pred, before comparision, downsampling factor = 4 is used to 
        # generate the burst from rgb to raw
        pred = pred.clamp(0.0, 1.0)
        loss = self.train_loss(pred, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, flow_vectors, meta_info = val_batch
        pred = self.forward(x)
        pred = pred.clamp(0.0, 1.0)
        mse = nn.MSELoss(pred, y)
        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()
        return psnr
        
    def on_validation_epoch_end(self, validation_step_outputs):
        #validation_steps_outputs is a list of outputs produced in validation step
        psnr = torch.stack(validation_step_outputs).mean()
        self.log('validation_psnr', psnr, on_epoch=True, on_step=False)

    def configure_optimizers(self):        
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6)            
        return [optimizer] #, [lr_scheduler]
    

# if(__name__ == '__main__'):
#     burst = torch.rand(1, 8, 4, 80, 80)
#     model = Burstormer()
#     out = model.forward(burst)
#     print(out.shape)