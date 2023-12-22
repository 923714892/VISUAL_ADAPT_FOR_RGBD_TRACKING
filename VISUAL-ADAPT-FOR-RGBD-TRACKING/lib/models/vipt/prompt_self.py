import torch
import torch.nn as nn
from .positional_encoding.untied.absolute import Untied2DPositionalEncoder
from .positional_encoding.untied.relative import RelativePosition2DEncoder, generate_2d_concatenated_self_attention_relative_positional_encoding_index
from .utils import token2feature, feature2token
from .self_attention import SelfAttentionBlock


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, spatial_dim=8, qkv_bias=False, attn_drop=0., proj_drop=0., kernel_size=3):
        super(SpatialAttention,self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.spatial_dim = spatial_dim
        self.summation = nn.Linear(dim , spatial_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        self.cross_heads = nn.Linear(self.spatial_dim, self.spatial_dim, bias=qkv_bias)

    def forward(self, x):
        # x: Bx, Cx, W, H
        # mask: [B, N, ] torch.bool
        Bx, N , Cx = x.shape
        space_attn = self.summation(x)
        x_expanded = x.view(Bx, Cx, -1).transpose(1, 2).unsqueeze(-2)
        space_attn = space_attn.view(Bx, self.spatial_dim, -1).transpose(1, 2)
        space_attn_expanded = space_attn.softmax(dim=1).unsqueeze(-1).expand(Bx, -1, self.spatial_dim, Cx)
        spatial_x = space_attn_expanded * x_expanded

        # attn_topk
        tokens = torch.topk(spatial_x, 1, dim=1, largest=True)[0].squeeze(1)

        qkv = self.qkv(tokens).reshape(Bx, self.spatial_dim, 3, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn_spatial_x = (attn @ v).transpose(1, 2).reshape(Bx, -1, Cx)
        attn_spatial_x = self.proj_drop(attn_spatial_x).transpose(1, 2)
        attn_spatial_x = self.cross_heads(attn_spatial_x).transpose(1, 2)
        out = (space_attn @ attn_spatial_x).reshape(Bx, N, Cx).permute(0, 1, 2).contiguous()
        out = self.norm(out) + x
        return out


class Fovea(nn.Module):

    def __init__(self, smooth=False , b = None):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            if b ==5 :
                self.smooth = nn.Parameter(torch.zeros(1) + 5.0)
            elif b == 1 :
                self.smooth = nn.Parameter(torch.zeros(1) + 1.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

class Prompt_block(nn.Module):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        # self.conv0_2 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        # self.conv0_3 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.conv2x2 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea_0 = Fovea(smooth=smooth, b = 1)
        self.fovea_1 = Fovea(smooth = smooth, b = 1)
        self.untied_template_pos_enc_p = Untied2DPositionalEncoder(dim=hide_channel, num_heads=hide_channel, h=8, w=8)
        self.untied_search_pos_enc_p = Untied2DPositionalEncoder(dim=hide_channel, num_heads=hide_channel, h=16, w=16)
        self.rpe_index_p = generate_2d_concatenated_self_attention_relative_positional_encoding_index(
            (8, 8), (16, 16))
        self.rpe_bias_table_p = RelativePosition2DEncoder(hide_channel, self.rpe_index_p.max() + 1)
        self.self_attention_p = SelfAttentionBlock(dim=hide_channel,num_heads=hide_channel)
        self.sparse_attention_a = SpatialAttention(dim=hide_channel, spatial_dim=hide_channel)

        self.drop_out = nn.Dropout(0.1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, RGB_feature_s, RGB_feature_t,D_feature_s,D_feature_t ):
        """ Forward pass with input x. """
        x0 = self.conv0_0(RGB_feature_s)
        x3 = self.conv0_1(RGB_feature_t)
        x1 = self.conv0_0(D_feature_s)
        x2 = self.conv0_1(D_feature_t)
        untied_template_pos_enc_q , untied_template_pos_enc_k= self.untied_template_pos_enc_p()
        untied_search_pos_enc_q ,untied_search_pos_enc_k = self.untied_search_pos_enc_p()
        attn_pos_enc_p = (torch.cat((untied_template_pos_enc_q,untied_search_pos_enc_q),dim=1) @ torch.cat((untied_template_pos_enc_k,untied_search_pos_enc_k),dim=1).transpose(-2, -1)).unsqueeze(0)
        attn_pos_enc_p = attn_pos_enc_p + self.rpe_bias_table_p(self.rpe_index_p)
        x1 = feature2token(x1)
        x2 = feature2token(x2)
        xx = torch.cat((x2,x1),dim=1)
        prompt_self = self.self_attention_p(x = xx, q_ape=None, k_ape=None, attn_pos=attn_pos_enc_p)
        x2 = prompt_self[:,:64,:]
        x1 = prompt_self[:,64:,:]
        prompt_self = self.sparse_attention_a(x1)
        prompt_self = token2feature(prompt_self)
        x1 = token2feature(x1)
        x2 = token2feature(x2)
        x = self.fovea_1(x0) + prompt_self
        x = self.drop_out(x)
        x = self.conv1x1(x)
        x_s = self.fovea_0(x0) + x1
        x_t = self.fovea_0(x3) + x2
        x_s = self.conv2x2(x_s)
        x_t = self.conv2x2(x_t)
        return x ,x_s , x_t

class prompt_embed(nn.Module):
    def __init__(self,dim):
        super(prompt_embed,self).__init__()
        self.dim = dim
        self.prompt_block = Prompt_block(inplanes=self.dim , hide_channel= 8 , smooth= True )


    def forward(self,x , D_feature_t ,  D_feature_s):
        RGB_feature_tt = x[:,:64,:]
        RGB_feature_s = x[:,64:,:]
        RGB_feature_t = RGB_feature_tt
        RGB_feature_s = token2feature(RGB_feature_s)
        RGB_feature_t = token2feature(RGB_feature_t)
        D_feature_t = token2feature(D_feature_t)
        D_feature_s = token2feature(D_feature_s)
        Fuse_feature,D_feature_s,D_feature_t  = self.prompt_block(RGB_feature_s, RGB_feature_t,D_feature_s,D_feature_t)
        Fuse_feature = feature2token(Fuse_feature)
        zero_token = torch.zeros_like(RGB_feature_tt)
        Fuse_feature =torch.cat((zero_token,Fuse_feature),dim=1)
        D_feature_t = feature2token(D_feature_t)
        D_feature_s = feature2token(D_feature_s)
        return Fuse_feature , D_feature_t ,D_feature_s
