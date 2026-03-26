import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name='', scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out) if fan_out > 0 else 0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu': return nn.ReLU(inplace)
    elif act == 'relu6': return nn.ReLU6(inplace)
    elif act == 'leakyrelu': return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu': return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu': return nn.GELU()
    elif act == 'hswish': return nn.Hardswish(inplace)
    else: raise NotImplementedError(f'activation layer [{act}] is not found')

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes if out_planes is not None else in_planes
        if self.in_planes < ratio: ratio = self.in_planes
        self.reduced_channels = max(1, self.in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        return self.sigmoid(out)

class GAG(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super(GAG, self).__init__()
        if kernel_size == 1: groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        
    def forward(self, g, x):
        psi_out = self.activation(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi_out)

class SH(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(SH, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class GIKDWConv(nn.Module):
    """Group-Invariant Kernel Depth-wise Convolution"""
    def __init__(self, in_channels, kernel_size, stride=1, padding=None, bias=False):
        super(GIKDWConv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2 if padding is None else padding
            
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        w0 = self.weight
        w90 = torch.rot90(self.weight, k=1, dims=[2, 3])
        w180 = torch.rot90(self.weight, k=2, dims=[2, 3])
        w270 = torch.rot90(self.weight, k=3, dims=[2, 3])
        
        out0 = F.conv2d(x, w0, self.bias, self.stride, self.padding, groups=self.in_channels)
        out90 = F.conv2d(x, w90, self.bias, self.stride, self.padding, groups=self.in_channels)
        out180 = F.conv2d(x, w180, self.bias, self.stride, self.padding, groups=self.in_channels)
        out270 = F.conv2d(x, w270, self.bias, self.stride, self.padding, groups=self.in_channels)
        
        return (out0 + out90 + out180 + out270) / 4.0

class GMKDC(nn.Module):
    """Group-Invariant Multi-Kernel Depth-wise Convolution"""
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(GMKDC, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                GIKDWConv(self.in_channels, ksize, stride=stride, padding=ksize//2, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for ksize in kernel_sizes
        ])
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs

class MKIR_G(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6'):
        super(MKIR_G, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False), 
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )        
        self.multi_scale_dwconv = GMKDC(self.ex_c, self.kernel_sizes, self.stride, activation, dw_parallel=dw_parallel)

        if self.add:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales
            
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False) 
        
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add:
            dout = 0
            for dwout in dwconv_outs: dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out

def mk_irb_bottleneck_g(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6'):
    convs = []
    xx = MKIR_G(in_c, out_c, s, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation)
    convs.append(xx)
    if n > 1:
        for i in range(1, n):
            xx = MKIR_G(out_c, out_c, 1, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation)
            convs.append(xx)
    return nn.Sequential(*convs)

class MKIRA_G(nn.Module):
    """MKIRA_G = Channel Attention (CA) + Spatial Attention (SA) + MKIR_G"""
    def __init__(self, in_c, out_c, stride=1, n=1, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1,3,5], activation='relu6', ca_ratio=16):
        super(MKIRA_G, self).__init__()
        self.CA = ChannelAttention(in_c, ratio=ca_ratio)
        self.SA = SpatialAttention()
        self.mkir_g = mk_irb_bottleneck_g(in_c, out_c, n, stride, expansion_factor=expansion_factor, dw_parallel=dw_parallel, add=add, kernel_sizes=kernel_sizes, activation=activation)

    def forward(self, x):
        out = self.CA(x) * x
        out = self.SA(out) * out
        return self.mkir_g(out)

class MK_UNet_GIK(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, channels=[8, 16, 32, 64, 128], kernels=[3, 5, 7], expand_ratio=2, depths=[1,1,1,1,1], gag_kernel=3, **kwargs):
        super().__init__()
        
        self.encoder1 = mk_irb_bottleneck_g(in_channels, channels[0], depths[0], 1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels)
        self.encoder2 = mk_irb_bottleneck_g(channels[0], channels[1], depths[1], 1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels)  
        self.encoder3 = mk_irb_bottleneck_g(channels[1], channels[2], depths[2], 1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels)
        self.encoder4 = mk_irb_bottleneck_g(channels[2], channels[3], depths[3], 1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels)
        self.encoder5 = mk_irb_bottleneck_g(channels[3], channels[4], depths[4], 1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels)

        self.AG1 = GAG(F_g=channels[3], F_l=channels[3], F_int=max(channels[3]//2, 1), kernel_size=gag_kernel, groups=max(channels[3]//2, 1))
        self.AG2 = GAG(F_g=channels[2], F_l=channels[2], F_int=max(channels[2]//2, 1), kernel_size=gag_kernel, groups=max(channels[2]//2, 1))
        self.AG3 = GAG(F_g=channels[1], F_l=channels[1], F_int=max(channels[1]//2, 1), kernel_size=gag_kernel, groups=max(channels[1]//2, 1))
        self.AG4 = GAG(F_g=channels[0], F_l=channels[0], F_int=max(channels[0]//2, 1), kernel_size=gag_kernel, groups=max(channels[0]//2, 1))

        self.decoder1 = MKIRA_G(channels[4], channels[3], n=1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels, ca_ratio=16)
        self.decoder2 = MKIRA_G(channels[3], channels[2], n=1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels, ca_ratio=16)
        self.decoder3 = MKIRA_G(channels[2], channels[1], n=1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels, ca_ratio=16)
        self.decoder4 = MKIRA_G(channels[1], channels[0], n=1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels, ca_ratio=8)
        self.decoder5 = MKIRA_G(channels[0], channels[0], n=1, expansion_factor=expand_ratio, add=True, kernel_sizes=kernels, ca_ratio=4)

        self.out1 = SH(channels[2], num_classes)
        self.out2 = SH(channels[1], num_classes)
        self.out3 = SH(channels[0], num_classes)
        self.out4 = SH(channels[0], num_classes)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        t1 = F.max_pool2d(self.encoder1(x), 2, 2)
        t2 = F.max_pool2d(self.encoder2(t1), 2, 2)
        t3 = F.max_pool2d(self.encoder3(t2), 2, 2)
        t4 = F.max_pool2d(self.encoder4(t3), 2, 2)
        out = F.max_pool2d(self.encoder5(t4), 2, 2)

        out = self.decoder1(out)
        out = F.relu(F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=False)) 
        t4_ag = self.AG1(g=out, x=t4)
        out = torch.add(out, t4_ag)

        out = self.decoder2(out)
        out = F.relu(F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=False)) 
        p1 = F.interpolate(self.out1(out), scale_factor=(8,8), mode='bilinear', align_corners=False)
        t3_ag = self.AG2(g=out, x=t3)
        out = torch.add(out, t3_ag)

        out = self.decoder3(out)
        out = F.relu(F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=False)) 
        p2 = F.interpolate(self.out2(out), scale_factor=(4,4), mode='bilinear', align_corners=False)
        t2_ag = self.AG3(g=out, x=t2)
        out = torch.add(out, t2_ag)

        out = self.decoder4(out)
        out = F.relu(F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=False)) 
        p3 = F.interpolate(self.out3(out), scale_factor=(2,2), mode='bilinear', align_corners=False)
        t1_ag = self.AG4(g=out, x=t1)
        out = torch.add(out, t1_ag)

        out = self.decoder5(out)
        out = F.relu(F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=False)) 
        p4 = self.out4(out)

        # Output order [p4, p3, p2, p1] maps to p1, p2, p3, p4 in training script unpack logic.
        return [p4, p3, p2, p1]
