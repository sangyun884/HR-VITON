import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torchvision import models
import os
from torch.nn.utils import spectral_norm
import numpy as np

import functools

import re


class MTVITON(nn.Module):
    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(MTVITON, self).__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer
        
        self.ClothEncoder = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8
        )
        
        self.PoseEncoder = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )
        
        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')
        
        if opt.warp_feature == 'T1':
            # in_nc -> skip connection + T1, T2 channel
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 2 + ngf * 4 , ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.warp_feature == 'encoder':
            # in_nc -> [x, skip_connection, warped_cloth_encoder_feature(E1)]
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 3, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 3, ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 3, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + input1_nc + input2_nc, output_nc, norm_layer=norm_layer, scale='same')
        if opt.out_layer == 'conv':
            self.out_layer = nn.Sequential(
                ResBlock(ngf + input1_nc + input2_nc, ngf, norm_layer=norm_layer, scale='same'),
                nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
            )
        
        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Flow Conv
        self.flow_conv = nn.Sequential(
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        )
        
        
    def forward(self, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))

        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW)

            if i == 0:
                T1 = E1_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)
                flow = self.flow_conv[i](E4).permute(0, 2, 3, 1)
                flow_list.append(flow)
                
                x = self.conv(T2)
                x = self.SegDecoder[i](x)
                
            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[4 - i](E2_list[4 - i]) 
                
                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)  # upsample n-1 flow
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')
                
                flow = flow + self.flow_conv[i](torch.cat([warped_T1, T2], 1)).permute(0, 2, 3, 1)  # F(n)
                flow_list.append(flow)

                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4-i], warped_T1], 1))
                if self.warp_feature == 'encoder':
                    warped_E1 = F.grid_sample(E1_list[4-i], flow_norm + grid, padding_mode='border')
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4-i], warped_E1], 1))
        
   
        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW)
        
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')
        
        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]

        return flow_list, x, warped_c, warped_cm

class ImprovedMTVITON(nn.Module):
    def __init__(self, opt, input1_nc, input2_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ImprovedMTVITON, self).__init__()
        self.warp_feature = opt.warp_feature
        self.out_layer_opt = opt.out_layer
        
        self.ClothEncoder = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer, scale='down'),  # 128
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),  # 64
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),  # 32
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),  # 16
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')  # 8
        )
        
        self.PoseEncoder = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down'),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer, scale='down')
        )
        
        self.conv = ResBlock(ngf * 4, ngf * 8, norm_layer=norm_layer, scale='same')
        
        if opt.warp_feature == 'T1':
            # in_nc -> skip connection + T1, T2 channel
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 2 + ngf * 4 , ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 2 + ngf * 4 , ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 2 + ngf * 4, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.warp_feature == 'encoder':
            # in_nc -> [x, skip_connection, warped_cloth_encoder_feature(E1)]
            self.SegDecoder = nn.Sequential(
                ResBlock(ngf * 8, ngf * 4, norm_layer=norm_layer, scale='up'),  # 16
                ResBlock(ngf * 4 * 3, ngf * 4, norm_layer=norm_layer, scale='up'),  # 32
                ResBlock(ngf * 4 * 3, ngf * 2, norm_layer=norm_layer, scale='up'),  # 64
                ResBlock(ngf * 2 * 3, ngf, norm_layer=norm_layer, scale='up'),  # 128
                ResBlock(ngf * 1 * 3, ngf, norm_layer=norm_layer, scale='up')  # 256
            )
        if opt.out_layer == 'relu':
            self.out_layer = ResBlock(ngf + input1_nc + input2_nc, output_nc, norm_layer=norm_layer, scale='same')
        if opt.out_layer == 'conv':
            self.out_layer = nn.Sequential(
                ResBlock(ngf + input1_nc + input2_nc, ngf, norm_layer=norm_layer, scale='same'),
                nn.Conv2d(ngf, output_nc, kernel_size=1, bias=True)
            )
        
        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Flow Conv
        # self.flow_conv = nn.ModuleList([
        #     nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        #     nn.Sequential(nn.Conv2d(ngf * 8 + ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                   nn.Conv2d(ngf * 4, 2, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.Sequential(nn.Conv2d(ngf * 8 + ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                   nn.Conv2d(ngf * 4, 2, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.Sequential(nn.Conv2d(ngf * 8 + ngf* 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                   nn.Conv2d(ngf* 4, 2, kernel_size=3, stride=1, padding=1, bias=True)),
        #     nn.Sequential(nn.Conv2d(ngf * 8 + ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True),
        #                   nn.Conv2d(ngf * 4, 2, kernel_size=3, stride=1, padding=1, bias=True)),]
        # )
        
        self.flow_conv = nn.ModuleList([
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        ]
        )
        
        self.bottleneck = nn.Sequential(
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True) , nn.ReLU()),
            nn.Sequential(nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1, bias=True), nn.ReLU()),
        )
        
    def normalize(self, x):
        return x
        # return (x - x.mean()) / x.std()
    def forward(self, input1, input2, upsample='bilinear'):
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1_list.append(self.ClothEncoder[i](input1))
                E2_list.append(self.PoseEncoder[i](input2))
            else:
                E1_list.append(self.ClothEncoder[i](E1_list[i - 1]))
                E2_list.append(self.PoseEncoder[i](E2_list[i - 1]))
        # for i in range(5):
            # print(f"E1_list[{i}].mean() = {E1_list[i].mean()}, E1_list[{i}].std() = {E1_list[i].std()}")
            # print(f"E2_list[{i}].mean() = {E2_list[i].mean()}, E2_list[{i}].std() = {E2_list[i].std()}")
       
            # print("flow_conv[i].mean:",self.flow_conv[i].weight.mean(),
            #       "flow_conv[i].std:",self.flow_conv[i].weight.std())
        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW)

            if i == 0:
                T1 = E1_list[4 - i]  # (ngf * 4) x 8 x 6
                T2 = E2_list[4 - i]
                E4 = torch.cat([T1, T2], 1)
                # print("i:0", "E4.mean():", E4.mean(), "E4.std():", E4.std())
                
                flow = self.flow_conv[i](self.normalize(E4)).permute(0, 2, 3, 1)
                flow_list.append(flow)
                
                x = self.conv(T2)
                x = self.SegDecoder[i](x)
                
            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[4 - i](E2_list[4 - i]) 
                
                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)  # upsample n-1 flow
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                warped_T1 = F.grid_sample(T1, flow_norm + grid, padding_mode='border')
                
                # flow = flow + self.flow_conv[i](torch.cat([warped_T1, T2], 1)).permute(0, 2, 3, 1)  # F(n)
                # flow = flow + self.flow_conv[i](torch.cat([warped_T1, T2, self.flow_conv_before[i-1](x)], 1)).permute(0, 2, 3, 1)  # F(n)
                flow = flow + self.flow_conv[i](self.normalize(torch.cat([warped_T1, self.bottleneck[i-1](x)], 1))).permute(0, 2, 3, 1)  # F(n)
                flow_list.append(flow)
                # print("i:", i, "input.mean()", torch.cat([warped_T1, self.bottleneck[i-1](x)], 1).mean(), "input.std()", torch.cat([warped_T1, self.bottleneck[i-1](x)], 1).std())

                if self.warp_feature == 'T1':
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4-i], warped_T1], 1))
                if self.warp_feature == 'encoder':
                    warped_E1 = F.grid_sample(E1_list[4-i], flow_norm + grid, padding_mode='border')
                    x = self.SegDecoder[i](torch.cat([x, E2_list[4-i], warped_E1], 1))
        
 
        N, _, iH, iW = input1.size()
        grid = make_grid(N, iH, iW)
        
        flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
        flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
        warped_input1 = F.grid_sample(input1, flow_norm + grid, padding_mode='border')
        
        x = self.out_layer(torch.cat([x, input2, warped_input1], 1))

        warped_c = warped_input1[:, :-1, :, :]
        warped_cm = warped_input1[:, -1:, :, :]

        return flow_list, x, warped_c, warped_cm

def make_grid(N, iH, iW):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1)
    grid = torch.cat([grid_x, grid_y], 3).cuda()
    return grid


class ResBlock(nn.Module):
    def __init__(self, in_nc, out_nc, scale='down', norm_layer=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        assert scale in ['up', 'down', 'same'], "ResBlock scale must be in 'up' 'down' 'same'"

        if scale == 'same':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=1, bias=True)
        if scale == 'up':
            self.scale = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(in_nc, out_nc, kernel_size=1,bias=True)
            )
        if scale == 'down':
            self.scale = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=2, padding=1, bias=use_bias)
            
        self.block = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(out_nc)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.scale(x)
        return self.relu(residual + self.block(residual))


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    

class VGGLoss(nn.Module):
    def __init__(self, layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False, Ddownx2=False, Ddropout=False, spectral=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.Ddownx2 = Ddownx2


        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, Ddropout, spectral=spectral)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        
        result = []
        if self.Ddownx2:
            input_downsampled = self.downsample(input)
        else:
            input_downsampled = input
        for i in range(num_D):
            
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, Ddropout=False, spectral=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm if spectral else lambda x: x

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if Ddropout:
                sequence += [[
                self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(nf), nn.LeakyReLU(0.2, True), nn.Dropout(0.5)
            ]]
            else:

                sequence += [[
                    self.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
                
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('no checkpoint')
        raise
    log = model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.cuda()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_D(input_nc, ndf=64, n_layers_D=3, norm='instance', use_sigmoid=False, num_D=2, getIntermFeat=False, gpu_ids=[], Ddownx2=False, Ddropout=False, spectral=False):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat, Ddownx2, Ddropout, spectral=spectral)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD.cuda()
    netD.apply(weights_init)
    return netD

# ----------------------------------------------------------------------------------------------------------------------
#                                             ALIASGenerator-related classes
# ----------------------------------------------------------------------------------------------------------------------
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        pass

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class MaskNorm(nn.Module):
    def __init__(self, feature_channels):
        super(MaskNorm, self).__init__()

        self.in_norm = nn.InstanceNorm2d(feature_channels, affine=False, track_running_stats=False)
        # self.bn_norm = nn.BatchNorm2d(feature_channels, affine=False, track_running_stats=False)

    def forward(self, x, label):
        label = label.detach()
        mn_foreground_region = self.mn(x * label, label)
        mn_background_region = self.mn(x * (1 - label), 1 - label)

        return mn_foreground_region + mn_background_region

    def mn(self, region, mask):
        shape = region.size()

        sum = torch.sum(region, (2, 3))  # (B, C)
        Sr = torch.sum(mask, (2, 3))  # (B, 1)
        # sum = torch.sum(region, (0, 2, 3))  # (B, C) -> (C)
        # Sr = torch.sum(mask, (0, 2, 3))  # (B, 1) -> (1)
        Sr[Sr == 0] = 1
        mu = sum / Sr  # (B, C)

        return self.in_norm(region + (1 - mask) * mu[:, :, None, None]) * torch.sqrt(Sr / (shape[2] * shape[3]))[:, :, None, None]
        # return self.bn_norm(region + (1 - mask) * mu[None, :, None, None]) * torch.sqrt(Sr / (shape[0] * shape[2] * shape[3]))[None, :, None, None]


class ALIAS(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()

        parsed = re.search('(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        pw = ks // 2

        self.noise_var = nn.Parameter(torch.zeros(norm_nc), requires_grad=True)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'mask':
            self.param_free_norm = MaskNorm(norm_nc)
        else:
            raise ValueError('%s is not a recognized param-free norm type in ALIAS' % param_free_norm_type)

    def forward(self, x, seg, misalign_mask=None):
        # Part 1. generate parameter-free normalized activations
        added_noise = (torch.randn(x.shape[0], x.shape[3], x.shape[2], 1).cuda() * self.noise_var).transpose(1, 3)

        if misalign_mask is None:
            normalized = self.param_free_norm(x + added_noise)
        else:
            normalized = self.param_free_norm(x + added_noise, misalign_mask)

        # Part 2. produce scaling and bias conditioned on semantic map
        seg = F.interpolate(seg, size=x.size()[2:], mode='nearest')

        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta

        return out


class ALIASResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, use_rn=False):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        # self.pad = nn.ReflectionPad2d(1)
        # self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0)
        # self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0)
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        normtype_list = ['instance3x3', 'batch3x3', 'mask3x3']
        our_norm_type = 'mask3x3' if use_rn else 'instance3x3'
        # our_norm_type = 'mask3x3' if use_rn else 'batch3x3'
        label_nc = opt.semantic_nc + 1 if use_rn else opt.semantic_nc

        self.alias_0 = ALIAS(our_norm_type, fin, label_nc)
        self.alias_1 = ALIAS(our_norm_type, fmiddle, label_nc)
        if self.learned_shortcut:
            self.alias_s = ALIAS(our_norm_type, fin, label_nc)

    def forward(self, x, seg, misalign_mask=None):
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.size()[2:], mode='nearest')

        x_s = self.shortcut(x, seg, misalign_mask)

        dx = self.alias_0(x, seg, misalign_mask)
        # dx = self.conv_0(self.pad(self.actvn(dx)))
        dx = self.conv_0(self.actvn(dx))
        dx = self.alias_1(dx, seg, misalign_mask)
        # dx = self.conv_1(self.pad(self.actvn(dx)))
        dx = self.conv_1(self.actvn(dx))
        out = x_s + dx

        return out

    def shortcut(self, x, seg, misalign_mask):
        if self.learned_shortcut:
            x_s = self.alias_s(x, seg, misalign_mask)
            x_s = self.conv_s(x_s)
        else:
            x_s = x

        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ALIASGenerator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.num_upsampling_layers = opt.num_upsampling_layers
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)
        self.fc = nn.Conv2d(9, 16 * nf, 3, padding=1)
        self.fc_1 = nn.Conv2d(9, 16, 3, padding=1)
        self.fc_2 = nn.Conv2d(9, 16, 3, padding=1)
        self.fc_3 = nn.Conv2d(9, 16, 3, padding=1)
        self.fc_4 = nn.Conv2d(9, 16, 3, padding=1)
        self.fc_5 = nn.Conv2d(9, 16, 3, padding=1)
        self.fc_6 = nn.Conv2d(9, 16, 3, padding=1)
        self.fc_7 = nn.Conv2d(9, 16, 3, padding=1)

        self.head_0 = ALIASResnetBlock(16 * nf, 16 * nf, opt, use_rn=True)
        self.G_middle_0 = ALIASResnetBlock(16 * nf + 16, 16 * nf, opt, use_rn=True)
        self.G_middle_1 = ALIASResnetBlock(16 * nf + 16, 16 * nf, opt, use_rn=True)
        self.up_0 = ALIASResnetBlock(16 * nf + 16, 8 * nf, opt, use_rn=True)
        self.up_1 = ALIASResnetBlock(8 * nf + 16, 4 * nf, opt, use_rn=True)
        self.up_2 = ALIASResnetBlock(4 * nf + 16, 2 * nf, opt)

        if opt.num_upsampling_layers == 'most':
            self.up_3 = ALIASResnetBlock(2 * nf + 16, 1 * nf, opt)
            self.up_4 = ALIASResnetBlock(1 * nf + 16, nf // 2, opt)
            final_nc = nf // 2
        else:
            self.up_3 = ALIASResnetBlock(2 * nf + 16, 1 * nf, opt)
            final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' % opt.num_upsampling_layers)

        sw = opt.fine_width // (2 ** num_up_layers)
        sh = opt.fine_height // (2 ** num_up_layers)

        return sw, sh

    def forward(self, seg, seg_div, rgb_img, misalign_mask):
        rgb_features = [self.__getattr__('fc_' + str(i))(F.interpolate(rgb_img, size=(self.sh * 2 ** i, self.sw * 2 ** i))) for i in range(1, 8)]

        x = F.interpolate(rgb_img, size=(self.sh, self.sw))
        x = self.fc(x)

        x = self.head_0(x, seg_div, misalign_mask)
        x = self.up(x)
        x = self.G_middle_0(torch.cat((x, rgb_features[0]), dim=1), seg_div, misalign_mask)

        if self.num_upsampling_layers == 'more' or self.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(torch.cat((x, rgb_features[1]), dim=1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_0(torch.cat((x, rgb_features[2]), dim=1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_1(torch.cat((x, rgb_features[3]), dim=1), seg_div, misalign_mask)
        x = self.up(x)
        x = self.up_2(torch.cat((x, rgb_features[4]), dim=1), seg)
        x = self.up(x)
        x = self.up_3(torch.cat((x, rgb_features[5]), dim=1), seg)

        if self.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(torch.cat((x, rgb_features[6]), dim=1), seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


# ----------------------------------------------------------------------------------------------------------------------
#                                             Clothflow-related classes
# ----------------------------------------------------------------------------------------------------------------------
class ClothFlow(nn.Module):
    def __init__(self, input1_nc=3, input2_nc=3, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ClothFlow, self).__init__()

        # Cloth & Cloth Mask Encoder
        self.FeatureExtraction1 = nn.Sequential(
            ResBlock(input1_nc, ngf, norm_layer=norm_layer),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer)
        )

        # Target Cloth Mask (=Person upper) Encoder
        self.FeatureExtraction2 = nn.Sequential(
            ResBlock(input2_nc, ngf, norm_layer=norm_layer),
            ResBlock(ngf, ngf * 2, norm_layer=norm_layer),
            ResBlock(ngf * 2, ngf * 4, norm_layer=norm_layer),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer),
            ResBlock(ngf * 4, ngf * 4, norm_layer=norm_layer)
        )

        # Cloth Conv 1x1
        self.conv1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Person Conv 1x1
        self.conv2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=1, bias=True),
        )

        # Flow Conv
        self.flow_conv = nn.Sequential(
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(ngf * 8, 2, kernel_size=3, stride=1, padding=1, bias=True),
        )

    def forward(self, input1, input2, upsample='nearest'):
        E1_list = []
        E2_list = []
        flow_list = []
        # warped_grid_list = []

        # Feature Pyramid Network
        for i in range(5):
            if i == 0:
                E1_list.append(self.FeatureExtraction1[i](input1))
                E2_list.append(self.FeatureExtraction2[i](input2))
            else:
                E1_list.append(self.FeatureExtraction1[i](E1_list[i - 1]))
                E2_list.append(self.FeatureExtraction2[i](E2_list[i - 1]))

        # Compute Clothflow
        for i in range(5):
            N, _, iH, iW = E1_list[4 - i].size()
            grid = make_grid(N, iH, iW)

            if i == 0:
                T1 = E1_list[4 - i]
                T2 = E2_list[4 - i]
                flow = self.flow_conv[i](torch.cat([T1, T2], 1)).permute(0, 2, 3, 1)
                flow_list.append(flow)
            else:
                T1 = F.interpolate(T1, scale_factor=2, mode=upsample) + self.conv1[4 - i](E1_list[4 - i])
                T2 = F.interpolate(T2, scale_factor=2, mode=upsample) + self.conv2[4 - i](E2_list[4 - i])
                flow = F.interpolate(flow_list[i - 1].permute(0, 3, 1, 2), scale_factor=2, mode=upsample).permute(0, 2, 3, 1)
                flow_norm = torch.cat([flow[:, :, :, 0:1] / ((iW/2 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((iH/2 - 1.0) / 2.0)], 3)
                flow = flow + self.flow_conv[i](torch.cat([F.grid_sample(T1, flow_norm + grid, padding_mode='border'), T2], 1)).permute(0, 2, 3, 1)
                flow_list.append(flow)

        return flow_list