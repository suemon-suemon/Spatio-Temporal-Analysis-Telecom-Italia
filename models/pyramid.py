# code modified from https://github.com/Xt-Chen/SARPN/blob/master/models/modules.py

import torch
import torch.nn.functional as F
import torch.nn as nn


class Refineblock(nn.Module):
    def __init__(self, num_features, kernel_size):
        super(Refineblock, self).__init__()
        padding=(kernel_size-1)//2

        self.conv1 = nn.Conv2d(1, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features//2)
        self.conv2 = nn.Conv2d(  
            num_features//2, num_features//2, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features//2)
        self.conv3 = nn.Conv2d(num_features//2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=True)

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.bn1(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv2(x_res)
        x_res = self.bn2(x_res)
        x_res = F.relu(x_res)
        x_res = self.conv3(x_res)

        x2 = x  + x_res
        return 


# Residual Pyramid Decoder
class PyramidDecoder(nn.Module):

    def __init__(self, rpd_num_features = 2048, top_num_features=2048):
        super(PyramidDecoder, self).__init__()

        self.conv = nn.Conv2d(top_num_features, rpd_num_features // 2, kernel_size=1, stride=1, bias=False)                                               
        self.bn = nn.BatchNorm2d(rpd_num_features//2)                                                    

        self.conv5 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features//2, kernel_size=3, stride=1, padding=1, bias=False),    
                                   nn.BatchNorm2d(rpd_num_features//2),                                                             
                                   nn.ReLU(),                                                                                   
                                   nn.Conv2d(rpd_num_features//2, 1, kernel_size=3, stride=1, padding=1, bias=False))               
        rpd_num_features = rpd_num_features // 2                                                                                              
        self.scale5 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                       

        self.conv4 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),                                                                
                                   nn.ReLU(),                                                                                   
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale4 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        
        

        self.conv3 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale3 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        

        self.conv2 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(rpd_num_features),
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))
        self.scale2 = Refineblock(num_features=rpd_num_features, kernel_size=3)                                                     

        rpd_num_features = rpd_num_features // 2                                                                                        

        self.conv1 = nn.Sequential(nn.Conv2d(rpd_num_features, rpd_num_features, kernel_size=3, stride=1, padding=1, bias=False),       
                                   nn.BatchNorm2d(rpd_num_features),                                                                
                                   nn.ReLU(),
                                   nn.Conv2d(rpd_num_features, 1, kernel_size=3, stride=1, padding=1, bias=False))                  

        self.scale1 = Refineblock(num_features=rpd_num_features, kernel_size=3)


    def forward(self, feature_pyramid, fused_feature_pyramid):

        scale1_size = [fused_feature_pyramid[0].size(2), fused_feature_pyramid[0].size(3)]
        scale2_size = [fused_feature_pyramid[1].size(2), fused_feature_pyramid[1].size(3)]
        scale3_size = [fused_feature_pyramid[2].size(2), fused_feature_pyramid[2].size(3)]
        scale4_size = [fused_feature_pyramid[3].size(2), fused_feature_pyramid[3].size(3)]
        scale5_size = [fused_feature_pyramid[4].size(2), fused_feature_pyramid[4].size(3)]
        
        # scale5
        scale5 = torch.cat((F.relu(self.bn(self.conv(feature_pyramid[4]))), fused_feature_pyramid[4]), 1)
        scale5_depth = self.scale5(self.conv5(scale5))

        # scale4
        scale4_res = self.conv4(fused_feature_pyramid[3])
        scale5_upx2 = F.interpolate(scale5_depth, size=scale4_size,
                                    mode='bilinear', align_corners=True)
        scale4_depth = self.scale4(scale4_res + scale5_upx2)

        # scale3 
        scale3_res = self.conv3(fused_feature_pyramid[2])
        scale4_upx2 = F.interpolate(scale4_depth, size=scale3_size,
                                    mode='bilinear', align_corners=True)
        scale3_depth = self.scale3(scale3_res + scale4_upx2)

        # scale2
        scale2_res = self.conv2(fused_feature_pyramid[1])
        scale3_upx2 = F.interpolate(scale3_depth, size=scale2_size,
                                    mode='bilinear', align_corners=True)
        scale2_depth = self.scale2(scale2_res + scale3_upx2)

        # scale1
        scale1_res = self.conv1(fused_feature_pyramid[0])
        scale2_upx2 = F.interpolate(scale2_depth, size=scale1_size,
                                    mode='bilinear', align_corners=True)
        scale1_depth = self.scale1(scale1_res + scale2_upx2)

        scale_depth = [scale5_depth, scale4_depth, scale3_depth, scale2_depth, scale1_depth]

        return scale_depth