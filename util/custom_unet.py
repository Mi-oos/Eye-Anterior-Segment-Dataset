import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
import math
import torchvision.models as models

# ==========================================
# 辅助模块 (LayerNorm, Down, ConvLayer 等)
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Group_shuffle_block(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        c_dim = dim_in // 4
        self.share_space1 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space1)
        self.conv1 = nn.Sequential(nn.Conv2d(c_dim, c_dim, 3, 1, 1, groups=c_dim), nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.share_space2 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space2)
        self.conv2 = nn.Sequential(nn.Conv2d(c_dim, c_dim, 3, 1, 1, groups=c_dim), nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.share_space3 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space3)
        self.conv3 = nn.Sequential(nn.Conv2d(c_dim, c_dim, 3, 1, 1, groups=c_dim), nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.share_space4 = nn.Parameter(torch.Tensor(1, c_dim, 8, 8), requires_grad=True)
        nn.init.ones_(self.share_space4)
        self.conv4 = nn.Sequential(nn.Conv2d(c_dim, c_dim, 3, 1, 1, groups=c_dim), nn.GELU(), nn.Conv2d(c_dim, c_dim, 1))
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.ldw = nn.Sequential(nn.Conv2d(dim_in, dim_in, 3, 1, 1, groups=dim_in), nn.GELU(), nn.Conv2d(dim_in, dim_out, 1))

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1 = x1 * self.conv1(F.interpolate(self.share_space1, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x2 = x2 * self.conv2(F.interpolate(self.share_space2, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x3 = x3 * self.conv3(F.interpolate(self.share_space3, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x4 = x4 * self.conv4(F.interpolate(self.share_space4, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x = torch.cat([x2,x4,x1,x3], dim=1)
        x = self.norm2(x)
        return self.ldw(x)

class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
    def forward(self, x):
        gt_pre = self.conv(x)
        x = x + x * torch.sigmoid(gt_pre)
        return x, gt_pre

class Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)
    def forward(self, x):
        boundary = torch.sigmoid(self.conv1(x))
        gt_pre = self.conv2(x)
        return (x + x * boundary + x * torch.sigmoid(gt_pre)), gt_pre, boundary

class Merge(nn.Module):
    def __init__(self, dim_in): super().__init__()
    def forward(self, x1, x2, gt_pre, w): return x1 + x2 + torch.sigmoid(gt_pre) * x2 * w

class Merge2(nn.Module):
    def __init__(self, dim_in): super().__init__()      
    def forward(self, x1, x2, gt_pre, boundary_pre, w1, w2):
        return x1 + x2 + torch.sigmoid(gt_pre) * x2 * w1 + boundary_pre * x2 * w2

# ==========================================
# 主网络：适配 GenSeg 并保留所有中间输出
# ==========================================
class CustomUNet(nn.Module):
    def __init__(self, in_channels=3, classes=1, c_list=None):
        super().__init__()
        self.c_list = [64, 64, 128, 256, 512, 512]
        c_list = self.c_list

        resnet = models.resnet34(pretrained=False)
        weight_path = "/home/zhanglei/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth"
        try:
            resnet.load_state_dict(torch.load(weight_path))
            print(f"✅ 成功加载 ResNet34 离线权重: {weight_path}")
        except Exception as e:
            print(f"⚠️ 权重加载失败，将随机初始化。错误: {e}")

        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.enc_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.enc_layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)         
        self.enc_layer2 = resnet.layer2                                        
        self.enc_layer3 = resnet.layer3                                        
        self.enc_layer4 = resnet.layer4                                        

        self.merge1 = Merge2(c_list[0])
        self.merge2 = Merge2(c_list[1])
        self.merge3 = Merge2(c_list[2])
        self.merge4 = Merge(c_list[3])
        self.merge5 = Merge(c_list[4])
        
        self.decoder1 = nn.Sequential(Group_shuffle_block(c_list[5], c_list[4])) 
        self.decoder2 = nn.Sequential(Group_shuffle_block(c_list[4], c_list[3])) 
        self.decoder3 = nn.Sequential(Group_shuffle_block(c_list[3], c_list[2]))  
        self.decoder4 = nn.Sequential(nn.Conv2d(c_list[2], c_list[1], 3, 1, 1))  
        self.decoder5 = nn.Sequential(nn.Conv2d(c_list[1], c_list[0], 3, 1, 1))  

        self.pred1 = Image_Prediction_Generator(c_list[4])
        self.pred2 = Image_Prediction_Generator(c_list[3])
        self.gate1 = Prediction_Generator(c_list[2])
        self.gate2 = Prediction_Generator(c_list[1])
        self.gate3 = Prediction_Generator(c_list[0])

        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Sequential(nn.Conv2d(c_list[0], classes, kernel_size=1))
        self._init_new_layers()

    def _init_new_layers(self):
        modules_to_init = [
            self.merge1, self.merge2, self.merge3, self.merge4, self.merge5,
            self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5,
            self.pred1, self.pred2, self.gate1, self.gate2, self.gate3,
            self.dbn1, self.dbn2, self.dbn3, self.dbn4, self.dbn5, self.final
        ]
        for m in modules_to_init: m.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None: m.bias.data.zero_()

    def forward(self, x):
        t1 = self.enc_layer0(x)     
        t2 = self.enc_layer1(t1)    
        t3 = self.enc_layer2(t2)    
        t4 = self.enc_layer3(t3)    
        t5 = self.enc_layer4(t4)    
        
        out = t5                    
        
        out = self.decoder1(out)
        out = F.gelu(self.dbn1(out)) 
        out, gt_pre5 = self.pred1(out)
        out = self.merge5(out, t5, gt_pre5, 0.1) 
        gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)

        out = self.decoder2(out)
        out = F.gelu(F.interpolate(self.dbn2(out), scale_factor=(2,2), mode='bilinear', align_corners=True)) 
        out, gt_pre4 = self.pred2(out)
        out = self.merge4(out, t4, gt_pre4, 0.2) 
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        
        out = self.decoder3(out)
        out = F.gelu(F.interpolate(self.dbn3(out), scale_factor=(2,2), mode='bilinear', align_corners=True)) 
        out, gt_pre3, weight1 = self.gate1(out)
        out = self.merge3(out, t3, gt_pre3, weight1, 0.3, 0.1) 
        weight1 = F.interpolate(weight1, scale_factor=8, mode='bilinear', align_corners=True)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        
        out = self.decoder4(out)
        out = F.gelu(F.interpolate(self.dbn4(out), scale_factor=(2,2), mode='bilinear', align_corners=True)) 
        out, gt_pre2, weight2 = self.gate2(out)
        out = self.merge2(out, t2, gt_pre2, weight2, 0.4, 0.2) 
        weight2 = F.interpolate(weight2, scale_factor=4, mode='bilinear', align_corners=True)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        
        out = self.decoder5(out)
        out = F.gelu(F.interpolate(self.dbn5(out), scale_factor=(2,2), mode='bilinear', align_corners=True)) 
        out, gt_pre1, weight3 = self.gate3(out)
        out = self.merge1(out, t1, gt_pre1, weight3, 0.5, 0.3) 
        weight3 = F.interpolate(weight3, scale_factor=2, mode='bilinear', align_corners=True)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        
        out = self.final(out)
        out = F.interpolate(out, scale_factor=(2,2), mode='bilinear', align_corners=True) 

        # 注意：因为 utils.py 的 BCELoss 使用的是 nn.BCELoss，它要求输入是概率值。
        # 因此我们在返回前对所有输出进行 sigmoid 激活。
        gt_pre1 = torch.sigmoid(gt_pre1)
        gt_pre2 = torch.sigmoid(gt_pre2)
        gt_pre3 = torch.sigmoid(gt_pre3)
        gt_pre4 = torch.sigmoid(gt_pre4)
        gt_pre5 = torch.sigmoid(gt_pre5)
        weight1 = torch.sigmoid(weight1)
        weight2 = torch.sigmoid(weight2)
        weight3 = torch.sigmoid(weight3)
        final_out = torch.sigmoid(out)

        # 返回格式：(5个区域预测), (3个边界预测), 最终预测
        return (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), (weight1, weight2, weight3), final_out