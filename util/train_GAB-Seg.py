import os
import sys
sys.path.append('.')
import time
import shutil
import wandb
import logging
import imgaug as ia
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

from util import util
# from util.data_loading import BasicDataset
from util.Sclera_loader_pma import ScleraDataset as BasicDataset
from util.dice_score import dice_loss
from util.config import get_config
from options.train_options import TrainOptions

from models_pix2pix import create_model
from models_pix2pix import networks
import segmentation_models_pytorch as smp
from deeplab import *

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem
import numpy as np
from medpy import metric  # 必须与测试脚本保持一致
from util.custom_unet import CustomUNet
from util.utils import GT_BceDiceLoss  # 引入论文专属 Loss

def jaccard_index(y_true, y_pred, smooth=1):
    if y_pred.dim() != 2:
        jac = 0.0
        for i in range(y_pred.size()[0]):
            intersection = torch.abs(y_true[i] * y_pred[i]).sum(dim=(-1, -2))
            sum_ = torch.sum(torch.abs(y_true[i]) + torch.abs(y_pred[i]), dim=(-1, -2))
            jac += (intersection + smooth) / (sum_ - intersection + smooth)
        jac = jac / y_pred.size()[0]
    else:
        intersection = torch.abs(y_true * y_pred).sum(dim=(-1, -2))
        sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred), dim=(-1, -2))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    #return (1 - jac) * smooth
    return jac

def jaccard_index_loss(y_true, y_pred, smooth=1):
    return (1 - jaccard_index(y_true, y_pred)) * smooth

    
def dice_coeff(y_true, y_pred, smooth=1e-5):
    """
    计算 Dice 系数 (Dice Coefficient) 作为评估指标
    """
    # 确保张量维度匹配计算
    if y_pred.dim() != 2:
        dice = 0.0
        for i in range(y_pred.size()[0]):
            intersection = torch.sum(y_true[i] * y_pred[i])
            sum_ = torch.sum(y_true[i]) + torch.sum(y_pred[i])
            dice += (2.0 * intersection + smooth) / (sum_ + smooth)
        dice = dice / y_pred.size()[0]
    else:
        intersection = torch.sum(y_true * y_pred)
        sum_ = torch.sum(y_true) + torch.sum(y_pred)
        dice = (2.0 * intersection + smooth) / (sum_ + smooth)
    return dice


class SoftFocalLoss(nn.Module):
    """
    专为已通过 Sigmoid/Clamp 的概率输出设计的 Focal Loss。
    有效解决 99.5% 绝对背景对微小斑点的梯度压制。
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

class SoftFocalLoss(nn.Module):
    """
    专为已通过 Sigmoid/Clamp 的概率输出设计的 Focal Loss。
    有效解决 99.5% 绝对背景对微小斑点的梯度压制。
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
        # 针对前景(1)和背景(0)分别计算交叉熵
        bce_loss = - (target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        # 动态调制系数 (1 - pt)^gamma
        pt = target * pred + (1 - target) * (1 - pred)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class SADLoss_V4(nn.Module):
    def __init__(self, region_weights=[0.5, 0.4, 0.3, 0.2, 0.1], bound_weights=[0.3, 0.2, 0.1]):
        super().__init__()
        # 主分支使用 Focal 替代 BCE
        self.focal = SoftFocalLoss(alpha=0.7, gamma=2.0)
        self.bce = nn.BCELoss(reduction='none')
        self.bce_mean = nn.BCELoss()

        self.region_weights = region_weights
        self.bound_weights = bound_weights

    def get_smooth_distance_penalty(self, mask, max_dist=10):
        """
        生成平滑的指数级空间距离惩罚场。
        距离真实病灶越远，权重平滑递增，避免梯度悬崖。
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # 初始权重为 1
        weight = torch.ones_like(mask)
        dilated = mask

        # 每次膨胀向外扩展，外部区域权重逐步叠加
        for i in range(max_dist):
            dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
            # 使用衰减系数，越往外惩罚力度增加越平缓
            penalty_factor = 0.5 * (0.8 ** i)
            weight += (1.0 - dilated) * penalty_factor

        if weight.dim() == 4 and weight.shape[1] == 1:
            weight = weight.squeeze(1)
        return weight

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

    def forward(self, gt_pre, key_points, masks_pred, true_masks, boundary_points):
        # 保证安全数值区间
        safe_masks_pred = torch.clamp(masks_pred, 1e-6, 1.0 - 1e-6)

        # ==========================================
        # 1. 主区域分支 (Focal + Dice) - 稳大局，保小点
        # ==========================================
        focal_main = self.focal(safe_masks_pred, true_masks)
        dice_main = self.dice_loss(safe_masks_pred, true_masks)
        loss_main = focal_main + 2 * dice_main

        # ==========================================
        # 2. 区域深监督 (BCE + Dice) - 激活浅层视力
        # ==========================================
        loss_region = 0.0
        for i, r_pred in enumerate(gt_pre):
            if i < len(self.region_weights) and self.region_weights[i] > 0:
                safe_r_pred = torch.clamp(r_pred, 1e-6, 1.0 - 1e-6)
                bce_r = self.bce_mean(safe_r_pred, true_masks)
                dice_r = self.dice_loss(safe_r_pred, true_masks)
                loss_region += self.region_weights[i] * (0.5 * bce_r + dice_r)

        # ==========================================
        # 3. 边界深监督 (Smooth DPBL) - 柔性防化服镇压噪声
        # ==========================================
        loss_boundary = 0.0
        dist_weights = self.get_smooth_distance_penalty(true_masks)
        for i, b_pred in enumerate(key_points):
            if i < len(self.bound_weights) and self.bound_weights[i] > 0:
                b_pred_safe = torch.clamp(b_pred, 1e-6, 1.0 - 1e-6)
                bce_loss_map = self.bce(b_pred_safe, boundary_points)
                # 距离权重图与像素级 BCE 相乘
                dpbl = (bce_loss_map * dist_weights).mean()
                dice_b = self.dice_loss(b_pred_safe, boundary_points)
                loss_boundary += self.bound_weights[i] * (0.5 * dpbl + dice_b)

        total_loss = loss_main + loss_region + loss_boundary
        return total_loss

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score_total = 0.0
    jaccard_score_total = 0.0

    # 遍历验证集
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            image, mask_true = batch['image'], batch['mask']

            # 将图像和标签移动到正确的设备和类型
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # 【修改点 1】测试评估时我们不需要中间的辅助预测，所以前两个变量用下划线 "_" 忽略，只拿第三个最终预测
            _, _, mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            
            # 【修改点 2】由于新网络 CustomUNet 的内部最后一步已经做过 torch.sigmoid(out) 了
            # 为了防止“双重 Sigmoid”导致梯度异常，这里必须删掉 torch.sigmoid，直接进行二值化
            mask_pred = (mask_pred > 0.5).float()
            
            # 同时计算并累加 Dice 和 Jaccard 分数
            dice_score_total += dice_coeff(mask_pred.squeeze(), mask_true.squeeze())
            jaccard_score_total += jaccard_index(mask_pred.squeeze(), mask_true.squeeze())

    net.train()
    # 返回一个包含两个平均分的元组: (平均 Dice, 平均 Jaccard)
    return dice_score_total / max(num_val_batches, 1), jaccard_score_total / max(num_val_batches, 1)
    
    # iterate over the validation set
    with torch.no_grad():
        # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for i, batch in enumerate(dataloader):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            
            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            JC_index += jaccard_index(mask_pred.squeeze(), mask_true.squeeze())

    net.train()
    return JC_index / max(num_val_batches, 1)

class SharedDynamicSubset(torch.utils.data.Dataset):
    """
    使用 PyTorch 共享内存技术的动态子集包装器。
    完美支持 num_workers > 0 的多进程数据流。
    """
    def __init__(self, full_dataset, initial_indices):
        self.full_dataset = full_dataset
        # 使用 .share_memory_() 确保主进程的修改能瞬间同步给 4 个后台 worker 进程
        self.indices = torch.tensor(initial_indices, dtype=torch.long).share_memory_()

    def update_indices(self, new_indices):
        # 使用 copy_ 进行在合内存处的原地覆盖（In-place update）
        self.indices.copy_(torch.tensor(new_indices, dtype=torch.long))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx].item()
        return self.full_dataset[real_idx]

top_3_dice = []  
top_3_miou = []
best_save_step = 0  # 新增：用于记录全局最优步数

opt = TrainOptions().parse()   # get training options
config = get_config(opt)
assert opt.cuda_index == int(opt.gpu_ids[0]), 'gpu types should be same'
device = torch.device('cuda:0' if opt.cuda_index == 0 else 'cuda:1')
save_path = './checkpoint_skin/'+'end2end-eye'+str(opt.seg_model)+'-'+str(opt.loss_lambda)+'-'+time.strftime("%Y%m%d-%H%M%S")
if not os.path.exists(save_path):
    os.mkdir(save_path) 
unet_save_path = save_path+'/'+str(opt.seg_model)+'.pkl'  

##### Initialize logging #####
# logger = wandb.init(project='end2end-unet-ISIC', name="unet-200", resume='allow', anonymous='must')
logger = wandb.init(project='end2end-eye', name=str(opt.seg_model)+"-40-"+str(opt.loss_lambda), resume='allow', anonymous='must', mode='offline')
logger.config.update(vars(opt))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

##### create models: pix2pix, unet #####
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
# load pre-trained model
#model_path =  '/data/li/semantic_segmentation1/pix2pix_model/20250207-145805-pix2pix-isic-40' # './pix2pix_model/20230315-055526-pix2pix-UNet-200' './pix2pix_model/20230320-194227-pix2pix-UNet-100' # 
model_path =   '/data/fuzhaocheng/GenSeg-main/pix2pix_pkl/20260503-145819-pix2pix-eye-100'
model.load_model(model_path + '/pix2pix_discriminator.pkl', model_path + '/pix2pix_generator.pkl')

'''
img = Image.open('/home/li/workspace/data/ISIC2018/labels/ISIC_0000019.png').convert('L').resize((256, 256), resample=Image.NEAREST)
img_ndarray = np.asarray(img)
img_ndarray = np.expand_dims(img_ndarray, axis=0)
img_tensor = dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '')torch.as_tensor((img_ndarray / 255.0).copy()).contiguous().to('cuda:1', torch.float).unsqueeze(0)
fake_image = model.netG(img_tensor)

show_image = np.uint8(fake_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').detach().numpy())
img = Image.fromarray(show_image)
img.save('test.png')
# cv2.imwrite('test.png', show_image)
sys.exit()
'''
if opt.seg_model == 'unet': 
    net = CustomUNet(
        in_channels=opt.output_nc, 
        classes=opt.classes
    )
elif opt.seg_model == 'deeplab':
    net = DeepLabV3(num_classes=1)
elif opt.seg_model == 'vit':
    net = ViT_seg(config, img_size=opt.patch_size, num_classes=opt.classes).cuda()
    net.load_from(config)
# net = DeepLab(n_classes=1, n_blocks=[3, 4, 15, 3], atrous_rates=[4, 8, 12], multi_grids=[1, 2, 4], output_stride=8,)
net = net.to(device=device)

##### define optimizer for unet #####
# 采用作者推荐的 AdamW，学习率 0.001，权重衰减 1e-2
optimizer_unet = optim.AdamW(net.parameters(),lr=opt.unet_learning_rate, betas=(0.9, 0.999), 
                             eps=1e-8, weight_decay=1e-1, amsgrad=False)
# 采用作者推荐的余弦退火学习率，T_max=50，最低学习率 eta_min=0.00001
scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=50, eta_min=0.00001)
# scheduler_unet = optim.lr_scheduler.CosineAnnealingLR(optimizer_unet, T_max=500, eta_min=1e-9)
grad_scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

##### prepare dataloader #####
#dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks', 1.0, '_segmentation')
#dataset = BasicDataset(opt.dataroot+'/Images', opt.dataroot+'/Masks')
dataset = BasicDataset(
    images_dir = opt.dataroot + '/Images',
    masks_pix2pix_dir = opt.dataroot + '/Masks_3Class',  
    masks_unet_dir = opt.dataroot + '/Masks' ,
    points_dir = opt.dataroot + '/points'  
)
#dermIS_dataset = BasicDataset('/data2/li/workspace/data/DermIS/Images', '/data2/li/workspace/data/DermIS/Masks', 1.0) # use as the extra dataset
n_test = 20
n_train = 60 
n_val = 20

# 建立 80 张图像的共享元优化池 (Meta-Pool)
# 前 80 张作为训练/验证的交替池，最后 20 张死死锁入测试集作为绝对盲测
meta_pool = np.array(list(range(80))) 
np.random.shuffle(meta_pool) # 初始随机洗牌

init_train_indices = meta_pool[:n_train]
init_val_indices = meta_pool[n_train:n_train+n_val]
test_indices = list(range(80, 120)) # 严格固定最后20张为独立测试集

# 使用我们自定义的共享内存 Subset
train_set = SharedDynamicSubset(dataset, init_train_indices)
val_set = SharedDynamicSubset(dataset, init_val_indices)
test_set = Subset(dataset, test_indices) # 测试集保持不动

loader_args = dict(batch_size=opt.batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
# out-domain test dataloader    

#dermIS_loader = DataLoader(dermIS_dataset, shuffle=False, **loader_args)

# Image Augmenter
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # iaa.Flipud(0.25), # vertical flips

    iaa.CropAndPad(percent=(0, 0.1)), # random crops and pad

    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.        
    iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},),
    iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},),
    iaa.Affine(rotate=(-15, 15),),
    iaa.Affine(shear=(-8, 8),)
], random_order=True) # apply augmenters in random order

fake_trans = transforms.Compose([
    transforms.RandomEqualize(p=0.5),  # histogram equalization
    transforms.RandomPosterize(4, p=1.0),  # Posterization
    transforms.RandomAdjustSharpness(0.3, p=0.5),  # Sharpness adjust
    transforms.RandomAutocontrast(p=0.5),  # contrast adjust
    transforms.ColorJitter(saturation=0.5),  # saturation adjust
])

logging.info('The number of training images = %d' % n_train)
logging.info('The number of validate images = %d' % n_val)
logging.info('The number of test images = %d' % n_test)
logging.info('The number of overall training images = %d' % len(train_set))

#logging.info('The number of DermIS images = %d' % len(dermIS_dataset))

##### Datasets: train_loader, val_loader, test_loader, PH2_loader, DermIS_loader #####
# n_epochs = opt.n_epochs
train_iters = opt.n_epochs     # training iterations
total_iters = 0.0              # the total number of training iterations
val_best_score = 0.0           # the best val score of unet
unet_best_score = 0.0          # the best score of unet

#DermIS_best_score = 0.0        # the best score of DermIS dataset
#损失函数修改与优化

#损失函数修改与优化：引入创新的 SAD-Loss
gt_criterion = SADLoss_V4().to(device)
criterionGAN = networks.GANLoss(opt.gan_mode).to(device)
criterionL1 = torch.nn.L1Loss()

class Generator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        real_image = batch['image_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        fake_image = self.module(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1)
        pred_fake = self.netD(fake_mask_image)
        loss_G_GAN = criterionGAN(pred_fake, True)
        # Second, G(A) = B
        loss_G_L1 = criterionL1(fake_image, real_image) * opt.lambda_L1
        # combine loss and calculate gradients
        loss_G = loss_G_GAN + loss_G_L1
        return loss_G


class Discriminator(ImplicitProblem):
    def training_step(self, batch):
        real_mask = batch['mask_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        real_image = batch['image_pix2pix'].type(torch.cuda.FloatTensor).to(device)
        fake_image = self.netG(real_mask)

        fake_mask_image = torch.cat((real_mask, fake_image), 1) 
        pred_fake = self.module(fake_mask_image.detach())
        loss_D_fake = criterionGAN(pred_fake, False)
        # Real
        real_mask_image = torch.cat((real_mask, real_image), 1)
        pred_real = self.module(real_mask_image)
        loss_D_real = criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D


show_index = 0

def calculate_metrics_strict(pred_mask, true_mask):
    """
    完全对齐 test_isic3.py 的指标计算逻辑
    """
    # 异常处理：真实标签和预测全黑（无斑点，且模型正确识别无斑点）
    if np.sum(pred_mask) == 0 and np.sum(true_mask) == 0:
        return 1.0, 1.0  # dice, iou
    
    # 异常处理：漏诊或误诊（有真实没预测，或有预测没真实）
    elif np.sum(pred_mask) == 0 or np.sum(true_mask) == 0:
        return 0.0, 0.0 
    
    # 正常情况计算
    dice = metric.binary.dc(pred_mask, true_mask)
    iou = metric.binary.jc(pred_mask, true_mask)
    
    return dice, iou

def apply_cgod_module(loss_real, loss_fake, net, loss_lambda):
    # 1. 收集所有多维权重（剔除 bias 和 BatchNorm）
    valid_params = [p for p in net.parameters() if p.requires_grad and len(p.shape) > 1]
    
    # 2. 截取最后 6 个卷积层（覆盖 masks_pred, key_points, gt_pre 的输出头）
    # 这样既能捕捉深层冲突，又不会导致显存溢出 (OOM)
    target_params = valid_params[-6:] 
    
    # 3. 计算多层梯度 (保持 Betty 需要的 create_graph=True)
    g_reals = torch.autograd.grad(loss_real, target_params, retain_graph=True, create_graph=True)
    g_fakes = torch.autograd.grad(loss_fake, target_params, retain_graph=True, create_graph=True)
    
    # 4. 展平并拼接成全局梯度向量
    g_real_flat = torch.cat([g.reshape(-1) for g in g_reals])
    g_fake_flat = torch.cat([g.reshape(-1) for g in g_fakes])
    
    # 5. 计算全局内积
    dot_prod = torch.dot(g_real_flat, g_fake_flat)
    
    norm_r = torch.norm(g_real_flat) + 1e-8
    norm_f = torch.norm(g_fake_flat) + 1e-8
    cos_sim = (dot_prod / (norm_r * norm_f)).item()
    global save_path
    with open(os.path.join(save_path, "gradient_cos_sim_records.txt"), "a") as f:
        f.write(f"{cos_sim}\n")
    # 6. 动态重加权
    if dot_prod < 0:
        norm_real = torch.dot(g_real_flat, g_real_flat) + 1e-8
        dynamic_weight = (1.0 - loss_lambda * (dot_prod / norm_real)).detach()
        dynamic_weight = torch.clamp(dynamic_weight, min=1.0, max=1.2)
        
        return dynamic_weight * loss_real + loss_lambda * loss_fake
    else:
        return loss_real + loss_lambda * loss_fake
    
class Unet(ImplicitProblem):
    def training_step(self, batch):
        images = batch['image'].to(device=device, dtype=torch.float32)
        true_masks_unet = batch['mask'].to(device=device).float()
        true_masks_p2p = batch['mask_pix2pix'].to(device=device).float()
        # ⚠️ 注意：你必须在 Dataloader 中增加提取边界图的逻辑
        boundary_points = batch['points'].to(device=device).float()
        # 1. 计算真实图像的 Loss，解包网络输出
        gt_pre, key_points, masks_pred = net(images)
        # 使用自定义的多尺度深度监督 Loss
        loss = gt_criterion(gt_pre, key_points, masks_pred, true_masks_unet, boundary_points)
        # ---------------- 开始数据增强与假图生成 ----------------
        fake_mask_unet_list = []
        fake_mask_p2p_list = []
        fake_points_list = [] # 用于存放增强后的点图
        # 将 Tensor 转回 CPU numpy 以便 imgaug 处理
        np_unet = true_masks_unet.cpu().numpy()
        np_p2p = true_masks_p2p.cpu().numpy()
        np_points = boundary_points.cpu().numpy() 
        for i in range(images.size(0)): 
            # 锁定随机种子！确保三种掩码发生【完全一致】的变形
            seq_det = seq.to_deterministic()
            
            # [1, 256, 256] -> [256, 256, 1] 适应 imgaug
            m_unet = np.transpose(np_unet[i], (1, 2, 0))
            m_points = np.transpose(np_points[i], (1, 2, 0)) # 【新增】
            
            # Pix2Pix 掩码是 [-1, 1]，先转回 [0, 1] 防止 imgaug 填充灰边
            m_p2p = np.transpose((np_p2p[i] + 1.0) / 2.0, (1, 2, 0))
            
            # 【核心修改】应用完全相同的增强
            m_unet_aug = seq_det.augment_image(m_unet)
            m_p2p_aug = seq_det.augment_image(m_p2p)
            m_points_aug = seq_det.augment_image(m_points) # 【新增】
            
            # P2P 掩码安全地转回 [-1, 1]
            m_p2p_aug = m_p2p_aug * 2.0 - 1.0
            
            # [256, 256, C] 转回 [C, 256, 256] 并保存
            fake_mask_unet_list.append(np.transpose(m_unet_aug, (2, 0, 1)))
            fake_mask_p2p_list.append(np.transpose(m_p2p_aug, (2, 0, 1)))
            fake_points_list.append(np.transpose(m_points_aug, (2, 0, 1))) # 【新增】

        # 重新打包为 GPU Tensor
        fake_mask_unet = torch.tensor(np.array(fake_mask_unet_list)).to(device=device, dtype=torch.float32)
        fake_mask_p2p = torch.tensor(np.array(fake_mask_p2p_list)).to(device=device, dtype=torch.float32)
        fake_boundary_points = torch.tensor(np.array(fake_points_list)).to(device=device, dtype=torch.float32) # 【新增】
        
        # 强制二值化 (0 和 1)
        # imgaug 的旋转或缩放可能会导致边缘产生插值模糊，二值化可以保证掩码和点图的纯净度
        fake_mask = torch.where(fake_mask_unet > 0.1, torch.ones_like(fake_mask_unet), torch.zeros_like(fake_mask_unet))
        fake_boundary_points = torch.where(fake_boundary_points > 0.1, torch.ones_like(fake_boundary_points), torch.zeros_like(fake_boundary_points)) # 【新增】
        
        # 将增强后的【三分类彩色掩码】喂给生成器！
        fake_image = model.netG(fake_mask_p2p)
        # --------------------------------------------------------

        fake_mask = fake_mask.detach()
        fake_boundary_points = fake_boundary_points.detach() # 【新增】
        fake_image = ((fake_image-fake_image.min()) / (fake_image.max()-fake_image.min())).detach()
        fake_image = fake_trans(fake_image.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)) / 255.0
        
        # 2. 计算假图的 Loss
        # 【修改】使用解包和新的损失函数，传入同步增强后的 fake_boundary_points
        fake_gt_pre, fake_key_points, fake_pred = net(fake_image)
        fake_loss = gt_criterion(fake_gt_pre, fake_key_points, fake_pred, fake_mask.float(), fake_boundary_points.float())
        
        # ---------------- Wandb 日志记录部分 ----------------
        global show_index
        show_index += 1
        if show_index % int(max(len(train_set)/10, 1)) == 0:
            ims_dict = {}
            show_image = images[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').numpy()
            
            show_mask = true_masks_unet[0]
            if show_mask.dim() == 2:  
               show_mask = show_mask.unsqueeze(0)  
            show_mask = show_mask.mul(255).permute(1, 2, 0).to('cpu').numpy()
            
            show_fake_image = fake_image[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu').detach().numpy()
            show_fake_mask = fake_mask[0].mul(255).permute(1, 2, 0).to('cpu').numpy()
            
            # 【可选】如果想在 wandb 监控增强后的点图，可以加上这两行
            # show_fake_points = fake_boundary_points[0].mul(255).permute(1, 2, 0).to('cpu').numpy()
            # ims_dict['show_fake_points'] = wandb.Image(show_fake_points)
            
            wandb_image = wandb.Image(show_image)
            ims_dict['show_image'] = wandb_image
            wandb_image = wandb.Image(show_mask)
            ims_dict['show_mask'] = wandb_image
            wandb_image = wandb.Image(show_fake_image)
            ims_dict['show_fake_image'] = wandb_image
            wandb_image = wandb.Image(show_fake_mask)
            ims_dict['show_fake_mask'] = wandb_image
            logger.log(ims_dict)
            show_index = 0
        
        unet_loss = apply_cgod_module(loss, fake_loss, net, opt.loss_lambda)
        return unet_loss


class Arch(ImplicitProblem):
    def training_step(self, batch):
        mask_valid = batch['mask'].to(device=device, dtype=torch.float32)
        image_valid = batch['image'].to(device=device, dtype=torch.float32)
        
        # 【新增】获取边界点图
        points_valid = batch['points'].to(device=device, dtype=torch.float32)

        # 【修改1】像 Unet 类一样，对网络的输出进行三个变量的解包
        gt_pre, key_points, mask_pred = self.unet(image_valid)
        
        # 【修改2】使用全新的 gt_criterion 替换旧的 criterion，并传入完整的 5 个参数
        loss_arch = gt_criterion(gt_pre, key_points, mask_pred, mask_valid, points_valid)
        
        # 注意：GT_BceDiceLoss 内部已经包含了充分的 Dice 惩罚，
        # 所以旧代码里的附加 dice_loss() 不需要再写了。
        return loss_arch


class SSEngine(Engine):

    @torch.no_grad()
    def validation(self):
        net.eval()
        dice_records = []
        jaccard_records = []
        
        for batch in val_loader:
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                _, _, mask_pred = net(image)
                preds = (mask_pred > 0.5).float().cpu().numpy()
                trues = mask_true.cpu().numpy()
                
                for i in range(image.size(0)):
                    pred_np = preds[i].squeeze()
                    true_np = trues[i].squeeze()
                    
                    d, iou = calculate_metrics_strict(pred_np, true_np)
                    dice_records.append(d)
                    jaccard_records.append(iou)

        # 提前四舍五入保留 4 位小数
        val_dice = round(float(np.mean(dice_records)), 4)
        val_jaccard = round(float(np.mean(jaccard_records)), 4)
        
        global top_3_dice, save_path

        # ------------------- 🟢 轨道一：Dice 前三名动态筛选 -------------------
        # 条件：队列不满3个，或者当前分数大于当前前三名中的最小值
        if len(top_3_dice) < 20 or val_dice > min(top_3_dice, key=lambda x: x[0])[0]:
            new_file_name = f"best_dice_step{self.global_step}_score{val_dice:.4f}.pkl"
            new_file_path = os.path.join(save_path, new_file_name)
            torch.save(net.state_dict(), new_file_path)
            
            # 压入新记录并降序排列
            top_3_dice.append((val_dice, self.global_step, new_file_path))
            top_3_dice.sort(key=lambda x: x[0], reverse=True)
            
            # 若超出3个，将最差的（末尾元素）文件实体删除，并从列表中移除
            if len(top_3_dice) > 20:
                _, _, dropped_file_path = top_3_dice.pop()
                if os.path.exists(dropped_file_path):
                    os.remove(dropped_file_path)

        net.train()
        global meta_pool, train_set, val_set
        # 重新随机打乱 80 张图的组合
        np.random.shuffle(meta_pool)
        new_train_idx = meta_pool[:60]
        new_val_idx = meta_pool[60:80]
        # 原地更新共享内存 Tensor，4个后台 worker 进程将在下一次读取时自动应用新划分
        train_set.update_indices(new_train_idx)
        val_set.update_indices(new_val_idx)
        # Betty 会自动且只打印字典里的内容
        return {"unet_val_dice": val_dice, "unet_val_jaccard": val_jaccard}
            


outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=opt.unroll_steps)
engine_config = EngineConfig(
    valid_step=opt.display_freq * opt.unroll_steps,
    train_iters=train_iters,
    roll_back=True,
)
netG = Generator(
    name='netG',
    module=model.netG,
    optimizer=model.optimizer_G,
    train_data_loader=train_loader,
    config=inner_config,
)

netD = Discriminator(
    name='netD',
    module=model.netD,
    optimizer=model.optimizer_D,
    train_data_loader=train_loader,
    config=inner_config,
)

unet = Unet(
    name='unet',
    module=net,
    optimizer=optimizer_unet,
    train_data_loader=train_loader,
    config=inner_config,
)

optimizer_arch = torch.optim.Adam(networks.arch_parameters(), lr=opt.arch_lr, betas=(0.5, 0.999), weight_decay=1e-5)
arch = Arch(
    name='arch',
    module=net,
    optimizer=optimizer_arch,
    train_data_loader=val_loader,
    config=outer_config,
)
target_devices = [device] if isinstance(device, str) else [str(device)]

for problem in [netG, netD, unet, arch]:
    if hasattr(problem, 'device'):
        problem.device = device
    if hasattr(problem, 'devices'):
        problem.devices = target_devices
problems = [netG, netD, unet, arch]

l2u = {netG: [unet], unet: [arch]}
u2l = {arch: [netG]}
# l2u = {}
# u2l = {}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
torch.save(net.state_dict(), save_path+'/unet_final.pkl')
print("\n" + "="*60)
print(f"🎉 Training Finished! ")

print("\n🏆 Top 3 Best Weights for Dice:")
for rank, (score, step, path) in enumerate(top_3_dice):
    print(f"  Rank {rank+1}: Step {step} | Score: {score:.4f} -> {os.path.basename(path)}")
    
print("\n🏆 Top 3 Best Weights for mIoU (Jaccard):")
for rank, (score, step, path) in enumerate(top_3_miou):
    print(f"  Rank {rank+1}: Step {step} | Score: {score:.4f} -> {os.path.basename(path)}")

print("="*60 + "\n")
