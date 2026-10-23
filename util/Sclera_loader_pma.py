import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ScleraDataset(Dataset):
    def __init__(self, images_dir: str, masks_pix2pix_dir: str, masks_unet_dir: str, points_dir: str = None):
        self.images_dir = Path(images_dir)
        self.masks_pix2pix_dir = Path(masks_pix2pix_dir)
        self.masks_unet_dir = Path(masks_unet_dir)
        self.points_dir = Path(points_dir) if points_dir else None
        
        raw_ids = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
        self.ids = sorted(raw_ids, key=lambda x: int(x))
        
        if not self.ids:
            raise RuntimeError(f'在 {images_dir} 中找不到任何图片！请检查路径。')
            
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
        ])
        
        self.mask_unet_transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def crop_all_by_mask(img_pil, mask_p2p_pil, mask_unet_pil):
        # 取消了 points_pil 的同步裁剪，只裁前三个
        mask_np = np.array(mask_p2p_pil)
        
        if len(mask_np.shape) == 3:
            mask_gray = np.any(mask_np > 0, axis=-1)
        else:
            mask_gray = mask_np > 0

        rows = np.any(mask_gray, axis=1)
        cols = np.any(mask_gray, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return img_pil, mask_p2p_pil, mask_unet_pil

        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        margin = 10
        ymin = max(0, ymin - margin)
        ymax = min(mask_np.shape[0], ymax + margin)
        xmin = max(0, xmin - margin)
        xmax = min(mask_np.shape[1], xmax + margin)

        box = (xmin, ymin, xmax, ymax)
        
        cropped_img = img_pil.crop(box)
        cropped_mask_p2p = mask_p2p_pil.crop(box)
        cropped_mask_unet = mask_unet_pil.crop(box)

        return cropped_img, cropped_mask_p2p, cropped_mask_unet

    @staticmethod
    def pad_and_resize(pil_img, target_size=256, is_mask=False):
        w, h = pil_img.size
        ratio = min(target_size/w, target_size/h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        resample_method = Image.NEAREST if is_mask else Image.BILINEAR
        pil_img = pil_img.resize((new_w, new_h), resample=resample_method)
        
        mode = pil_img.mode
        new_img = Image.new(mode, (target_size, target_size), 0)
        new_img.paste(pil_img, ((target_size - new_w) // 2, (target_size - new_h) // 2))
        return new_img

    def __getitem__(self, i):
        name = self.ids[i]
        
        img_path = self.images_dir / f"{name}.png"
        mask_p2p_path = self.masks_pix2pix_dir / f"{name}.png"
        mask_unet_path = self.masks_unet_dir / f"{name}.png"

        img = Image.open(img_path).convert('RGB')
        mask_p2p = Image.open(mask_p2p_path).convert('RGB')
        mask_unet = Image.open(mask_unet_path).convert('L')
        
        # 1. 仅对未处理的原图和掩码进行同步裁剪和缩放
        img, mask_p2p, mask_unet = self.crop_all_by_mask(img, mask_p2p, mask_unet)
        img = self.pad_and_resize(img, target_size=256, is_mask=False)
        mask_p2p = self.pad_and_resize(mask_p2p, target_size=256, is_mask=True)
        mask_unet = self.pad_and_resize(mask_unet, target_size=256, is_mask=True)

        # 2. 直接读取已处理好的 256x256 边界点图，跳过裁剪和缩放！
        if self.points_dir:
            points_path = self.points_dir / f"{name}.png"
            if points_path.exists():
                points = Image.open(points_path).convert('L')
            else:
                # 容错：如果没有点图，生成 256x256 全黑图像
                points = Image.new('L', (256, 256), 0)
        else:
            points = None

        # 3. 转换为 Tensor
        img_tensor = self.transform(img)
        mask_pix2pix_tensor = self.mask_transform(mask_p2p)
        mask_unet_tensor = self.mask_unet_transform(mask_unet) 
        
        return_dict = {
            'image': img_tensor,                       
            'mask': mask_unet_tensor,                  
            'image_pix2pix': img_tensor.clone(),       
            'mask_pix2pix': mask_pix2pix_tensor,       
            'name': name
        }
        
        # 加入 points，供 GT_BceDiceLoss 使用
        if points:
            return_dict['points'] = self.mask_unet_transform(points)

        return return_dict