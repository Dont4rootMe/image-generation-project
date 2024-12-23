from utils.class_registry import ClassRegistry
from pytorch_fid import fid_score
from torchvision import transforms
from PIL import Image
import torch
import os
from random import shuffle
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid")
class FID:    
    def __call__(self, orig_pth, synt_pth, device):
        fid = fid_score.calculate_fid_given_paths(
            [orig_pth, synt_pth],  # pathes to real and synthetic images
            batch_size=50,         # batch size for calculation of FID
            device=device,         # wether to use GPU or not
            dims=2048              # dimensions of features (Inception Network)
        )
        return fid

@metrics_registry.add_to_registry(name="ms-ssim")
class MS_SSIM:
    def __call__(self, orig_pth, synt_pth):
        real_images_files = sorted([os.path.join(orig_pth, f) for f in os.listdir(orig_pth) if f.endswith(('png', 'jpg', 'jpeg'))])
        fake_images_files = sorted([os.path.join(synt_pth, f) for f in os.listdir(synt_pth) if f.endswith(('png', 'jpg', 'jpeg'))])
        
        shuffle(real_images_files)
        real_images_files = real_images_files[:len(fake_images_files) * 100]

        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
        ])
        
        real_imgs = torch.stack([
            transform(Image.open(real_file).convert("RGB"))
            for real_file in real_images_files
        ], dim=0)
        
        fake_imgs = torch.stack([
            transform(Image.open(fake_file).convert("RGB"))
            for fake_file in fake_images_files
        ], dim=0)
        
        return ms_ssim(fake_imgs, real_imgs)