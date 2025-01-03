from utils.class_registry import ClassRegistry
from torch_fidelity import calculate_metrics
from torchvision import transforms
from PIL import Image
import torch
import os
from random import shuffle
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

metrics_registry = ClassRegistry()


@metrics_registry.add_to_registry(name="fid/isc/kid")
class FID:
    def get_name(self):
        return "fid"

    def __call__(self, orig_pth, synt_pth, device):
        metrics = calculate_metrics(
            input1=orig_pth,
            input2=synt_pth,
            kid_subset_size=8,
            cuda=(device == "cuda"),
            verbose=False,
            isc=True,  # Inception Score не требуется
            fid=True,  # FID не требуется
            kid=True,  # Вычисление KID
        )

        return {
            'metric/inception_score (isc)': metrics['inception_score_mean'],
            'metric/frechet_inception_distance (fid)': metrics['frechet_inception_distance'],
            'metric/kernel_inception_distance (kid)': metrics['kernel_inception_distance_mean']
        }


@metrics_registry.add_to_registry(name="ms-ssim")
class MS_SSIM:
    def get_name(self):
        return "ms-ssim"

    def __call__(self, orig_pth, synt_pth, *args, **kwargs):
        real_images_files = sorted([os.path.join(orig_pth, f) for f in os.listdir(orig_pth) if f.endswith(('png', 'jpg', 'jpeg'))])
        fake_images_files = sorted([os.path.join(synt_pth, f) for f in os.listdir(synt_pth) if f.endswith(('png', 'jpg', 'jpeg'))])

        shuffle(real_images_files)
        real_images_files = real_images_files[:len(fake_images_files) * 100]

        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=5, betas=(0.0448, 0.3001, 0.2363))

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

        return {'metric/ms-ssim': ms_ssim(fake_imgs, real_imgs)}
