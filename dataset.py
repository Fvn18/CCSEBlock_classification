from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from torchvision.transforms import RandAugment
from torchvision.transforms import functional as F
from utils import worker_init_fn


class GenericDataLoader:
    def __init__(self, config=None):
        self.config = config or {}
        get = self.config.get
        
        self.dataset_name = get('dataset', 'fer2013') 
        self.horizontal_flip_prob = get('horizontal_flip_prob', 0.6)
        self.random_rotation_degrees = get('random_rotation_degrees', 20)
        self.random_erasing_prob = get('random_erasing_prob', 0.07)
        self.color_jitter_prob = get('color_jitter_prob', 0.7)
        self.random_affine_prob = get('random_affine_prob', 0.6)
        self.gaussian_blur_prob = get('gaussian_blur_prob', 0.5)
        self.resize_size = get('resize_size', 72)
        self.crop_size = get('crop_size', 64)
        self.color_jitter_brightness = get('color_jitter_brightness', 0.35)
        self.color_jitter_contrast = get('color_jitter_contrast', 0.35)
        self.color_jitter_saturation = get('color_jitter_saturation', 0.25)
        self.random_affine_translate = get('random_affine_translate', (0.12, 0.12))
        self.random_affine_scale = get('random_affine_scale', (0.92, 1.08))
        self.gaussian_blur_sigma = get('gaussian_blur_sigma', (0.1, 0.4))

        self.mean = get('mean', [0.5])
        self.std = get('std', [0.5])
        
        self.use_tta = get('use_tta', False)
        self.tta_angles = get('tta_angles', [5.0, -5.0, 2.5, -2.5])
        
        self.seed = get('seed', 42)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def _stack_crops_test(self, crops):
        return torch.stack([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])(crop) for crop in crops
        ])

    def _stack_crops_tta(self, crops):
        normalized_crops = []
        
        for crop in crops:
            tensor = transforms.ToTensor()(crop)
            tensor = transforms.Normalize(mean=self.mean, std=self.std)(tensor)
            normalized_crops.append(tensor)
        
        center_crop_pil = transforms.CenterCrop(self.crop_size)(crops[0])
        
        for angle in self.tta_angles:
            rotated = F.rotate(center_crop_pil, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            tensor = transforms.ToTensor()(rotated)
            tensor = transforms.Normalize(mean=self.mean, std=self.std)(tensor)
            normalized_crops.append(tensor)
        
        return torch.stack(normalized_crops, dim=0)

    def _build_transform(self, mode):
        get = self.config.get

        use_grayscale = self.dataset_name.lower() in ['fer2013', 'fer2013_new']
        
        if mode == "train":
            if use_grayscale:
                grayscale_channels = get('grayscale_output_channels', 1)
                base_transforms = [
                    transforms.Grayscale(num_output_channels=grayscale_channels),
                    transforms.Resize(self.resize_size),
                    transforms.RandomCrop(self.crop_size, padding=4),
                    transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                    transforms.RandomRotation(degrees=self.random_rotation_degrees),
                ]
            else:

                base_transforms = [
                    transforms.Resize(self.resize_size),
                    transforms.RandomCrop(self.crop_size, padding=4),
                    transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob),
                    transforms.RandomRotation(degrees=self.random_rotation_degrees),
                ]
            
            if get('use_randaugment', False):
                base_transforms.append(RandAugment(num_ops=get('randaugment_n', 2), magnitude=get('randaugment_m', 10), generator=self.generator))
            else:
                base_transforms.extend([
                    transforms.RandomApply([
                        transforms.ColorJitter(
                            brightness=self.color_jitter_brightness,
                            contrast=self.color_jitter_contrast,
                            saturation=self.color_jitter_saturation)
                    ], p=self.color_jitter_prob),
                    transforms.RandomApply([
                        transforms.RandomAffine(
                            degrees=0,
                            translate=self.random_affine_translate,
                            scale=self.random_affine_scale)
                    ], p=self.random_affine_prob),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3, sigma=self.gaussian_blur_sigma)
                    ], p=self.gaussian_blur_prob),
                ])
            base_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
                transforms.RandomErasing(p=self.random_erasing_prob, scale=(0.015, 0.045), ratio=(0.5, 2.0))
            ])
            transform = transforms.Compose(base_transforms)
        else:
            if use_grayscale:
                grayscale_channels = get('grayscale_output_channels', 1)  
                if get('use_tta', False):
                    transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=grayscale_channels),
                        transforms.Resize(self.resize_size),
                        transforms.TenCrop(self.crop_size),
                        transforms.Lambda(self._stack_crops_tta),
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Grayscale(num_output_channels=grayscale_channels),
                        transforms.Resize(self.resize_size),
                        transforms.TenCrop(self.crop_size),
                        transforms.Lambda(self._stack_crops_test),
                    ])
            else:
                if get('use_tta', False):
                    transform = transforms.Compose([
                        transforms.Resize(self.resize_size),
                        transforms.TenCrop(self.crop_size),
                        transforms.Lambda(self._stack_crops_tta),
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Resize(self.resize_size),
                        transforms.TenCrop(self.crop_size),
                        transforms.Lambda(self._stack_crops_test),
                    ])
        return transform

    def get_loader(self, path, batch_size, num_workers, mode):
        transform = self._build_transform(mode)
        shuffle = mode == "train"
        dataset = ImageFolder(path, transform=transform)
        
        loader_kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': num_workers > 0,
            'prefetch_factor': 3 if num_workers > 0 else None,
            'drop_last': mode == 'train',
            'timeout': 0 if num_workers == 0 else 30
        }
        loader_kwargs['generator'] = self.generator
        
        if num_workers > 0:
            from functools import partial
            loader_kwargs['worker_init_fn'] = partial(worker_init_fn, base_seed=self.seed)
        
        data_loader = DataLoader(dataset, **loader_kwargs)
        return data_loader

def load_data(path, batch_size, num_workers, mode, config=None):
    loader = GenericDataLoader(config=config)
    return loader.get_loader(path, batch_size, num_workers, mode)
