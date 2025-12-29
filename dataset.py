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
            base_transforms = []

            if use_grayscale:
                base_transforms.append(
                    transforms.Grayscale(num_output_channels=get('grayscale_output_channels', 1))
                )

            if get('enable_resize', True):
                base_transforms.append(transforms.Resize(self.resize_size))

            if get('enable_random_crop', True):
                base_transforms.append(
                    transforms.RandomCrop(self.crop_size, padding=get('random_crop_padding', 4))
                )

            if self.horizontal_flip_prob > 0:
                base_transforms.append(
                    transforms.RandomHorizontalFlip(p=self.horizontal_flip_prob)
                )

            if get('enable_rotation', False):
                base_transforms.append(
                    transforms.RandomRotation(self.random_rotation_degrees)
                )

            if get('use_randaugment', False):
                base_transforms.append(
                    RandAugment(
                        num_ops=get('randaugment_n', 2),
                        magnitude=get('randaugment_m', 10)
                    )
                )
            else:
                if get('enable_color_jitter', False):
                    base_transforms.append(
                        transforms.RandomApply([
                            transforms.ColorJitter(
                                self.color_jitter_brightness,
                                self.color_jitter_contrast,
                                self.color_jitter_saturation
                            )
                        ], p=self.color_jitter_prob)
                    )

                if get('enable_affine', False):
                    base_transforms.append(
                        transforms.RandomApply([
                            transforms.RandomAffine(
                                degrees=0,
                                translate=self.random_affine_translate,
                                scale=self.random_affine_scale
                            )
                        ], p=self.random_affine_prob)
                    )

                if get('enable_blur', False):
                    base_transforms.append(
                        transforms.RandomApply([
                            transforms.GaussianBlur(3, self.gaussian_blur_sigma)
                        ], p=self.gaussian_blur_prob)
                    )

            base_transforms.extend([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

            if get('enable_erasing', False):
                base_transforms.append(
                    transforms.RandomErasing(p=self.random_erasing_prob)
                )

            transform = transforms.Compose(base_transforms)
        else:
            use_tencrop = get('use_tencrop', False)
            use_tta = get('use_tta', False)

            if use_tta and not use_tencrop:
                raise ValueError(
                    "Invalid config: use_tta=True requires use_tencrop=True in test mode."
                )

            base = []

            if use_grayscale:
                base.append(
                    transforms.Grayscale(
                        num_output_channels=get('grayscale_output_channels', 1)
                    )
                )

            if get('enable_resize', True):
                base.append(transforms.Resize(self.resize_size))

            if use_tencrop:
                transform = transforms.Compose(
                    base + [
                        transforms.TenCrop(self.crop_size),
                        transforms.Lambda(
                            self._stack_crops_tta if use_tta else self._stack_crops_test
                        )
                    ]
                )
            else:
                transform = transforms.Compose(
                    base + [
                        transforms.CenterCrop(self.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(self.mean, self.std)
                    ]
                )

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
