import os 
import torch 
import random
import numpy as np
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader 

STATS = { 
    'cifar10':  {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)}, 
    'cifar100': {'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)}, 
    'imagenet': {'mean': (0.485, 0.456, 0.406),  'std': (0.229, 0.224, 0.225)}, 
    'cub200':   {'mean': (0.485, 0.456, 0.406),  'std': (0.229, 0.224, 0.225)}, 
    'default':  {'mean': (0.5, 0.5, 0.5),        'std': (0.5, 0.5, 0.5)} 
} 

def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    worker_seed = base_seed + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    os.environ['PYTHONHASHSEED'] = str(worker_seed)

_global_seed = 42

def set_global_seed(seed: int):
    global _global_seed
    _global_seed = seed

def universal_worker_init_fn(worker_id: int) -> None:
    worker_init_fn(worker_id, _global_seed)

class UniversalLoader: 
    def __init__(self, config): 
        self.cfg = config 
        self.dataset_name = config['basic']['dataset'].lower() 
        self.data_root = config['data']['root'] 
        self.img_size = config['data']['img_size'] 
        self.batch_size = config['training']['batch_size'] 
        self.num_workers = config['basic']['num_workers'] 
        self.seed = config['basic'].get('seed', 42)

        set_global_seed(self.seed)
        
        stats = STATS.get(self.dataset_name, STATS['imagenet']) 
        self.mean, self.std = stats['mean'], stats['std'] 

    def _get_transforms(self, mode='train'): 
        is_cifar = 'cifar' in self.dataset_name 
        
        if mode == 'train': 
            if is_cifar: 
                return transforms.Compose([ 
                    transforms.RandomCrop(32, padding=4), 
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize(self.mean, self.std) 
                ]) 
            else: 
                return transforms.Compose([ 
                    transforms.RandomResizedCrop(self.img_size), 
                    transforms.RandomHorizontalFlip(), 
                    transforms.ToTensor(), 
                    transforms.Normalize(self.mean, self.std) 
                ]) 
        else: 
            if is_cifar: 
                return transforms.Compose([ 
                    transforms.ToTensor(), 
                    transforms.Normalize(self.mean, self.std) 
                ]) 
            else: 
                crop_padding = int(self.img_size / 224 * 32) 
                return transforms.Compose([ 
                    transforms.Resize(self.img_size + 32), 
                    transforms.CenterCrop(self.img_size), 
                    transforms.ToTensor(), 
                    transforms.Normalize(self.mean, self.std) 
                ]) 

    def get_loader(self, mode='train'): 
        transform = self._get_transforms(mode) 
        shuffle = (mode == 'train') 
        drop_last = (mode == 'train') 
        
        if self.dataset_name in ['cifar10', 'cifar100']: 
            if mode == 'train': 
                path = self.cfg['data']['train_path'] 
            else: 
                path = self.cfg['data']['val_path'] 
            
            if not os.path.exists(path): 
                raise FileNotFoundError(f"Dataset path not found: {path}") 
                
            dataset = datasets.ImageFolder(root=path, transform=transform)
        else: 
            if mode == 'train': 
                path = self.cfg['data']['train_path'] 
            else: 
                path = self.cfg['data']['val_path'] 
            
            if not os.path.exists(path): 
                raise FileNotFoundError(f"Dataset path not found: {path}") 
                
            dataset = datasets.ImageFolder(root=path, transform=transform) 

        loader = DataLoader( 
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            drop_last=drop_last, 
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=universal_worker_init_fn if self.num_workers > 0 else None
        ) 
        
        return loader 

def load_data(path, batch_size, num_workers, mode, config): 
    loader_factory = UniversalLoader(config) 
    return loader_factory.get_loader(mode)