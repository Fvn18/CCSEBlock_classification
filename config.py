import yaml 
import os 

def get_defaults(): 
    return { 
        "basic": { 
            "dataset": "cifar10", 
            "device": "auto", 
            "num_workers": 4, 
            "seed": 42, 
            "use_amp": True 
        }, 
        "model": { 
            "name": "resnet18", 
            "scalable_model_scale": "tiny" 
        }, 
        "data": { 
            "root": "./data", 
            "img_size": 32, 
            "grayscale_output_channels": 3 
        }, 
        "training": { 
            "batch_size": 128, 
            "epochs": 100, 
            "loss_type": "ce", 
            "label_smoothing": 0.0, 
            "mixup_alpha": 0.0, 
            "cutmix_alpha": 0.0 
        }, 
        "optimizer": { 
            "optimizer": "sgd", 
            "lr": 0.1, 
            "momentum": 0.9, 
            "weight_decay": 5e-4 
        }, 
        "scheduler": { 
            "scheduler": "cosine", 
            "warmup_epochs": 5 
        } 
    } 

def flatten_config(config): 
    flat_config = {} 
    for section, values in config.items(): 
        if isinstance(values, dict): 
            for k, v in values.items(): 
                flat_config[k] = v 
        else: 
            flat_config[section] = values 
    return flat_config 

def load_config(config_path): 
    if not os.path.exists(config_path): 
        print(f"Config file {config_path} not found. Using defaults.") 
        return get_defaults() 
    
    with open(config_path, 'r', encoding='utf-8') as f: 
        config = yaml.safe_load(f) 
        
    defaults = get_defaults() 
    for section, values in defaults.items(): 
        if section not in config: 
            config[section] = values 
        else: 
            for k, v in values.items(): 
                if k not in config[section]: 
                    config[section][k] = v 
                    
    return config