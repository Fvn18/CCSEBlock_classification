import yaml
import subprocess
from pathlib import Path

SEEDS = [42, 3407, 2025]
MODELS = [
    "resnet18",          
    "se_resnet18", 
    "eca_resnet18", 
    "coordatt_resnet18", 
    "simam_resnet18", 
    "ema_resnet18", 
    "ccfilm_resnet18"
]

def run_experiment(model_name, seed):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    cfg["model"]["name"] = model_name
    cfg["basic"]["seed"] = seed
    
    tmp_config = f"tmp_config_{model_name}_{seed}.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(cfg, f)
    
    print(f"\n>>> Running: {model_name} | Seed: {seed}")
    subprocess.run(["python", "train.py", "--config", tmp_config])
    
    Path(tmp_config).unlink()

if __name__ == "__main__":
    for model in MODELS:
        for seed in SEEDS:
            run_experiment(model, seed)