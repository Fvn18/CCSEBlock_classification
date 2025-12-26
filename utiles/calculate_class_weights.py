
import os
import numpy as np
from collections import defaultdict

def count_class_samples(train_dir):
    class_counts = defaultdict(int)
    
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        class_counts[class_name] = len(files)
        
        print(f"Class '{class_name}': {len(files)} images")
    
    return class_counts

def calculate_class_weights(class_counts, method='inverse_frequency'):
    class_weights = {}
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Number of classes: {num_classes}")
    
    if method == 'inverse_frequency':
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (num_classes * count)
            
    elif method == 'inverse_frequency_sqrt':
        for class_name, count in class_counts.items():
            class_weights[class_name] = np.sqrt(total_samples / (num_classes * count))
            
    elif method == 'balanced':
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (num_classes * count)
            
    else:
        raise ValueError(f"Unknown weight calculation method: {method}")
    
    min_weight = min(class_weights.values())
    for class_name in class_weights:
        class_weights[class_name] = class_weights[class_name] / min_weight
    
    return class_weights

def print_class_weights(class_weights):
    print("\nClass weights (normalized, minimum weight = 1):")
    print("-" * 40)
    
    sorted_weights = sorted(class_weights.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, weight in sorted_weights:
        print(f"{class_name.ljust(10)}: {weight:.4f}")
    
    print("-" * 40)


import os

def save_class_weights(class_weights, output_file='utiles/class_weights.json'):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    import json

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(class_weights, f, ensure_ascii=False, indent=4)

        print(f"\nClass weights saved to: {output_file}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, '../fer2013', 'train')
    
    print("FER2013 Dataset Training Set Class Weights Calculation")
    print("=" * 50)
    print(f"Training set directory: {train_dir}")
    
    if not os.path.exists(train_dir):
        print(f"Error: Training set directory does not exist: {train_dir}")
        return
    
    print("\nCounting samples per class:")
    class_counts = count_class_samples(train_dir)
    
    if not class_counts:
        print("Error: No class folders or images found")
        return
    
    print("\nCalculating class weights (using inverse frequency method):")
    class_weights = calculate_class_weights(class_counts, method='inverse_frequency_sqrt')
    
    print_class_weights(class_weights)
    
    save_class_weights(class_weights)
    
    print("\nPyTorch-compatible weight dictionary (class index -> weight):")
    sorted_classes = sorted(class_counts.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted_classes)}
    
    print("Class index mapping:")
    for cls, idx in class_to_idx.items():
        print(f"  '{cls}': {idx}")
    
    print("\nWeight dictionary:")
    weights_dict = {class_to_idx[cls]: class_weights[cls] for cls in sorted_classes}
    print("class_weights = {")
    for idx, weight in sorted(weights_dict.items()):
        print(f"    {idx}: {weight:.6f},")
    print("}")

if __name__ == "__main__":
    main()