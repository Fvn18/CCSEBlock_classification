import torch
import torch.nn as nn
import time
import psutil
import os
import importlib.util
import numpy as np
from tabulate import tabulate

def load_model(model_name, num_classes=7, **kwargs):
    model_mapping = {
        'ExtraNet': {
            'module_path': 'ExtraNet.py',
            'class_name': 'CNNmodel',
            'default_params': {
                'use_spatial_attention': False,
                'use_simple_fusion': True
            }
        },
        'ExtraNet_CCSE': {
            'module_path': 'ExtraNet_CCSE.py',
            'class_name': 'ExtraNet_CCSE',
            'default_params': {
                'use_spatial_attention': False,
                'use_simple_fusion': True
            }
        },
        'ExtraNet_CCSE_Lite': {
            'module_path': 'ExtraNet_CCSE_Lite.py',
            'class_name': 'ExtraNet_Lite_CCSE',
            'default_params': {
                'use_simple_fusion': True
            }
        }
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unsupported model name: {model_name}. Supported models: {list(model_mapping.keys())}")
    
    model_info = model_mapping[model_name]
    
    module_spec = importlib.util.spec_from_file_location(model_name, model_info['module_path'])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    
    model_class = getattr(module, model_info['class_name'])
    
    params = model_info['default_params'].copy()
    params.update(kwargs)
    
    model = model_class(num_classes=num_classes, **params)
    
    model.eval()
    
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model, verbose=True):
    param_count = count_parameters(model)
    
    model_size_mb = param_count * 4 / (1024 * 1024)
    
    if verbose:
        print(f"Model parameters: {param_count:,}")
        print(f"Model size: {model_size_mb:.4f} MB")
    
    return param_count, model_size_mb

def model_summary(model, input_size=(1, 64, 64)):
    print(f"\n{'='*60}")
    print(f"Model structure summary")
    print(f"{'='*60}")
    
    print(f"Model name: {model.__class__.__name__}")
    print(model)
    
    param_count, model_size = get_model_size(model, verbose=False)
    
    device = next(model.parameters()).device
    input_tensor = torch.randn(1, *input_size, device=device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {param_count:,}")
    print(f"Model size: {model_size:.4f} MB")
    print(f"{'='*60}\n")
    
    return {
        'param_count': param_count,
        'model_size_mb': model_size,
        'input_shape': input_tensor.shape,
        'output_shape': output.shape
    }

def test_inference_speed(model, input_size=(1, 64, 64), batch_sizes=[1, 8, 16, 32], iterations=100, warmup=10):
    device = next(model.parameters()).device
    print(f"\n{'='*60}")
    print(f"Inference speed test (device: {device})")
    print(f"{'='*60}")
    
    results = {}
    
    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, *input_size, device=device)
        
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = (total_time / iterations) * 1000  # Convert to milliseconds
        fps = iterations * batch_size / total_time
        
        print(f"Batch size: {batch_size}")
        print(f"  Average inference time: {avg_time:.4f} ms")
        print(f"  FPS: {fps:.2f}")
        
        results[batch_size] = {
            'avg_time_ms': avg_time,
            'fps': fps
        }
    
    print(f"{'='*60}\n")
    return results

def test_memory_usage(model, input_size=(1, 64, 64), batch_size=32):
    device = next(model.parameters()).device
    print(f"\n{'='*60}")
    print(f"Memory usage test (device: {device})")
    print(f"{'='*60}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model_memory = torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
    
    input_tensor = torch.randn(batch_size, *input_size, device=device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory / (1024 * 1024)
        print(f"CUDA peak memory: {peak_memory_mb:.4f} MB")
    else:
        process = psutil.Process(os.getpid())
        cpu_memory_mb = process.memory_info().rss / (1024 * 1024)
        print(f"CPU memory usage: {cpu_memory_mb:.4f} MB")
        peak_memory_mb = cpu_memory_mb
    
    print(f"{'='*60}\n")
    
    return {
        'peak_memory_mb': peak_memory_mb,
        'device': device.type
    }

def test_5crop_compatibility(model, input_size=(1, 64, 64), batch_size=4):
    device = next(model.parameters()).device
    print(f"\n{'='*60}")
    print(f"5-crop compatibility test")
    print(f"{'='*60}")
    
    try:
        input_5crop = torch.randn(batch_size, 5, *input_size, device=device)
        
        with torch.no_grad():
            output = model(input_5crop)
        
        print(f"5-crop input shape: {input_5crop.shape}")
        print(f"5-crop output shape: {output.shape}")
        print(f"✓ 5-crop compatibility test passed")
        print(f"{'='*60}\n")
        return True
    except Exception as e:
        print(f"✗ 5-crop compatibility test failed: {str(e)}")
        print(f"{'='*60}\n")
        return False

def benchmark_all_models(model_names=None, input_size=(1, 64, 64), batch_sizes=[1, 32], iterations=100):
    if model_names is None:
        model_names = ['ExtraNet', 'ExtraNet_CCSE', 'ExtraNet_CCSE_Lite']
    
    all_results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print(f"{'='*80}")
        
        try:
            model = load_model(model_name)
            model = model.to(device)
            
            summary = model_summary(model, input_size)
            inference_results = test_inference_speed(model, input_size, batch_sizes, iterations)
            memory_results = test_memory_usage(model, input_size)
            five_crop_passed = test_5crop_compatibility(model, input_size)
            
            all_results[model_name] = {
                'summary': summary,
                'inference': inference_results,
                'memory': memory_results,
                'five_crop_compatible': five_crop_passed
            }
            
        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")
            all_results[model_name] = {'error': str(e)}
    
    generate_comparison_report(all_results, batch_sizes)
    
    return all_results

def generate_comparison_report(results, batch_sizes):
    print(f"\n{'='*80}")
    print(f"Model performance comparison report")
    print(f"{'='*80}")
    
    table_data = []
    headers = ['Model Name', 'Parameters', 'Model Size(MB)']
    
    for bs in batch_sizes:
        headers.extend([f'时间@{bs}(ms)', f'FPS@{bs}'])
    
    for model_name, model_results in results.items():
        if 'error' in model_results:
            row = [model_name, 'N/A', 'N/A'] + ['Error'] * (2 * len(batch_sizes))
        else:
            row = [
                model_name,
                f"{model_results['summary']['param_count']:,}",
                f"{model_results['summary']['model_size_mb']:.2f}"
            ]
            
            for bs in batch_sizes:
                if bs in model_results['inference']:
                    row.extend([
                        f"{model_results['inference'][bs]['avg_time_ms']:.2f}",
                        f"{model_results['inference'][bs]['fps']:.0f}"
                    ])
                else:
                    row.extend(['N/A', 'N/A'])
        
        table_data.append(row)
    
    try:
        table_data.sort(key=lambda x: int(x[1].replace(',', '')) if x[1] != 'N/A' else float('inf'))
    except:
        pass
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print(f"\n{'='*80}")
    print(f"Memory usage comparison")
    print(f"{'='*80}")
    
    memory_data = []
    for model_name, model_results in results.items():
        if 'error' not in model_results:
            memory_data.append([
                model_name,
                f"{model_results['memory']['peak_memory_mb']:.2f} MB",
                model_results['memory']['device']
            ])
    
    print(tabulate(memory_data, headers=['Model Name', 'Peak Memory', 'Device'], tablefmt='grid'))
    
    print(f"\n{'='*80}")
    print(f"5-crop compatibility")
    print(f"{'='*80}")
    
    for model_name, model_results in results.items():
        if 'error' not in model_results:
            status = '✓ Passed' if model_results['five_crop_compatible'] else '✗ Failed'
            print(f"{model_name}: {status}")

if __name__ == "__main__":
    print("FER2013 Model Performance Benchmark Tool")
    print("===========================================")
    
    test_config = {
        'input_size': (1, 64, 64),  # Match FER2013 dataset
        'batch_sizes': [1, 4, 16, 32],  # Test different batch sizes
        'iterations': 100  # Number of test iterations per batch size
    }
    
    print(f"Test configuration:")
    print(f"  Input size: {test_config['input_size']}")
    print(f"  Batch sizes: {test_config['batch_sizes']}")
    print(f"  Iterations: {test_config['iterations']}")
    print()
    
    all_results = benchmark_all_models(
        input_size=test_config['input_size'],
        batch_sizes=test_config['batch_sizes'],
        iterations=test_config['iterations']
    )
    
    print(f"\n{'='*80}")
    print(f"Testing completed!")
    print(f"{'='*80}")
