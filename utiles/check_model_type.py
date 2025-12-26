
import torch
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_model_state_dict(weights_path):
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict'], "Using 'state_dict' key"
            elif 'model_state_dict' in checkpoint:
                return checkpoint['model_state_dict'], "Using 'model_state_dict' key"
            elif any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                return checkpoint, "Directly use dictionary as state_dict"
            else:
                for key in checkpoint.keys():
                    if isinstance(checkpoint[key], dict) and any(k.endswith('.weight') for k in checkpoint[key].keys()):
                        return checkpoint[key], f"Using state_dict from '{key}' key"
        
        raise ValueError("Unrecognized model file format")
    except Exception as e:
        return None, str(e)

def analyze_model_keys(state_dict):
    keys = list(state_dict.keys())
    
    ccse_features = [
        'ccse',  # CCSEBlock
        'ccse_block',
        'cascaded_attention',
        'channel_attention',
        'spatial_attention'
    ]
    
    extranet_features = [
        'eca',  # ECAAttention
        'eca_attention',
        'spatial',  # SpatialAttention
        'spatial_attention'
    ]
    
    ccse_score = 0
    extranet_score = 0
    
    for key in keys:
        key_lower = key.lower()
        
        for feature in ccse_features:
            if feature in key_lower:
                ccse_score += 1
                break
        
        for feature in extranet_features:
            if feature in key_lower:
                extranet_score += 1
                break
    
    conv_pattern = re.compile(r'conv(\d+)\.(\d+)\.weight')
    conv_layers = {}
    
    for key in keys:
        match = conv_pattern.search(key)
        if match:
            block_idx = int(match.group(1))
            if block_idx not in conv_layers:
                conv_layers[block_idx] = 0
            conv_layers[block_idx] += 1
    
    return {
        'ccse_score': ccse_score,
        'extranet_score': extranet_score,
        'total_keys': len(keys),
        'conv_layer_distribution': conv_layers
    }

def compare_with_model_architecture(state_dict, model_class, num_classes=7):
    try:
        model = model_class(num_classes=num_classes)
        model_state_dict = model.state_dict()
        
        state_keys = set(state_dict.keys())
        model_keys = set(model_state_dict.keys())
        
        common_keys = state_keys.intersection(model_keys)
        shape_match_count = 0
        shape_mismatch_count = 0
        
        for key in common_keys:
            if state_dict[key].shape == model_state_dict[key].shape:
                shape_match_count += 1
            else:
                shape_mismatch_count += 1
        
        if len(model_keys) > 0:
            key_match_rate = len(common_keys) / len(model_keys) * 100
            shape_match_rate = shape_match_count / len(common_keys) * 100 if common_keys else 0
        else:
            key_match_rate = 0
            shape_match_rate = 0
        
        return {
            'compatible': key_match_rate > 80 and shape_match_rate > 95,
            'key_match_rate': key_match_rate,
            'shape_match_rate': shape_match_rate,
            'common_keys': len(common_keys),
            'model_keys': len(model_keys),
            'state_keys': len(state_keys)
        }
    except Exception as e:
        return {
            'compatible': False,
            'error': str(e)
        }

def main():
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} file does not exist")
        return
    
    print(f"Checking model: {model_path}")
    print("="*60)
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"Model file size: {file_size:.2f} MB")
    
    state_dict, message = get_model_state_dict(model_path)
    if state_dict is None:
        print(f"Unable to parse model file: {message}")
        return
    
    print(f"Model parsing: {message}")
    print(f"Weights contain {len(state_dict)} parameters")
    
    print("\nWeight key examples (first 5):")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {i+1}. {key} - Shape: {state_dict[key].shape}")
    
    print("\n" + "="*60)
    print("Method 1: Key name feature analysis")
    key_analysis = analyze_model_keys(state_dict)
    
    print(f"CCSE feature matches: {key_analysis['ccse_score']}")
    print(f"ExtraNet feature matches: {key_analysis['extranet_score']}")
    print(f"Convolutional layer distribution: {key_analysis['conv_layer_distribution']}")
    
    try:
        import model.ExtraNet as ExtraNet
        import model.ExtraNet_CCSE as ExtraNet_CCSE
        
        print("\n" + "="*60)
        print("Method 2: Model architecture compatibility analysis")
        
        extranet_compat = compare_with_model_architecture(state_dict, ExtraNet)
        print("\nExtraNet compatibility analysis:")
        if 'error' in extranet_compat:
            print(f"  Error: {extranet_compat['error']}")
        else:
            print(f"  Key match rate: {extranet_compat['key_match_rate']:.2f}%")
            print(f"  Shape match rate: {extranet_compat['shape_match_rate']:.2f}%")
            print(f"  Common keys: {extranet_compat['common_keys']}/{extranet_compat['model_keys']}")
            print(f"  Compatibility: {'High' if extranet_compat['compatible'] else 'Low'}")
        
        extranet_ccse_compat = compare_with_model_architecture(state_dict, ExtraNet_CCSE)
        print("\nExtraNet_CCSE compatibility analysis:")
        if 'error' in extranet_ccse_compat:
            print(f"  Error: {extranet_ccse_compat['error']}")
        else:
            print(f"  Key match rate: {extranet_ccse_compat['key_match_rate']:.2f}%")
            print(f"  Shape match rate: {extranet_ccse_compat['shape_match_rate']:.2f}%")
            print(f"  Common keys: {extranet_ccse_compat['common_keys']}/{extranet_ccse_compat['model_keys']}")
            print(f"  Compatibility: {'High' if extranet_ccse_compat['compatible'] else 'Low'}")
            
        print("\n" + "="*60)
        print("Method 3: Direct loading test")
        
        try:
            extranet = ExtraNet(num_classes=7)
            extranet.load_state_dict(state_dict, strict=True)
            extranet_load_success = True
        except Exception as e:
            extranet_load_success = False
            extranet_error = str(e)
        
        try:
            extranet_ccse = ExtraNet_CCSE(num_classes=7)
            extranet_ccse.load_state_dict(state_dict, strict=True)
            extranet_ccse_load_success = True
        except Exception as e:
            extranet_ccse_load_success = False
            extranet_ccse_error = str(e)
            
    except ImportError as e:
        print(f"Failed to import model class: {e}")
        extranet_load_success = False
        extranet_ccse_load_success = False
    
    print("\n" + "="*60)
    print("Comprehensive judgment result:")
    
    evidence = []
    
    if key_analysis['ccse_score'] > key_analysis['extranet_score'] * 1.5:
        evidence.append("Key name features strongly support CCSE model")
    elif key_analysis['extranet_score'] > key_analysis['ccse_score'] * 1.5:
        evidence.append("Key name features strongly support ExtraNet model")
    
    if 'extranet_compat' in locals() and 'extranet_ccse_compat' in locals():
        if extranet_compat.get('compatible', False) and not extranet_ccse_compat.get('compatible', False):
            evidence.append("Architecture compatibility supports ExtraNet model")
        elif extranet_ccse_compat.get('compatible', False) and not extranet_compat.get('compatible', False):
            evidence.append("Architecture compatibility supports ExtraNet_CCSE model")
        elif extranet_compat.get('key_match_rate', 0) > extranet_ccse_compat.get('key_match_rate', 0) + 20:
            evidence.append("Architecture compatibility slightly supports ExtraNet model")
        elif extranet_ccse_compat.get('key_match_rate', 0) > extranet_compat.get('key_match_rate', 0) + 20:
            evidence.append("Architecture compatibility slightly supports ExtraNet_CCSE model")
    
    if extranet_load_success and not extranet_ccse_load_success:
        evidence.append("Direct loading test confirms ExtraNet model")
    elif extranet_ccse_load_success and not extranet_load_success:
        evidence.append("Direct loading test confirms ExtraNet_CCSE model")
    elif extranet_load_success and extranet_ccse_load_success:
        evidence.append("Both models can be loaded successfully, structures are very similar")
    
    if evidence:
        print("\nJudgment basis:")
        for i, item in enumerate(evidence, 1):
            print(f"  {i}. {item}")
    
    print("\nFinal conclusion:")
    if extranet_load_success and not extranet_ccse_load_success:
        print("best_model.pth is ExtraNet model")
    elif extranet_ccse_load_success and not extranet_load_success:
        print("best_model.pth is ExtraNet_CCSE model")
    elif extranet_load_success and extranet_ccse_load_success:
        print("Cannot fully determine, both models can be loaded")
        print("   Suggest checking training logs in the code repository for more information")
    else:
        if 'extranet_compat' in locals() and 'extranet_ccse_compat' in locals():
            if extranet_compat.get('key_match_rate', 0) > extranet_ccse_compat.get('key_match_rate', 0):
                print(f"Based on compatibility analysis, most likely ExtraNet model (key match rate: {extranet_compat['key_match_rate']:.1f}%)")
            else:
                print(f"Based on compatibility analysis, most likely ExtraNet_CCSE model (key match rate: {extranet_ccse_compat['key_match_rate']:.1f}%)")
        else:
            print("Cannot determine model type")
            print("   Please check model file format or provide more information")

if __name__ == "__main__":
    main()
