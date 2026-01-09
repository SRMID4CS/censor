import torch

def get_param_indices_fedcola(model, verbose=False):
    """
    Returns a dictionary with parameter indices for:
    1. img embedding layer
    2. txt embedding layer
    3. img transformer blocks
    4. txt transformer blocks
    5. task heads
    """
    indices = {
        "img_embedding": [],
        "txt_embedding": [],
        "img_blocks": [],
        "txt_blocks": [],
        "heads": [],
    }
    idx = 0
    for name, param in model.named_parameters():
        if "embeddings.0" in name:
            indices["img_embedding"].append(idx)
        elif "embeddings.1" in name:
            indices["txt_embedding"].append(idx)
        elif "blockses.0" in name:
            indices["img_blocks"].append(idx)
        elif "blockses.1" in name:
            indices["txt_blocks"].append(idx)
        elif "head" in name:
            indices["heads"].append(idx)
        idx += 1
    if verbose:
        for k, v in indices.items():
            print(f"{k}: {v}")
    return indices

def select_indices_fedcola(model, select=("img_embedding", "txt_embedding", "img_blocks", "txt_blocks", "heads")):
    """
    Returns a sorted list of indices for the selected components.
    select: tuple/list of keys from get_param_indices
    
    Automatically excludes img_blocks in full_resnet mode since transformers are bypassed.
    """
    indices_dict = get_param_indices_fedcola(model)
    
    # Check if model is in full_resnet mode
    is_full_resnet = getattr(model, 'full_resnet', False)
    
    selected = []
    for key in select:
        # Skip img_blocks in full_resnet mode - transformers are not used for images
        if is_full_resnet and key == "img_blocks":
            # print(f"[FedCola Indices] Skipping img_blocks in full_resnet mode (transformers bypassed)")
            continue
        selected.extend(indices_dict.get(key, []))
    return sorted(selected)

def add_indices(indices_set, new_indices):
    """
    Adds new indices to an existing set (list) of indices.
    """
    indices_set = set(indices_set)
    indices_set.update(new_indices)
    return sorted(indices_set)

def print_indices_summary(model, img_indices='def', txt_indices='def'):
    """
    Prints a summary of which gradient indices will be used for reconstruction.
    Useful for debugging and understanding the attack surface.
    """
    is_full_resnet = getattr(model, 'full_resnet', False)
    use_resnet = getattr(model, 'use_resnet', False)
    
    print("\n" + "="*60)
    print("GRADIENT INDICES CONFIGURATION")
    print("="*60)
    print(f"Model Mode: {'Full ResNet' if is_full_resnet else ('Hybrid ResNet' if use_resnet else 'ViT')}")
    print(f"Image Indices: {img_indices}")
    print(f"Text Indices: {txt_indices}")
    
    if img_indices.startswith('fedcola'):
        to_select = []
        if 'img_emb' in img_indices:
            to_select.append("img_embedding")
        if 'img_block' in img_indices:
            if is_full_resnet:
                print(f"  ⚠️  img_blocks requested but will be SKIPPED (full_resnet mode)")
            else:
                to_select.append("img_blocks")
        print(f"Image components: {to_select}")
        
    if txt_indices.startswith('fedcola'):
        to_select = []
        if 'txt_emb' in txt_indices:
            to_select.append("txt_embedding")
        if 'txt_block' in txt_indices:
            to_select.append("txt_blocks")
        print(f"Text components: {to_select}")
    
    print("="*60 + "\n")