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
    """
    indices_dict = get_param_indices_fedcola(model)
    selected = []
    for key in select:
        selected.extend(indices_dict.get(key, []))
    return sorted(selected)

def add_indices(indices_set, new_indices):
    """
    Adds new indices to an existing set (list) of indices.
    """
    indices_set = set(indices_set)
    indices_set.update(new_indices)
    return sorted(indices_set)