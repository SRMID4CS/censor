# Quick Reference: ResNet Model Variants

## Model Selection Guide

| Use Case | Model Name | Command |
|----------|-----------|---------|
| **Baseline ViT** | `mome_small_patch16` | See existing scripts |
| **Hybrid + Pos Embed** | `mome_resnet18_small` | `bash run-joint-coco-resnet.sh` |
| **Hybrid NO Pos** | `mome_resnet18_small_nopos` | `bash run-joint-coco-resnet-hybrid-nopos.sh` |
| **Full ResNet** | `mome_resnet18_full` | `bash run-joint-coco-resnet-full.sh` |

## Quick Comparison

```
┌─────────────────────┬──────────────┬──────────────┬───────────┬──────────────┐
│ Model               │ Conv Layers  │ Transformers │ Pos Embed │ Use Case     │
├─────────────────────┼──────────────┼──────────────┼───────────┼──────────────┤
│ mome_small_patch16  │ Patch Conv   │      ✓       │     ✓     │ Baseline     │
│ mome_resnet18_small │ ResNet-18    │      ✓       │     ✓     │ Hybrid+Pos   │
│ _small_nopos        │ ResNet-18    │      ✓       │     ✗     │ Hybrid Only  │
│ _resnet18_full      │ ResNet-18    │      ✗       │     ✗     │ Pure CNN     │
└─────────────────────┴──────────────┴──────────────┴───────────┴──────────────┘
```

## Available Variants

### ResNet-18
- `mome_resnet18_small` (hybrid + pos)
- `mome_resnet18_small_nopos` (hybrid, no pos)
- `mome_resnet18_full` (full, no transformers)

### ResNet-34
- `mome_resnet34_small` (hybrid + pos)
- `mome_resnet34_full` (full, no transformers)

### ResNet-50
- `mome_resnet50_small` (hybrid + pos)
- `mome_resnet50_full` (full, no transformers)

## Example Commands

```bash
# Test hybrid ResNet-18 with positional embeddings
python run_joint_mm_rec.py --model_name mome_resnet18_small --pretrained ...

# Test hybrid ResNet-18 without positional embeddings
python run_joint_mm_rec.py --model_name mome_resnet18_small_nopos --pretrained ...

# Test full ResNet-18 (no transformers for images)
python run_joint_mm_rec.py --model_name mome_resnet18_full --pretrained ...

# Test with ResNet-50 backbone
python run_joint_mm_rec.py --model_name mome_resnet50_full --pretrained ...
```

## Key Differences

### Spatial Inductive Bias
- **ViT**: ❌ None (patch-based)
- **Hybrid ResNet**: ✅ Strong (convolutional)
- **Full ResNet**: ✅ Strong (convolutional)

### Positional Information
- **ViT**: ✅ Learned embeddings
- **Hybrid + Pos**: ✅ Learned embeddings
- **Hybrid No Pos**: ❌ None
- **Full ResNet**: ❌ None (implicit in conv)

### Attention Mechanism
- **ViT**: ✅ 12 transformer blocks
- **Hybrid**: ✅ 12 transformer blocks
- **Full ResNet**: ❌ None for images

## Files Modified/Created

### Modified
- `FedCola/src/models/mome.py` - Added FullResNetBackbone, updated forward logic

### Created
- `run-joint-coco-resnet-full.sh` - Full ResNet mode script
- `run-joint-coco-resnet-hybrid-nopos.sh` - Hybrid no pos script
- `RESNET_IMPLEMENTATION.md` - Complete documentation

### Existing (Reuse)
- `run-joint-coco-resnet.sh` - Already configured for hybrid + pos
