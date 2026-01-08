# ResNet Configurations for Gradient Inversion Research

## Overview
This implementation provides multiple ResNet-based configurations to test gradient inversion attacks against different architectural choices.

## Model Variants

### 1. **ViT Baseline** - `mome_small_patch16`
- **Image Processing**: Patch Embedding (16x16 patches) → Positional Embeddings → Transformer Blocks
- **Spatial Bias**: None (patch-agnostic)
- **Positional Info**: Learned positional embeddings
- **Transformer Blocks**: Yes
- **Script**: Use existing `run-joint-coco.sh`

### 2. **Hybrid ResNet (with pos)** - `mome_resnet18_small`
- **Image Processing**: ResNet Layers → Projection → Positional Embeddings → Transformer Blocks
- **Spatial Bias**: Strong (convolutional)
- **Positional Info**: Learned positional embeddings
- **Transformer Blocks**: Yes
- **Script**: `run-joint-coco-resnet.sh` (existing)

### 3. **Hybrid ResNet (no pos)** - `mome_resnet18_small_nopos`
- **Image Processing**: ResNet Layers → Projection → Transformer Blocks (NO positional embeddings)
- **Spatial Bias**: Strong (convolutional only)
- **Positional Info**: None
- **Transformer Blocks**: Yes
- **Script**: `run-joint-coco-resnet-hybrid-nopos.sh`

### 4. **Full ResNet** - `mome_resnet18_full`
- **Image Processing**: ResNet Layers → Global Avg Pool → Projection → Output
- **Spatial Bias**: Strong (convolutional)
- **Positional Info**: None
- **Transformer Blocks**: **NO** (bypassed for images)
- **Script**: `run-joint-coco-resnet-full.sh`

## Architecture Comparison

| Model | Embedding | Transformers | Pos Embed | Image Params | Best For |
|-------|-----------|--------------|-----------|--------------|----------|
| `mome_small_patch16` | Patch Conv | ✓ | ✓ | ~5 | Baseline ViT |
| `mome_resnet18_small` | ResNet-18 | ✓ | ✓ | ~100 | Hybrid + pos info |
| `mome_resnet18_small_nopos` | ResNet-18 | ✓ | ✗ | ~100 | Hybrid, pure conv bias |
| `mome_resnet18_full` | ResNet-18 | ✗ | ✗ | ~100 | Pure CNN |

## Image Processing Flow

### ViT Mode
```
Input: (B, 3, 224, 224)
  ↓ Patch Embedding (14×14 patches)
(B, 196, 384)
  ↓ Add CLS + Pos Embeddings
(B, 197, 384)
  ↓ 12 Transformer Blocks
(B, 197, 384)
  ↓ Task Head [:, 0]
(B, num_classes)
```

### Hybrid ResNet Mode (with/without pos)
```
Input: (B, 3, 224, 224)
  ↓ ResNet conv1 → layer1-4 (overlapping receptive fields)
(B, 512, 7, 7)
  ↓ Reshape + Project
(B, 49, 384)
  ↓ Add CLS [+ Pos Embeddings (optional)]
(B, 50, 384)
  ↓ 12 Transformer Blocks
(B, 50, 384)
  ↓ Task Head [:, 0]
(B, num_classes)
```

### Full ResNet Mode
```
Input: (B, 3, 224, 224)
  ↓ ResNet conv1 → layer1-4
(B, 512, 7, 7)
  ↓ Global Average Pool
(B, 512)
  ↓ Project to embed_dim
(B, 384)
  ↓ Reshape for compatibility
(B, 1, 384)
  ↓ Task Head [:, 0]
(B, num_classes)
```

## Available Models

### Hybrid ResNet (ResNet + Transformers)
- `mome_resnet18_small` - ResNet-18 + transformers + pos embeddings
- `mome_resnet18_small_nopos` - ResNet-18 + transformers (NO pos embeddings)
- `mome_resnet34_small` - ResNet-34 + transformers + pos embeddings
- `mome_resnet50_small` - ResNet-50 + transformers + pos embeddings

### Full ResNet (ResNet Only, NO Transformers for Images)
- `mome_resnet18_full` - ResNet-18 end-to-end
- `mome_resnet34_full` - ResNet-34 end-to-end
- `mome_resnet50_full` - ResNet-50 end-to-end

## Usage Examples

```bash
# Baseline ViT
bash run-joint-coco.sh

# Hybrid ResNet-18 with positional embeddings
bash run-joint-coco-resnet.sh

# Hybrid ResNet-18 WITHOUT positional embeddings
bash run-joint-coco-resnet-hybrid-nopos.sh

# Full ResNet-18 (no transformers for images)
bash run-joint-coco-resnet-full.sh
```

## Gradient Inversion Implications

### Parameter Counts

| Component | ViT | Hybrid ResNet | Full ResNet |
|-----------|-----|---------------|-------------|
| Embedding | ~5 params | ~100 params | ~100 params |
| Blocks | 12 transformers | 12 transformers | 0 (images) |
| Gradient Structure | Simple patch conv | Multi-layer conv+bn | Multi-layer conv+bn |

### Expected Research Insights

1. **ViT Baseline**
   - Patch-agnostic processing
   - Minimal spatial inductive bias
   - How do patch embeddings affect reconstruction?

2. **Hybrid + Pos Embeddings**
   - Strong spatial bias from ResNet
   - Additional positional information
   - Does double spatial info help/hurt attacks?

3. **Hybrid NO Pos Embeddings**
   - Only convolutional spatial bias
   - No explicit position encoding
   - Effect of pure CNN bias without positional signals

4. **Full ResNet**
   - Complete CNN architecture
   - No attention mechanisms for images
   - Most different from ViT
   - Standard CNN gradient structure
   - How does removing transformers affect reconstruction?

## Key Features

### Compatibility
- ✅ Works with existing `get_param_indices_fedcola`
- ✅ Same task head interface
- ✅ 224×224 image support
- ✅ No changes to data loading
- ✅ Text modality unchanged (always uses BERT + transformers)

### Parameter Selection
The `get_param_indices_fedcola` function correctly identifies:
- `img_embedding`: All ResNet parameters (conv, bn, projection) for hybrid/full modes
- `img_blocks`: Transformer blocks (empty for full ResNet mode)
- Text processing remains unchanged

### Pretrained Weights
- Hybrid and Full ResNet modes support `--pretrained` flag
- Loads ImageNet pretrained weights for ResNet backbone
- Text modality uses pretrained BERT embeddings

## Implementation Details

### Code Structure

**New Classes:**
1. `FullResNetBackbone` - Complete ResNet for end-to-end processing
2. `ResNetEmbedding` (updated) - Hybrid mode with optional pos embeddings

**Modified Methods:**
1. `ModalityAgnosticTransformer.__init__` - Supports `full_resnet` mode
2. `ModalityAgnosticTransformer.forward` - Skips transformers for full ResNet images

### Configuration Parameters

```python
kwargs = {
    'use_resnet': True/False,      # Enable ResNet embedding
    'full_resnet': True/False,     # Enable full ResNet (bypass transformers)
    'resnet_type': 'resnet18',     # ResNet depth: 18/34/50/101/152
    'pretrained': True/False,      # Load ImageNet pretrained weights
    'use_pos_embed': True/False,   # Add positional embeddings (hybrid only)
}
```

## Testing Strategy

To comprehensively test gradient inversion:

1. **Test all 4 configurations** to understand architectural effects
2. **Compare reconstruction quality** across different spatial biases
3. **Analyze gradient structures** - more/fewer parameters
4. **Evaluate with/without positional information**

## Notes

- All configurations maintain 224×224 input size
- Text modality always uses BERT embeddings + transformers
- Full ResNet mode bypasses transformers ONLY for image modality
- Multimodal clients process text through transformers normally
- Compatible with federated learning setup
