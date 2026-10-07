---
license: mit
datasets:
- ChilleD/SVAMP
- tatsu-lab/alpaca
- databricks/databricks-dolly-15k
language:
- en
metrics:
- perplexity
base_model:
- openai-community/gpt2
pipeline_tag: text-generation
library_name: transformers
---
# GPT-2 Instruct Model Fine-tuning

This project fine-tunes the GPT-2 instruct model on the Databricks Dolly-15k dataset for instruction-following tasks.

## Model & Dataset

- **Base Model**: [FurkanNar/gpt-2_instruct](https://huggingface.co/FurkanNar/gpt-2_instruct)
- **Dataset**: [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- **Train Samples**: 13,500 (90% of dataset)
- **Validation Samples**: 1,500 (10% of dataset)

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Max Sequence Length | 128 tokens |
| Epochs | 4 |
| Batch Size | 4 |
| Learning Rate | 2e-5 |
| Max Gradient Norm | 1.0 |
| Mixed Precision | FP16 (enabled) |
| GPU Memory Fraction | 50% |

## Training Results

### Epoch Summaries

| Epoch | Train Loss | Train Perplexity | Val Loss | Val Perplexity |
|-------|------------|------------------|----------|----------------|
| 1 | 3.1501 | 23.3380 | 2.8375 | 17.0723 |
| 2 | 2.9357 | 18.8347 | 2.7899 | 16.2790 |
| 3 | 2.8122 | 16.6459 | 2.7688 | 15.9397 |
| 4 | 2.7149 | 15.1032 | 2.7618 | 15.8282 |

### Loss Progress

![Loss Progress](loss_progress.png)

The model shows consistent improvement in both training and validation loss across all epochs. The validation loss decreases from 2.8375 to 2.7618, indicating the model is learning effectively without significant overfitting.

### Perplexity Progress

![Perplexity Progress](perplexity_progress.png)

Perplexity follows a similar downward trend, with training perplexity dropping from 23.34 to 15.10 and validation perplexity from 17.07 to 15.83. The narrowing gap between training and validation perplexity suggests the model is generalizing well.

## Model Artifacts

The trained model is saved locally in the `saved_model/` directory:
- `model.safetensors` - Model weights in safetensors format
- `config.json` - Model configuration
- `generation_config.json` - Generation parameters

## Training Configuration

The training script includes several optimizations for memory efficiency:
- Mixed precision training (FP16) to reduce memory usage
- Gradient clipping to prevent exploding gradients
- GPU memory fraction limiting to 50% to prevent OOM errors
- Memory fragmentation reduction via `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Cooling delays between training phases to prevent GPU overheating