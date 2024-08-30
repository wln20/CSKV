# CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
## Abstract
Large Language Models (LLMs) have been widely adopted to process long-context tasks.
However, the large memory overhead of the key-value (KV) cache poses significant challenges in long-context scenarios.
Existing training-free KV cache compression methods typically focus on quantization and token pruning, which have compression limits, and excessive sparsity can lead to severe performance degradation.
Other methods design new architectures with le
ss KV overhead but require significant training overhead.
To address the above two drawbacks, we further explore the redundancy in the channel dimension and apply an architecture-level design with minor training costs.
Therefore, we introduce **CSKV**, a training-efficient **C**hannel **S**hrinking technique for **KV** cache compression:
(1) We first analyze the singular value distribution of the KV cache, revealing significant redundancy and compression potential along the channel dimension.
Based on this observation, we propose using low-rank decomposition for key and value layers and storing the low-dimension features.
(2) To preserve model performance, we introduce a bi-branch KV cache, including a window-based full-precision KV cache and a low-precision compressed KV cache.
(3) To reduce the training costs, we minimize the layer-wise reconstruction loss for the compressed KV cache instead of retraining the entire LLMs.
Extensive experiments show that **CSKV** can reduce the memory overhead of the KV cache by 80% while maintaining the model's long-context capability.
Moreover, we show that our method can be seamlessly combined with quantization to further reduce the memory overhead, achieving a compression ratio of up to 95%.

## Setup
Run the following commands to install required packages:
```bash
conda create -n cskv python==3.9
conda activate cskv
pip install -e .
```
If you intend to use the flash-attention implementation, please install [flash_attn](https://github.com/Dao-AILab/flash-attention) according to the official guidance.

In the following sections, we would use `longchat-7b-v1.5-32k` as an example to show the usage of this repo.

## Fine-tune
First generate the required clibration statistics of ASVD: 
```bash
cd cskv_src/scripts
python asvd_init_calib.py --model_path lmsys/longchat-7b-v1.5-32k --model_id longchat-7b-v1.5-32k 
```

The generated data would be saved in `../data/asvd_data/asvd_init_ckpts/{model_id}/` by default.

Start fine-tuning:
```bash
python train.py --model_path lmsys/longchat-7b-v1.5-32k \
--model_id longchat-7b-v1.5-32k \
--k_density 0.5 \
--v_density 0.5 \
--use_asvd \
--use_window \
--k_bits 16 \
--v_bits 16
```

The checkpoint would be saved in `../data/kvcache_compressor_checkpoints/{model_id}/` by default.

## Inference with fine-tuned model
```bash
python demo.py --model_path lmsys/longchat-7b-v1.5-32k \
--model_id longchat-7b-v1.5-32k \
--k_density 0.5 \
--v_density 0.5 \
--use_asvd \
--use_window \
--k_bits 16 \
--v_bits 16
```

