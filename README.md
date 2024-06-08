# SVD-KV

## Setup
```bash
conda create -n kv_svd python==3.9
conda activate kv_svd

pip install -e .
pip install -r requirements.txt
```

## Train
```bash
cd svdkv_src/scripts
python train.py [args]
```

## Inference
```bash
python demo.py [args]
```

