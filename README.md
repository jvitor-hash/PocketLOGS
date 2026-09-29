![PocketLOGS](ascii-art-text.png)

### PocketLOGS-char-10.66M-Base

## Description
This is an experimental model used as a learning exercise. Use at your own caution.
Some of the logs generated can be cut off due to the small context length.

## Model information

| Field | Value |
| --- | --- |
| Model name | PocketLOGS-char-10.66M-Base |
| Context length | 256 |
| Vocabulary size | 91 |
| Trained dataset | [Loghub](https://github.com/logpai/loghub) (Apache, Linux, Mac, Proxifier) |
| Tool used to train | [nanoGPT](https://github.com/karpathy/nanoGPT) |
| Version | v1 |
| Date of training | 2026-09-29 |
| Iterations training | 700 |

## Training Summary

Training time: 2hrs (due to some technical difficulties)

| Iter | Train | Val | Gap |
| ---: | ---: | ---: | ---: |
| 100 | 2.4794 | 2.5059 | +0.027 |
| 200 | 1.6485 | 1.8839 | +0.235 |
| 300 | 0.9945 | 1.1639 | +0.169 |
| 400 | 0.6002 | 0.8784 | +0.278 |
| 500 | 0.3690 | 0.7107 | +0.342 |
| 600 | 0.2578 | 0.6572 | +0.399 |
| 700 | 0.2053 | 0.6190 | +0.414 |

## Model Usage

```bash
$ python -m venv .venv
$ source .venv/bin/activate  # Depends on your system
$ pip3 install -r requirements.txt
$ python3 inference.py
```

### Examples

```bash
$ python inference.py \
    --model out-logs_char/model_release.pt \
    --prompt "2025-01-01" \
    --tokens 500 \
    --temperature 0.7 \
    --top_k 100 \
    --samples 3 \
    --seed 42
```

### Full Options

```text
--model PATH        Path to model .pt file            (default: pocketlogs-char-10.66M.pt)
--meta PATH         Path to meta.pkl                  (default: meta.pkl next to --model)
--prompt TEXT       Seed text                         (default: "2025")
--tokens N          Number of tokens to generate      (default: 300)
--temperature F     Sampling temperature              (default: 0.8)
--top_k N           Top-k sampling cutoff             (default: 200)
--samples N         Number of samples to draw         (default: 1)
--device {auto,cpu,cuda}                              (default: auto)
--seed N            Random seed for reproducibility   (default: None)
```

### Typical output

```text
# device=cpu  params=10,663,168
# block_size=1024  vocab_size=68
# prompt='2025'  tokens=300  temp=0.8  top_k=200

--- sample 1 ---
```

## Configuration used for training

```python
# Data
dataset = 'logs_char'
data_dir = f'data/{dataset}'
out_dir = f'out-{dataset}'

# Model architecture — small enough for 6GB VRAM
block_size = 256        # context length in characters (covers ~1-2 log lines)
n_layer = 6             # number of transformer layers
n_head = 6              # attention heads per layer
n_embd = 384            # embedding dimension
dropout = 0.1           # no dropout for small models
bias = False            # no bias in linear layers (slightly faster)

# Training
batch_size = 32         # start here; reduce to 16 if OOM
learning_rate = 5e-4    # for small models, higher LR works
max_iters = 5000        # total training steps
lr_decay_iters = 5000   # match max_iters
min_lr = 1e-4           # minimum learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
always_save_checkpoint = True

# Learning rate schedule
decay_lr = True
warmup_iters = 1000

# Evaluation
eval_interval = 100
eval_iters = 50
log_interval = 10

# System
device = 'cuda'
dtype = 'float32'
compile = True
```