# Related-key MCP Attention Inception-based Neural Distinguisher

This repository contains the code accompanying the research paper  
**"Related-Key Multi-Pair Neural Distinguishers: Analysis and Applications to Lightweight Block Ciphers"**.  

Supported block ciphers:
- **PRESENT-80** (SPN)
- **SIMECK-32/64** (Feistel)
- **LEA-128** (ARX)
- **HIGHT** (Generalised Feistel)

## Requirements and setup

This project uses TensorFlow/Keras and optionally CuPy. On Windows, use a compatible CUDA toolkit to enable CuPy for GPU acceleration. Steps below assume Command Prompt (cmd.exe).

1) Install core dependencies
```bat
pip install --upgrade pip
pip install -r requirements.txt
```

2) Install CuPy matching your CUDA version (RECOMMENDED for GPU)
- CUDA 12.x: `pip install cupy-cuda12x`
- CUDA 11.x: `pip install cupy-cuda11x`

---

## Usage

### Related-Key Bit Selection
```bash
python finding_key_bit.py --cipher-name <cipher_name> --input-diff <input_difference> --csv-output <file_name>.csv
```
- For example:
  ```bash
  python finding_key_bit.py --cipher-name present80 --input-diff 0x80 --csv-output results_present.csv
  ```
 - Example result:
  ```bash
  Running sweep on cipher present80 with input difference 0x80...
  Round 1: searching best delta-key bit (input_diff=0x80)...
  Searching for best delta_key (Hamming weight = 1):

  Best delta_key bit: 33 with score = 0.77987
      → best_bit=33, score=0.77987
  Round 2: searching best delta-key bit (input_diff=0x80)...
  Searching for best delta_key (Hamming weight = 1)
  ...
  Round 8: searching best delta-key bit (input_diff=0x80)...
  Searching for best delta_key (Hamming weight = 1):

  Best delta_key bit: 56 with score = 0.00441
      → best_bit=56, score=0.00441
  Stopping: best_score 0.00441 < threshold 0.10000

  Summary (round, best_bit, score):
  1 33 0.77987
  2 56 0.77555
  3 56 0.77577
  4 56 0.77023
  5 56 0.74208
  6 56 0.43150
  7 56 0.10608
  8 56 0.00441
  Exported sweep results to: results_present.csv
  ```
### Geometric Dataset Analysis
``` bash
python visualize_dataset.py --cipher <cipher_name> --scenario related-key --rounds <num_rounds> --pairs <num_pairs> --difference <input_difference> --samples <num_samples> --plot
```
- Example:
  ```bash
  python visualize_dataset.py --cipher simmeck3264 --scenario single-key --rounds 7 --pairs 8 --difference 0x20 --samples 50000 --plot
  ```
 -> Results wil be saved to `analysis_results/simmeck3264_single-key_r7_p8_0x20`
### Training Neural model
``` bash
python main.py --cipher <cipher_name>  --rounds <num_rounds> --pairs <num_pairs> --difference <input_difference> --delta-key-bit <key_bits>
```
- Example:
  ```bash
  python main.py --cipher lea --rounds 8 --pairs 32 --difference 0x8000000080 --delta-key-bit 94 --num-samples-train 1_000_000 --chunk-size 100_000
  ```
### Staged Training
```bash
python staged_train.py --cipher <cipher_name> --pairs <num_pairs> --difference <input_difference> --delta-key-bit <key_bits> --stages-rounds <stage1_rounds>,<stage2_rounds>,<stage3_rounds> --init-weights <.weights.h5 file directory>
```

- Example:
  ```bash
  python staged_train.py --cipher hight --pairs 32 --difference 0x4000000000000000 --delta-key-bit 97 --stages-rounds 12,14,14 --stages-epochs 20,15,10 --init-weights /checkpoints/hight64/4_pairs/hight_best_12r.weights.h5 --use-chunks --chunk-size 1_000_000
  ```
  + `--use-chunks` & `chunk-size` are optional to avoid OOM while training

###  Evaluating a trained model
Use `eval_nets.py` to run repeated-test evaluation with statistical reporting (Z-score, p-value).
```python
python eval_nets.py --cipher-module cipher.<cipher_name> --model-path <model weights path> --rounds <num_rounds> --pairs <num_pairs> --input-diff <input_difference> --n-repeat <number of repeat> --delta-key-bit <trained_key_bit>
```

- Example:
  ```bash
  python eval_nets.py --cipher-module cipher.simmeck3264 --model-path "/checkpoints/simmeck3264/32_pairs/simmeck3264_best_15r.weights.h5" --rounds 15 --pairs 32 --input-diff 0x40 --n-repeat 5 --delta-key-bit 57
  ```