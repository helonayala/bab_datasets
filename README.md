# bab_datasets

Lightweight dataset loader utilities for nonlinear system ID experiments.

## Install (local)

```bash
pip install -e .
```

## Install (from GitHub)

```bash
pip install git+https://github.com/helonayala/bab_datasets.git
```

## Usage

```python
import bab_datasets as nod

# Load with preprocessing
train_val = nod.load_experiment("multisine_05", preprocess=True, plot=True)

# Unpack
u, y = train_val  # y is the raw signal; yf (if present) is available as data.y_filt
print(train_val)

# Slice
print(train_val[:100])
```

## Notes

- The loader prefers local files under `data/` in this repo.
- If a file is missing, it attempts to download from the configured URL.
- For manual trimming, use `plot=True` and set `end_idx` after visually checking the zoomed plot.
- You can change the zoom window with `zoom_last_n`.
- If present, `ref` is exposed as `data.y_ref` and `yf` is exposed as `data.y_filt` (not used for modeling by default).

## Datasets

- 01_rampa_positiva.mat  -> `rampa_positiva`
- 02_rampa_negativa.mat  -> `rampa_negativa`
- 03_random_steps_01.mat -> `random_steps_01`
- 03_random_steps_02.mat -> `random_steps_02`
- 03_random_steps_03.mat -> `random_steps_03`
- 03_random_steps_04.mat -> `random_steps_04`
- 04_swept_sine.mat      -> `swept_sine`
- 05_multisine_01.mat    -> `multisine_05`
- 06_multisine_02.mat    -> `multisine_06`
