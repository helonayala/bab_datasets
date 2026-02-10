# bab_datasets

Lightweight loaders for the ball-and-beam (BAB) system ID dataset, including proprioceptive signals, synced video, and alignment utilities.

## Install (from GitHub)

```bash
pip install git+https://github.com/helonayala/bab_datasets.git
```

To install with extras from GitHub:

```bash
# proprioceptive only (core)
pip install "git+https://github.com/helonayala/bab_datasets.git"

# core + video support
pip install "git+https://github.com/helonayala/bab_datasets.git#egg=bab_datasets[video]"

# core + video + notebook
pip install "git+https://github.com/helonayala/bab_datasets.git#egg=bab_datasets[video,notebook]"
```

## Usage

```python
import bab_datasets as nod

# Load with preprocessing
train_val = nod.load_experiment(
    "multisine_05",
    preprocess=True,
    plot=True,
    end_ref_tolerance=1e-8,
    y_dot_method="savgol",
)

# Unpack
u, y, y_ref, y_dot = train_val  # y is raw; y_ref is reference; y_dot from Savitzky-Golay
print(train_val)

# Slice
print(train_val[:100])
```

## Usage Examples

### 1) Proprioceptive data only

Then load a dataset:

```python
import bab_datasets as nod

data = nod.load_experiment("swept_sine", preprocess=True)
t = data.time_vector()
```

See the demo notebook: [demo_bab_datasets.ipynb](demo_bab_datasets.ipynb).

### 2) Proprioceptive + video (synced)

Install with video + notebook extras:

```bash
pip install -e .[video,notebook]
```

Then load synced video frames:

```python
import bab_datasets as nod

data = nod.load_experiment("swept_sine", preprocess=True)
video_t, frames, sync = nod.load_synced_video_frames(
    video_name="swept_sine",
    data=data,
)
```

See the demo notebook: [demo_video_sync.ipynb](demo_video_sync.ipynb).

### Sync Preview (GIF)

![Sync preview (middle)](figures/sync_preview_middle.gif)
![Sync preview (start)](figures/sync_preview_start.gif)

## Notes

- The loader prefers local files under `data/` in this repo.
- If a file is missing, it attempts to download from the configured URL.
- The end index defaults to the sample right before `y_ref` goes to zero.
- For manual trimming, use `plot=True` and set `end_idx` after visually checking the zoomed plots.
- You can change the zoom window size with `zoom_last_n` (default: 200).
- If present, `ref` is exposed as `data.y_ref` and `yf` is exposed as `data.y_filt` (not used for modeling by default).

## Video Add-on (Optional)

The video utilities are optional and do not affect the core dataset API. They require OpenCV.

Install with the `video` extra to include OpenCV (and other video dependencies).

Set `BAB_DATASETS_VIDEO_DIR` to the folder that contains the `.MOV` files, or pass `video_dir` explicitly.

```python
import bab_datasets as nod

sync = nod.sync_video_with_dataset(
    video_name="swept_sine",
    dataset_name="swept_sine",
    video_dir="/absolute/path/to/videos_BAB",
    roi="lower_left",
    roi_frac=0.25,
)

print(sync.frame_start, sync.sample_start)
```

The sync uses the LED onset in the lower-left quadrant and aligns it with the trigger in the dataset.

Angle extraction and modeling are intended to live in a separate `hybrid_sysid` project (to be linked once published).

See `demo_video_sync.ipynb` for a quick sync walkthrough.

## Datasets

Short descriptions and naming rationale:

- **rampa_positiva / rampa_negativa**: Semi‑static ramp tests with monotonic reference changes (positive or negative).
- **random_steps_01..04**: Random step reference sequences for beam position, with increasing step rates across runs.
- **swept_sine**: Broadband swept‑sine excitation for system identification.
- **multisine_05 / multisine_06**: Broadband multisine excitation, repeated twice with different random phases.

- 01_rampa_positiva.mat  -> `rampa_positiva`
- 02_rampa_negativa.mat  -> `rampa_negativa`
- 03_random_steps_01.mat -> `random_steps_01`
- 03_random_steps_02.mat -> `random_steps_02`
- 03_random_steps_03.mat -> `random_steps_03`
- 03_random_steps_04.mat -> `random_steps_04`
- 04_swept_sine.mat      -> `swept_sine`
- 05_multisine_01.mat    -> `multisine_05`
- 06_multisine_02.mat    -> `multisine_06`
