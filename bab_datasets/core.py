import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from urllib.error import URLError

DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "rampa_positiva": {
        "filename": "01_rampa_positiva.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/01_rampa_positiva.mat",
    },
    "rampa_negativa": {
        "filename": "02_rampa_negativa.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/02_rampa_negativa.mat",
    },
    "random_steps_01": {
        "filename": "03_random_steps_01.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_01.mat",
    },
    "random_steps_02": {
        "filename": "03_random_steps_02.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_02.mat",
    },
    "random_steps_03": {
        "filename": "03_random_steps_03.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_03.mat",
    },
    "random_steps_04": {
        "filename": "03_random_steps_04.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/03_random_steps_04.mat",
    },
    "swept_sine": {
        "filename": "04_swept_sine.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/04_swept_sine.mat",
    },
    "multisine_05": {
        "filename": "05_multisine_01.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/05_multisine_01.mat",
    },
    "multisine_06": {
        "filename": "06_multisine_02.mat",
        "url": "https://raw.githubusercontent.com/helonayala/sysid/main/data/06_multisine_02.mat",
    },
}

ALIASES = {
    "01_rampa_positiva": "rampa_positiva",
    "02_rampa_negativa": "rampa_negativa",
    "03_random_steps_01": "random_steps_01",
    "03_random_steps_02": "random_steps_02",
    "03_random_steps_03": "random_steps_03",
    "03_random_steps_04": "random_steps_04",
    "04_swept_sine": "swept_sine",
    "05_multisine_01": "multisine_05",
    "06_multisine_02": "multisine_06",
}


@dataclass
class InputOutputData:
    name: str
    u: np.ndarray
    y: np.ndarray
    sampling_time: float
    y_ref: Optional[np.ndarray] = None
    y_raw: Optional[np.ndarray] = None
    y_filt: Optional[np.ndarray] = None
    state_initialization_window_length: Optional[int] = None

    def __iter__(self):
        yield self.u
        yield self.y
        yield self.y_ref

    def __getitem__(self, item):
        return InputOutputData(
            name=f"{self.name}[{item}]",
            u=self.u[item],
            y=self.y[item],
            sampling_time=self.sampling_time,
            y_ref=None if self.y_ref is None else self.y_ref[item],
            y_raw=None if self.y_raw is None else self.y_raw[item],
            y_filt=None if self.y_filt is None else self.y_filt[item],
            state_initialization_window_length=self.state_initialization_window_length,
        )

    def __repr__(self) -> str:
        base = (
            f"InputOutputData \"{self.name}\" u.shape={self.u.shape} y.shape={self.y.shape}\n"
            f"sampling_time={self.sampling_time:.6e}"
        )
        if self.y_ref is not None:
            base += f" y_ref.shape={self.y_ref.shape}"
        if self.state_initialization_window_length is not None:
            base += f" state_initialization_window_length={self.state_initialization_window_length}"
        return base


def list_experiments():
    return sorted(DATASET_REGISTRY.keys())


def _resolve_name(name: str) -> str:
    if name in DATASET_REGISTRY:
        return name
    if name in ALIASES:
        return ALIASES[name]
    return name


def _default_data_dir() -> str:
    pkg_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(pkg_dir, "..", "data"))


def _download_if_needed(url: str, filename: str, data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        try:
            urlretrieve(url, path)
        except URLError as exc:
            raise RuntimeError(
                f"Failed to download {filename}. Place the file in {data_dir} and retry."
            ) from exc
    return path


def _load_mat(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    data = scipy.io.loadmat(path)
    t = data["time"].flatten()
    u = data["u"].flatten()
    y = data["y"].flatten()
    trig = data["trigger"].flatten()
    y_filt = data["yf"].flatten() if "yf" in data else None
    y_ref = data["ref"].flatten() if "ref" in data else None
    return t, u, y, trig, y_filt, y_ref


def _find_trigger_start(trig: np.ndarray) -> int:
    idx = np.where(trig != 0)[0]
    if idx.size == 0:
        return 0
    return int(idx[0])


def _plot_end_zoom(y: np.ndarray, n_last: int = 2000):
    n_last = int(max(100, n_last))
    start = max(0, len(y) - n_last)
    plt.figure(figsize=(10, 4))
    plt.plot(y, color="red", label="Output (y)")
    plt.xlim([start, len(y)])
    plt.title(f"Zoom on last {n_last} samples (select end index)")
    plt.xlabel("Sample")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _plot_signals(t, u, y, trig, title: str):
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, u, color="blue", label="Input (u)")
    plt.ylabel("Input")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.title(f"Data: {title}")

    plt.subplot(3, 1, 2)
    plt.plot(t, y, color="red", label="Output (y)")
    plt.ylabel("Output")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, trig, color="black", label="Trigger")
    plt.ylabel("Trigger")
    plt.xlabel("Time (s)")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def load_experiment(
    name: str,
    preprocess: bool = True,
    plot: bool = False,
    end_idx: Optional[int] = None,
    resample_factor: int = 50,
    zoom_last_n: int = 2000,
    data_dir: Optional[str] = None,
) -> InputOutputData:
    """
    Load a dataset by key and return InputOutputData.

    Example:
        import bab_datasets as nod
        data = nod.load_experiment("multisine_05", preprocess=True, plot=True)
        u, y = data
    """
    name = _resolve_name(name)
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_experiments()}")

    entry = DATASET_REGISTRY[name]
    data_dir = data_dir or _default_data_dir()
    path = _download_if_needed(entry["url"], entry["filename"], data_dir)

    t, u, y, trig, y_filt, y_ref = _load_mat(path)

    if plot:
        _plot_signals(t, u, y, trig, entry["filename"])

    ts = float(np.average(np.diff(t)))

    if not preprocess:
        y_out = y
        return InputOutputData(
            name=name,
            u=u,
            y=y_out,
            sampling_time=ts,
            y_ref=y_ref,
            y_raw=y,
            y_filt=y_filt,
        )

    start_idx = _find_trigger_start(trig)

    if end_idx is None:
        if plot:
            _plot_end_zoom(y, n_last=zoom_last_n)
        end_idx = len(u)

    u = u[start_idx:end_idx]
    y = y[start_idx:end_idx]
    if y_filt is not None:
        y_filt = y_filt[start_idx:end_idx]
    if y_ref is not None:
        y_ref = y_ref[start_idx:end_idx]
    t = t[start_idx:end_idx] - t[start_idx]

    if resample_factor and resample_factor > 1:
        u = u[::resample_factor]
        y = y[::resample_factor]
        t = t[::resample_factor]
        if y_filt is not None:
            y_filt = y_filt[::resample_factor]
        if y_ref is not None:
            y_ref = y_ref[::resample_factor]
        ts = float(np.average(np.diff(t)))

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(t, u)
        plt.title("Resampled u")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(t, y, label="y")
        if y_ref is not None:
            plt.plot(t, y_ref, label="y_ref")
        plt.title("Resampled y (and y_ref if available)")
        plt.ylabel("Value")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    y_out = y
    return InputOutputData(
        name=name,
        u=u,
        y=y_out,
        sampling_time=ts,
        y_ref=y_ref,
        y_raw=y,
        y_filt=y_filt,
    )
