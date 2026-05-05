from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def set_academic_style(font_size: int = 16) -> None:
    """
    Set a clean academic plotting style.

    Purpose:
        Make figures suitable for manuscripts or presentations:
        larger fonts, bold labels, clear axes, high readability.
    """
    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "figure.titlesize": font_size + 4,
        "figure.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, out_path: str | Path, dpi: int = 300) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_npz_scalar(npz, key: str, default=None):
    if key not in npz.files:
        return default
    value = npz[key]
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Compute Pearson r and p-value.

    Uses scipy if available. If scipy is unavailable, returns p=nan.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan, np.nan

    try:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        return float(r), float(p)
    except Exception:
        r = np.corrcoef(x, y)[0, 1]
        return float(r), np.nan


def add_regression_line(ax, x: np.ndarray, y: np.ndarray) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return

    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return

    coef = np.polyfit(x, y, 1)
    x_grid = np.linspace(np.min(x), np.max(x), 100)
    y_grid = coef[0] * x_grid + coef[1]
    ax.plot(x_grid, y_grid, linewidth=2.5)


def annotate_r_p(ax, x: np.ndarray, y: np.ndarray, loc=(0.05, 0.92)) -> None:
    r, p = pearson_r_p(x, y)
    if np.isfinite(r):
        if np.isfinite(p):
            text = f"r = {r:.2f}\np = {p:.2e}"
        else:
            text = f"r = {r:.2f}"
    else:
        text = "r = NA"

    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"),
    )