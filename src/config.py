from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal


@dataclass
class PathConfig:
    input_data: str
    symbolic_output_root: str
    hmm_output_root: str
    score_output_root: str
    figure_output_root: str
    metadata_csv: str
    dataset_npz: str


@dataclass
class PreprocessConfig:
    standardize_method: Literal["zscore", "robust"]
    smooth: bool
    smooth_window: int
    center_diff_on_diff_series: bool
    fill_first_diff: Literal["zero", "repeat"]


@dataclass
class RepresentationConfig:
    mode: Literal["gaussian_2d", "categorical_legacy"]
    use_activation: bool
    use_trend: bool
    activation_thresholds: List[float]
    trend_thresholds: List[float]
    alpha_values: List[float]
    beta_values: List[float]
    activation_encoding: Literal["ternary"]
    trend_encoding: Literal["ternary"]
    feature_standardize: bool = False


@dataclass
class HMMConfig:
    emission_type: Literal["gaussian", "categorical"]
    covariance_type: Literal["full", "diag", "spherical", "tied"]
    n_hidden_states_values: List[int]
    n_iter: int
    tol: float
    random_state: int
    verbose: bool


@dataclass
class ScoreConfig:
    weights: Dict[str, float]
    normalize_scores_across_runs: bool


@dataclass
class VisualizationConfig:
    dpi: int
    fig_format: Literal["png", "pdf", "svg"]
    top_n_score_runs: int
    show_titles: bool
    save_symbolic_distribution: bool
    save_transition_matrix: bool
    save_fo_bar: bool
    save_mdt_bar: bool
    save_score_bar: bool


@dataclass
class ExperimentConfig:
    paths: PathConfig
    preprocess: PreprocessConfig
    representation: RepresentationConfig
    hmm: HMMConfig
    score: ScoreConfig
    visualization: VisualizationConfig


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # backward compatibility:
    # if old config still uses "symbolic", map it to "representation"
    if "representation" not in raw and "symbolic" in raw:
        symbolic = raw["symbolic"]
        raw["representation"] = {
            "mode": "gaussian_2d",
            "use_activation": True,
            "use_trend": True,
            "activation_thresholds": symbolic.get("deviation_thresholds", [1.0]),
            "trend_thresholds": symbolic.get("momentum_thresholds", [1.0]),
            "alpha_values": symbolic.get("alpha_values", [1.0]),
            "beta_values": symbolic.get("beta_values", [1.0]),
            "activation_encoding": "ternary",
            "trend_encoding": "ternary",
            "feature_standardize": False,
        }

    # backward compatibility:
    # old hmm config may not contain emission_type / covariance_type
    if "emission_type" not in raw["hmm"]:
        raw["hmm"]["emission_type"] = "gaussian"
    if "covariance_type" not in raw["hmm"]:
        raw["hmm"]["covariance_type"] = "full"

    return ExperimentConfig(
        paths=PathConfig(**raw["paths"]),
        preprocess=PreprocessConfig(**raw["preprocess"]),
        representation=RepresentationConfig(**raw["representation"]),
        hmm=HMMConfig(**raw["hmm"]),
        score=ScoreConfig(**raw["score"]),
        visualization=VisualizationConfig(**raw["visualization"]),
    )