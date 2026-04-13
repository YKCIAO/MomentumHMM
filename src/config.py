from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Dict


@dataclass
class PathConfig:
    input_data: str
    symbolic_output_root: str
    hmm_output_root: str
    score_output_root: str
    figure_output_root: str


@dataclass
class PreprocessConfig:
    standardize_method: Literal["zscore", "robust"]
    smooth: bool
    smooth_window: int
    center_diff_on_diff_series: bool
    fill_first_diff: Literal["zero", "repeat"]


@dataclass
class SymbolicConfig:
    deviation_thresholds: List[float]
    momentum_thresholds: List[float]
    alpha_values: List[float]
    beta_values: List[float]
    category_mode: Literal["pair_index", "weighted_rank"]


@dataclass
class HMMConfig:
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
    symbolic: SymbolicConfig
    hmm: HMMConfig
    score: ScoreConfig
    visualization: VisualizationConfig


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return ExperimentConfig(
        paths=PathConfig(**raw["paths"]),
        preprocess=PreprocessConfig(**raw["preprocess"]),
        symbolic=SymbolicConfig(**raw["symbolic"]),
        hmm=HMMConfig(**raw["hmm"]),
        score=ScoreConfig(**raw["score"]),
        visualization=VisualizationConfig(**raw["visualization"]),
    )