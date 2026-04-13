from __future__ import annotations

import matplotlib.pyplot as plt


def plot_top_score_runs(
    scored_runs: list[dict],
    top_n: int,
    show_titles: bool,
):
    top_runs = scored_runs[:top_n]
    labels = [f"K={r['n_hidden_states']}\n{r['symbolic_dir'].split('/')[-1]}" for r in top_runs]
    values = [r["final_score"] for r in top_runs]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(top_runs))))
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Final score")
    if show_titles:
        ax.set_title(f"Top {top_n} scoring runs")
    return fig