import os
import re
import glob
import numpy as np
import pandas as pd

# =========================
# 1. 改成你的 2.0 representation 根目录
# =========================
ROOT_DIR = r"D:\CodeHome\python\MomentumHMM\outputs\representation"
OUTPUT_EXCEL = os.path.join(ROOT_DIR, "representation_2d_quality_summary.xlsx")


def find_key(npz, candidates):
    for k in candidates:
        if k in npz.files:
            return k
    return None


def safe_scalar(npz, key, default=None):
    if key not in npz.files:
        return default
    arr = npz[key]
    if np.asarray(arr).size == 1:
        return np.asarray(arr).item()
    return arr


def calc_ternary_stats(arr):
    arr = np.asarray(arr).ravel()
    total = arr.size
    if total == 0:
        return None

    neg1 = np.sum(arr == -1)
    zero = np.sum(arr == 0)
    pos1 = np.sum(arr == 1)

    unique_vals = np.unique(arr)
    unexpected = [x for x in unique_vals if x not in (-1, 0, 1)]

    return {
        "total_points": int(total),
        "neg1_count": int(neg1),
        "zero_count": int(zero),
        "pos1_count": int(pos1),
        "neg1_ratio": float(neg1 / total),
        "zero_ratio": float(zero / total),
        "pos1_ratio": float(pos1 / total),
        "balance_abs_diff_pos_neg": float(abs(pos1 - neg1) / total),
        "unexpected_values": str(unexpected) if len(unexpected) > 0 else "",
    }


def calc_x_stats(X):
    X = np.asarray(X)

    if X.ndim != 2:
        return {
            "X_shape": str(X.shape),
            "X_valid": False,
            "X_problem": "X不是二维矩阵",
        }

    out = {
        "X_shape": str(X.shape),
        "X_valid": True,
        "X_n_timepoints_total": int(X.shape[0]),
        "X_n_features": int(X.shape[1]),
        "X_has_nan": bool(np.isnan(X).any()),
        "X_has_inf": bool(np.isinf(X).any()),
    }

    for j in range(X.shape[1]):
        col = X[:, j]
        out[f"X_f{j}_mean"] = float(np.nanmean(col))
        out[f"X_f{j}_std"] = float(np.nanstd(col))
        out[f"X_f{j}_min"] = float(np.nanmin(col))
        out[f"X_f{j}_max"] = float(np.nanmax(col))
        out[f"X_f{j}_n_unique"] = int(len(np.unique(col[np.isfinite(col)])))

    return out


def evaluate_activation(stats):
    if stats is None:
        return "missing"

    z = stats["zero_ratio"]
    n = stats["neg1_ratio"]
    p = stats["pos1_ratio"]

    comments = []

    if z < 0.40:
        comments.append("0太少(activation阈值偏低/过敏)")
    elif z > 0.85:
        comments.append("0太多(activation阈值偏高/信息偏弱)")
    else:
        comments.append("0比例可接受")

    if abs(n - p) > 0.10:
        comments.append("±1不平衡")
    else:
        comments.append("±1较平衡")

    if p + n < 0.10:
        comments.append("非零activation太少")
    elif p + n > 0.60:
        comments.append("非零activation偏多")
    else:
        comments.append("非零activation适中")

    return "; ".join(comments)


def evaluate_trend(stats):
    if stats is None:
        return "missing"

    z = stats["zero_ratio"]
    n = stats["neg1_ratio"]
    p = stats["pos1_ratio"]

    comments = []

    if z < 0.50:
        comments.append("0太少(trend阈值偏低/动量过敏)")
    elif z > 0.90:
        comments.append("0太多(trend阈值偏高/动量信息偏弱)")
    else:
        comments.append("0比例可接受")

    if abs(n - p) > 0.10:
        comments.append("±1不平衡")
    else:
        comments.append("±1较平衡")

    if p + n < 0.08:
        comments.append("非零trend太少")
    elif p + n > 0.50:
        comments.append("非零trend偏多")
    else:
        comments.append("非零trend适中")

    return "; ".join(comments)


def evaluate_x_stats(x_stats):
    if x_stats is None:
        return "missing"

    if not x_stats.get("X_valid", False):
        return x_stats.get("X_problem", "X invalid")

    comments = []

    if x_stats["X_has_nan"]:
        comments.append("X含NaN")
    if x_stats["X_has_inf"]:
        comments.append("X含Inf")

    n_features = x_stats["X_n_features"]

    if n_features == 1:
        comments.append("单通道模型")
    elif n_features == 2:
        comments.append("二维joint模型")
    else:
        comments.append(f"异常特征数={n_features}")

    low_var_channels = 0
    for j in range(n_features):
        std = x_stats.get(f"X_f{j}_std", np.nan)
        n_unique = x_stats.get(f"X_f{j}_n_unique", 0)

        if np.isfinite(std) and std < 1e-6:
            low_var_channels += 1

        if n_unique <= 1:
            comments.append(f"f{j}完全塌缩")
        elif n_unique <= 2:
            comments.append(f"f{j}取值偏少")
        else:
            comments.append(f"f{j}有变化")

    if low_var_channels > 0:
        comments.append(f"{low_var_channels}个通道低方差")

    if len(comments) == 0:
        comments.append("X基本正常")

    return "; ".join(comments)


def overall_recommendation(act_eval, trend_eval, x_eval):
    text = " | ".join([str(act_eval), str(trend_eval), str(x_eval)])

    bad_keywords = [
        "missing",
        "过敏",
        "信息偏弱",
        "非零activation太少",
        "非零trend太少",
        "X含NaN",
        "X含Inf",
        "完全塌缩",
        "低方差",
        "异常特征数",
    ]

    bad = sum(kw in text for kw in bad_keywords)

    if bad == 0:
        return "较推荐"
    elif bad <= 2:
        return "可人工复核"
    else:
        return "优先排除"


def parse_folder_params(folder_name):
    """
    解析 2.0 文件夹名，例如：
    act_1.0000__trend_1.1000__a_1.0000__b_1.0000
    """
    result = {
        "activation_threshold": None,
        "trend_threshold": None,
        "alpha": None,
        "beta": None,
    }

    patterns = {
        "activation_threshold": r"act_([0-9.]+)",
        "trend_threshold": r"trend_([0-9.]+)",
        "alpha": r"a_([0-9.]+)",
        "beta": r"b_([0-9.]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, folder_name)
        if m:
            result[key] = float(m.group(1))

    return result


def infer_model_type(alpha, beta, feature_names=None):
    if feature_names is not None:
        try:
            names = [str(x) for x in feature_names]
            return "+".join(names)
        except Exception:
            pass

    if alpha is not None and beta is not None:
        if alpha > 0 and beta > 0:
            return "activation+trend"
        if alpha > 0 and beta == 0:
            return "activation_only"
        if alpha == 0 and beta > 0:
            return "trend_only"

    return "unknown"


def analyze_one_run(folder):
    folder_name = os.path.basename(folder)

    row = {
        "folder_name": folder_name,
        "folder_path": folder,
    }

    row.update(parse_folder_params(folder_name))

    rep_npz_path = os.path.join(folder, "representation_outputs.npz")
    hmm_npz_path = os.path.join(folder, "hmm_ready_features.npz")

    row["representation_npz_path"] = rep_npz_path if os.path.exists(rep_npz_path) else ""
    row["hmm_ready_npz_path"] = hmm_npz_path if os.path.exists(hmm_npz_path) else ""

    if not os.path.exists(rep_npz_path):
        raise FileNotFoundError(f"Missing representation_outputs.npz: {rep_npz_path}")

    if not os.path.exists(hmm_npz_path):
        raise FileNotFoundError(f"Missing hmm_ready_features.npz: {hmm_npz_path}")

    rep = np.load(rep_npz_path, allow_pickle=True)
    hmm = np.load(hmm_npz_path, allow_pickle=True)

    # 2.0 key names
    act_key = find_key(rep, ["activation_code", "deviation_code", "dev_code"])
    trend_key = find_key(rep, ["trend_code", "momentum_code", "mom_code"])
    X_key = find_key(hmm, ["X"])

    feature_names = hmm["feature_names"] if "feature_names" in hmm.files else None

    # Prefer npz scalar values over folder parsing
    for key_in_npz, key_in_row in [
        ("activation_threshold", "activation_threshold"),
        ("trend_threshold", "trend_threshold"),
        ("alpha", "alpha"),
        ("beta", "beta"),
    ]:
        val = safe_scalar(hmm, key_in_npz, default=None)
        if val is not None:
            row[key_in_row] = float(val)

    row["activation_key"] = act_key if act_key else ""
    row["trend_key"] = trend_key if trend_key else ""
    row["X_key"] = X_key if X_key else ""

    if feature_names is not None:
        row["feature_names"] = ",".join([str(x) for x in feature_names])
    else:
        row["feature_names"] = ""

    row["model_type"] = infer_model_type(row["alpha"], row["beta"], feature_names)

    act_stats = calc_ternary_stats(rep[act_key]) if act_key else None
    trend_stats = calc_ternary_stats(rep[trend_key]) if trend_key else None
    x_stats = calc_x_stats(hmm[X_key]) if X_key else None

    if act_stats:
        for k, v in act_stats.items():
            row[f"act_{k}"] = v

    if trend_stats:
        for k, v in trend_stats.items():
            row[f"trend_{k}"] = v

    if x_stats:
        for k, v in x_stats.items():
            row[k] = v

    act_eval = evaluate_activation(act_stats)
    trend_eval = evaluate_trend(trend_stats)
    x_eval = evaluate_x_stats(x_stats)

    row["activation_eval"] = act_eval
    row["trend_eval"] = trend_eval
    row["X_eval"] = x_eval
    row["overall_recommendation"] = overall_recommendation(act_eval, trend_eval, x_eval)

    return row


def main():
    run_folders = sorted([
        f for f in glob.glob(os.path.join(ROOT_DIR, "*"))
        if os.path.isdir(f)
    ])

    if len(run_folders) == 0:
        print(f"没有找到 representation run folders: {ROOT_DIR}")
        return

    rows = []
    for folder in run_folders:
        try:
            row = analyze_one_run(folder)
            rows.append(row)
            print(f"[OK] {folder}")
        except Exception as e:
            print(f"[ERROR] {folder}: {e}")

    if len(rows) == 0:
        print("没有成功分析任何 run。")
        return

    df = pd.DataFrame(rows)

    priority_cols = [
        "folder_name",
        "model_type",
        "activation_threshold",
        "trend_threshold",
        "alpha",
        "beta",
        "feature_names",

        "act_neg1_ratio",
        "act_zero_ratio",
        "act_pos1_ratio",
        "act_balance_abs_diff_pos_neg",

        "trend_neg1_ratio",
        "trend_zero_ratio",
        "trend_pos1_ratio",
        "trend_balance_abs_diff_pos_neg",

        "X_shape",
        "X_n_features",
        "X_has_nan",
        "X_has_inf",
        "X_f0_mean",
        "X_f0_std",
        "X_f0_n_unique",
        "X_f1_mean",
        "X_f1_std",
        "X_f1_n_unique",

        "activation_eval",
        "trend_eval",
        "X_eval",
        "overall_recommendation",

        "representation_npz_path",
        "hmm_ready_npz_path",
    ]

    cols = [c for c in priority_cols if c in df.columns] + [
        c for c in df.columns if c not in priority_cols
    ]
    df = df[cols]

    sort_cols = [c for c in ["activation_threshold", "trend_threshold", "alpha", "beta"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="summary")

    print("\n完成。2.0 representation 质量检查结果已保存到：")
    print(OUTPUT_EXCEL)


if __name__ == "__main__":
    main()