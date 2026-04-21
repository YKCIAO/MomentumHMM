import os
import re
import glob
import numpy as np
import pandas as pd

# =========================
# 1. 改成你的根目录
# =========================
ROOT_DIR = r"D:\CodeHome\python\MomentumHMM\outputs\symbolic_search"
OUTPUT_EXCEL = os.path.join(ROOT_DIR, "symbolic_search_summary.xlsx")


def find_key(npz, candidates):
    for k in candidates:
        if k in npz:
            return k
    return None


def calc_trit_stats(arr):
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
        "neg1_ratio": neg1 / total,
        "zero_ratio": zero / total,
        "pos1_ratio": pos1 / total,
        "unexpected_values": str(unexpected) if len(unexpected) > 0 else ""
    }


def calc_category_stats(arr):
    arr = np.asarray(arr).ravel()
    total = arr.size
    if total == 0:
        return None

    unique, counts = np.unique(arr, return_counts=True)
    ratio_map = {int(u): c / total for u, c in zip(unique, counts)}

    out = {
        "category_total_points": int(total),
        "category_n_present": len(unique),
        "category_max_ratio": max(ratio_map.values())
    }

    # 统计活跃类和稀有类
    out["category_n_active_gt5pct"] = sum(v > 0.05 for v in ratio_map.values())
    out["category_n_rare_lt1pct"] = sum(v < 0.01 for v in ratio_map.values())

    # 每个类别单独展开
    for u in sorted(unique):
        out[f"cat_{int(u)}_ratio"] = ratio_map[int(u)]

    return out


def evaluate_deviation(dev_stats):
    if dev_stats is None:
        return "missing"

    z = dev_stats["zero_ratio"]
    n = dev_stats["neg1_ratio"]
    p = dev_stats["pos1_ratio"]

    comments = []

    if z < 0.40:
        comments.append("0太少(偏敏感)")
    elif z > 0.80:
        comments.append("0太多(偏迟钝)")
    else:
        comments.append("0比例可接受")

    if abs(n - p) > 0.10:
        comments.append("±1不平衡")
    else:
        comments.append("±1较平衡")

    return "; ".join(comments)


def evaluate_momentum(mom_stats):
    if mom_stats is None:
        return "missing"

    z = mom_stats["zero_ratio"]
    n = mom_stats["neg1_ratio"]
    p = mom_stats["pos1_ratio"]

    comments = []

    if z < 0.50:
        comments.append("0太少(动量过敏)")
    elif z > 0.85:
        comments.append("0太多(动量信息偏弱)")
    else:
        comments.append("0比例可接受")

    if abs(n - p) > 0.10:
        comments.append("±1不平衡")
    else:
        comments.append("±1较平衡")

    return "; ".join(comments)


def evaluate_category(cat_stats):
    if cat_stats is None:
        return "missing"

    comments = []

    if cat_stats["category_n_present"] <= 3:
        comments.append("类别明显塌缩")
    elif cat_stats["category_n_present"] >= 8:
        comments.append("类别覆盖较广")
    else:
        comments.append("类别覆盖中等")

    if cat_stats["category_max_ratio"] > 0.40:
        comments.append("最大类别占比过高")
    else:
        comments.append("无极端dominance")

    if cat_stats["category_n_active_gt5pct"] < 5:
        comments.append("活跃类别偏少")
    else:
        comments.append("活跃类别尚可")

    return "; ".join(comments)


def overall_recommendation(dev_eval, mom_eval, cat_eval):
    text = " | ".join([dev_eval, mom_eval, cat_eval])

    bad = 0
    for kw in ["偏敏感", "偏迟钝", "动量过敏", "动量信息偏弱", "类别明显塌缩", "最大类别占比过高"]:
        if kw in text:
            bad += 1

    if bad == 0:
        return "较推荐"
    elif bad <= 2:
        return "可人工复核"
    else:
        return "优先排除"


def parse_folder_params(folder_name):
    """
    解析类似：
    dev_0.5000__mom_1.0000__a_0.7000__b_0.4000
    """
    result = {
        "deviation_threshold": None,
        "momentum_threshold": None,
        "a_weight": None,
        "b_weight": None
    }

    patterns = {
        "deviation_threshold": r"dev_([0-9.]+)",
        "momentum_threshold": r"mom_([0-9.]+)",
        "a_weight": r"a_([0-9.]+)",
        "b_weight": r"b_([0-9.]+)"
    }

    for key, pat in patterns.items():
        m = re.search(pat, folder_name)
        if m:
            result[key] = float(m.group(1))

    return result


def analyze_one_npz(npz_path):
    folder_name = os.path.basename(os.path.dirname(npz_path))
    row = {
        "folder_name": folder_name,
        "npz_path": npz_path
    }

    row.update(parse_folder_params(folder_name))

    data = np.load(npz_path, allow_pickle=True)

    dev_key = find_key(data, ["deviation_code", "dev_code", "deviation"])
    mom_key = find_key(data, ["momentum_code", "mom_code", "momentum"])
    cat_key = find_key(data, ["category_9", "category", "obs_seq", "observations"])

    row["deviation_key"] = dev_key if dev_key else ""
    row["momentum_key"] = mom_key if mom_key else ""
    row["category_key"] = cat_key if cat_key else ""

    dev_stats = calc_trit_stats(data[dev_key]) if dev_key else None
    mom_stats = calc_trit_stats(data[mom_key]) if mom_key else None
    cat_stats = calc_category_stats(data[cat_key]) if cat_key else None

    if dev_stats:
        for k, v in dev_stats.items():
            row[f"dev_{k}"] = v

    if mom_stats:
        for k, v in mom_stats.items():
            row[f"mom_{k}"] = v

    if cat_stats:
        for k, v in cat_stats.items():
            row[k] = v

    dev_eval = evaluate_deviation(dev_stats)
    mom_eval = evaluate_momentum(mom_stats)
    cat_eval = evaluate_category(cat_stats)

    row["deviation_eval"] = dev_eval
    row["momentum_eval"] = mom_eval
    row["category_eval"] = cat_eval
    row["overall_recommendation"] = overall_recommendation(dev_eval, mom_eval, cat_eval)

    return row


def main():
    npz_files = glob.glob(os.path.join(ROOT_DIR, "*", "symbolic_outputs.npz"))

    if len(npz_files) == 0:
        print(f"没有找到 symbolic_outputs.npz: {ROOT_DIR}")
        return

    rows = []
    for f in sorted(npz_files):
        try:
            row = analyze_one_npz(f)
            rows.append(row)
            print(f"[OK] {f}")
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

    df = pd.DataFrame(rows)

    priority_cols = [
        "folder_name",
        "deviation_threshold", "momentum_threshold", "a_weight", "b_weight",
        "dev_neg1_ratio", "dev_zero_ratio", "dev_pos1_ratio",
        "mom_neg1_ratio", "mom_zero_ratio", "mom_pos1_ratio",
        "category_n_present", "category_n_active_gt5pct", "category_n_rare_lt1pct",
        "category_max_ratio",
        "deviation_eval", "momentum_eval", "category_eval",
        "overall_recommendation",
        "npz_path"
    ]

    cols = [c for c in priority_cols if c in df.columns] + [c for c in df.columns if c not in priority_cols]
    df = df[cols]

    df = df.sort_values(["deviation_threshold", "momentum_threshold"], na_position="last")

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="summary")

    print("\n完成。结果已保存到：")
    print(OUTPUT_EXCEL)


if __name__ == "__main__":
    main()