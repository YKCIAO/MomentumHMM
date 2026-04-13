import os
import glob
import json
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py


def normalize_id(x, remove_suffix=None):
    """
    Convert subject ID to a clean string for matching.
    """
    if pd.isna(x):
        return None
    x = str(x).strip()
    # optional: remove a fixed suffix
    if remove_suffix is not None and x.endswith(remove_suffix):
        x = x[:-len(remove_suffix)]

    return x


def load_mat_file(mat_path):
    """
    Load one .mat file and return a numpy array of shape (278, 478).
    Supports both traditional MAT and v7.3 MAT files.
    """
    # First try scipy.io.loadmat
    try:
        data = sio.loadmat(mat_path)
        for key, value in data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray):
                if value.shape == (278, 478):
                    return value.astype(np.float32)
                elif value.shape == (478, 278):
                    return value.T.astype(np.float32)

    except NotImplementedError:
        pass
    except Exception as e:
        print(f"[Warning] scipy load failed for {os.path.basename(mat_path)}: {e}")

    # Then try h5py for v7.3 MAT
    try:
        with h5py.File(mat_path, "r") as f:
            for key in f.keys():
                value = f[key]
                if isinstance(value, h5py.Dataset):
                    arr = np.array(value)
                    if arr.shape == (278, 478):
                        return arr.astype(np.float32)
                    elif arr.shape == (478, 278):
                        return arr.T.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to load {mat_path}: {e}")

    raise ValueError(f"No valid 278x478 matrix found in {mat_path}")


def build_dataset(mat_dir, excel_path, output_dir, mat_pattern="*.mat",remove_suffix=None):
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Read Excel ----------
    df = pd.read_excel(excel_path)

    required_cols = ["ID", "Age", "Gender"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in Excel.")

    df["ID"] = df["ID"].apply(lambda x: normalize_id(x))

    # Build a lookup dictionary from Excel
    meta_dict = {}
    for _, row in df.iterrows():
        sid = row["ID"]
        if sid is None:
            continue
        meta_dict[sid] = {
            "Age": row["Age"],
            "Gender": row["Gender"]
        }

    # ---------- Find mat files ----------
    mat_files = sorted(glob.glob(os.path.join(mat_dir, mat_pattern)))
    if len(mat_files) == 0:
        raise FileNotFoundError(f"No .mat files found in {mat_dir}")

    all_data = []
    ids = []
    ages = []
    genders = []
    failed_files = []
    unmatched_files = []

    for mat_path in mat_files:
        filename = os.path.basename(mat_path)
        subj_id = normalize_id(os.path.splitext(filename)[0], remove_suffix=remove_suffix)  # file name without .mat

        if subj_id not in meta_dict:
            print(f"[Unmatched] {subj_id} not found in Excel")
            unmatched_files.append(subj_id)
            continue

        try:
            ts = load_mat_file(mat_path)

            if ts.shape != (278, 478):
                raise ValueError(f"Final shape is {ts.shape}, expected (278, 478)")

            all_data.append(ts)
            ids.append(subj_id)
            ages.append(meta_dict[subj_id]["Age"])
            genders.append(meta_dict[subj_id]["Gender"])

            print(f"[OK] {subj_id}: {ts.shape}")

        except Exception as e:
            print(f"[Failed] {subj_id}: {e}")
            failed_files.append({
                "file": filename,
                "error": str(e)
            })

    if len(all_data) == 0:
        raise RuntimeError("No valid subject data were loaded.")

    # ---------- Stack ----------
    X = np.stack(all_data, axis=0).astype(np.float32)   # [N, 278, 478]
    ids = np.array(ids, dtype=object)
    ages = np.array(ages)
    genders = np.array(genders, dtype=object)

    print("\nFinal dataset shape:", X.shape)

    # ---------- Save main array ----------
    np.save(os.path.join(output_dir, "timeseries.npy"), X)

    # ---------- Save metadata table ----------
    metadata_df = pd.DataFrame({
        "ID": ids,
        "Age": ages,
        "Gender": genders
    })
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    # ---------- Save JSON ----------
    metadata_json = []
    for i in range(len(ids)):
        metadata_json.append({
            "ID": str(ids[i]),
            "Age": None if pd.isna(ages[i]) else float(ages[i]) if isinstance(ages[i], (int, float, np.integer, np.floating)) else ages[i],
            "Gender": str(genders[i])
        })

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, indent=2, ensure_ascii=False)

    # ---------- Save compact bundled dataset ----------
    np.savez_compressed(
        os.path.join(output_dir, "dataset.npz"),
        timeseries=X,
        id=ids,
        age=ages,
        gender=genders
    )

    # ---------- Save logs ----------
    with open(os.path.join(output_dir, "failed_files.json"), "w", encoding="utf-8") as f:
        json.dump(failed_files, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "unmatched_files.json"), "w", encoding="utf-8") as f:
        json.dump(unmatched_files, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print("  - timeseries.npy")
    print("  - metadata.csv")
    print("  - metadata.json")
    print("  - dataset.npz")
    print("  - failed_files.json")
    print("  - unmatched_files.json")

    return X, metadata_df


if __name__ == "__main__":
    mat_dir = r"/path/to/your/mat_files"
    excel_path = r"/path/to/your/subject_info.xlsx"
    output_dir = r"/path/to/save/output"

    X, metadata_df = build_dataset(
        mat_dir=mat_dir,
        excel_path=excel_path,
        output_dir=output_dir,
        mat_pattern="*.mat",
        remove_suffix="_Tx278"
    )