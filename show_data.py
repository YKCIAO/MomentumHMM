import numpy as np

file_path = r"D:\CodeHome\python\MomentumHMM\outputs\representation\act_0.8000__trend_0.8000__a_3.0000__b_1.0000\representation_outputs.npz"

data = np.load(file_path, allow_pickle=True)

print(data.files)

for key in [
    "x_std",
    "dx_std",
    "feature_tensor",
    "X",
    "lengths",
    "n_subjects",
    "n_rois",
    "n_timepoints",
]:
    if key in data:
        value = data[key]

        if hasattr(value, "shape"):
            print(key, value.shape)
        else:
            print(key, value)