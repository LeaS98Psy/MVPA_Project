"""
MVPA on Haxby dataset (faces vs. houses)

Loads fMRI data, applies mask, trains linear SVM with leave-one-run-out
cross-validation, performs permutation test, and saves confusion matrix.
"""

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------------------------------
print("Loading Haxby dataset...")
haxby = datasets.fetch_haxby()
func_img = haxby.func[0]
mask_img = haxby.mask_vt[0]

# Labels and runs from the session target file
labels_df = pd.read_csv(haxby.session_target[0], sep=" ")
conditions = labels_df['labels'].values
runs = labels_df['chunks'].values

# -----------------------------------------------------------------------------
# 2. Mask data to extract time series per voxel
# -----------------------------------------------------------------------------
masker = NiftiMasker(mask_img=mask_img, standardize='zscore_sample')
X_all = masker.fit_transform(func_img)   # (n_volumes, n_voxels)
n_volumes = X_all.shape[0]

# Trim labels/runs to match actual number of volumes
conditions = conditions[:n_volumes]
runs = runs[:n_volumes]

# Keep only faces and houses
valid_idx = np.where((conditions == 'face') | (conditions == 'house'))[0]
X = X_all[valid_idx, :]
y = conditions[valid_idx]
runs_valid = runs[valid_idx]

print(f"X shape: {X.shape}, y shape: {y.shape}, runs shape: {runs_valid.shape}")

# -----------------------------------------------------------------------------
# 3. SVM with leave-one-run-out cross-validation
# -----------------------------------------------------------------------------
svm = LinearSVC(dual='auto', random_state=42, max_iter=5000)
logo = LeaveOneGroupOut()
scores = cross_val_score(svm, X, y, groups=runs_valid, cv=logo, scoring='accuracy')
print(f"Mean accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

# -----------------------------------------------------------------------------
# 4. Permutation test (shuffle labels within each run)
# -----------------------------------------------------------------------------
n_perm = 500
perm_scores = []
for _ in tqdm(range(n_perm), desc="Running permutation test"):
    y_perm = y.copy()
    for run in np.unique(runs_valid):
        idx_run = (runs_valid == run)
        y_perm[idx_run] = np.random.permutation(y_perm[idx_run])
    score = cross_val_score(svm, X, y_perm, groups=runs_valid, cv=logo, scoring='accuracy').mean()
    perm_scores.append(score)

p_value = np.mean(np.array(perm_scores) >= np.mean(scores))
print(f"Permutation test p-value: {p_value:.4f} (based on {n_perm} permutations)")

# -----------------------------------------------------------------------------
# 5. Confusion matrix (cross-validated predictions)
# -----------------------------------------------------------------------------
y_pred = cross_val_predict(svm, X, y, groups=runs_valid, cv=logo)
cm = confusion_matrix(y, y_pred)
print("Confusion matrix:")
print(cm)

# Plot and save
os.makedirs("outputs", exist_ok=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['face', 'house'])
disp.plot()
plt.title("Confusion Matrix - SVM (leave-one-run-out)")
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved in outputs/confusion_matrix.png")