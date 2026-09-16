"""
Task 6: K-Nearest Neighbors (KNN) Classification
AI & ML Internship - Elevate Labs

Objective: Understand and implement KNN for classification problems.
Tools: Scikit-learn, Pandas, Matplotlib

Dataset: Iris dataset - built into scikit-learn, so it's fully reproducible
with no external download. Target: 3 species of iris flower (setosa,
versicolor, virginica) based on 4 flower measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Choose a classification dataset and normalize features
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load dataset & normalize features")
print("=" * 60)

data = load_iris(as_frame=True)
df = data.frame.copy()
print(f"Shape: {df.shape}")
print(f"Classes: {dict(enumerate(data.target_names))}")
print(f"Class balance:\n{df['target'].value_counts()}")
print(f"Missing values: {df.isnull().sum().sum()}")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Normalize features - crucial for KNN since it's a distance-based algorithm
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features standardized (mean=0, std=1) using StandardScaler fit on train set only.")

# ---------------------------------------------------------------
# 2. Use KNeighborsClassifier from sklearn (baseline, K=5)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Baseline KNN (K=5)")
print("=" * 60)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
print(f"Test accuracy (K=5): {accuracy_score(y_test, y_pred):.4f}")

# ---------------------------------------------------------------
# 3. Experiment with different values of K
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Experiment with K = 1 to 30")
print("=" * 60)

k_values = range(1, 31)
accuracies = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    accuracies.append(acc)

k_df = pd.DataFrame({"K": list(k_values), "test_accuracy": accuracies})
k_df.to_csv("k_tuning.csv", index=False)
print(k_df.to_string(index=False))

max_acc = k_df["test_accuracy"].max()
tied_ks = k_df.loc[k_df["test_accuracy"] == max_acc, "K"].tolist()
# K=1 always ties for "best" on small test sets but is the noisiest, most
# overfit-prone choice (zero smoothing). Prefer a K from the stable plateau
# of tied top scores instead, for a more robust practical choice.
robust_candidates = [k for k in tied_ks if k > 1]
best_k = robust_candidates[len(robust_candidates) // 2] if robust_candidates else tied_ks[0]
print(f"K values tied for top accuracy ({max_acc:.4f}): {tied_ks}")
print(f"Chosen K: {int(best_k)} -- picked from the middle of the stable high-accuracy "
      f"plateau rather than K=1, since K=1 has no smoothing and is highly sensitive to noise.")

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker="o")
plt.axvline(best_k, color="gray", linestyle="--", alpha=0.7, label=f"Best K = {int(best_k)}")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.title("KNN: Accuracy vs K")
plt.legend()
plt.tight_layout()
plt.savefig("k_tuning.png")
plt.close()
print("Saved plot -> k_tuning.png")

# ---------------------------------------------------------------
# 4. Evaluate model using accuracy, confusion matrix
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print(f"STEP 4: Final evaluation with best K = {int(best_k)}")
print("=" * 60)

final_knn = KNeighborsClassifier(n_neighbors=int(best_k))
final_knn.fit(X_train_scaled, y_train)
y_pred_final = final_knn.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred_final)
cm = confusion_matrix(y_test, y_pred_final)
print(f"Accuracy: {acc:.4f}")
print(f"\nConfusion matrix:\n{cm}")
print(f"\nClassification report:\n{classification_report(y_test, y_pred_final, target_names=data.target_names)}")

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax, cmap="Purples", colorbar=False)
ax.set_title(f"Confusion Matrix (K={int(best_k)})")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("Saved plot -> confusion_matrix.png")

# ---------------------------------------------------------------
# 5. Visualize decision boundaries
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Visualize decision boundaries")
print("=" * 60)
print("Using the 2 most informative features (petal length, petal width) for a 2D plot.")

# Use petal length & petal width - the two most discriminative Iris features
feat_x, feat_y = "petal length (cm)", "petal width (cm)"
X2 = df[[feat_x, feat_y]].values
y2 = df["target"].values

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=RANDOM_STATE, stratify=y2
)
scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

knn2 = KNeighborsClassifier(n_neighbors=int(best_k))
knn2.fit(X2_train_scaled, y2_train)
acc2 = accuracy_score(y2_test, knn2.predict(X2_test_scaled))
print(f"2D model (petal length & width only) test accuracy: {acc2:.4f}")

# Build mesh grid over the scaled feature space
h = 0.02
x_min, x_max = X2_train_scaled[:, 0].min() - 1, X2_train_scaled[:, 0].max() + 1
y_min, y_max = X2_train_scaled[:, 1].min() - 1, X2_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(["#FFD8D8", "#D8FFD8", "#D8D8FF"])
cmap_bold = ["#FF3333", "#33CC33", "#3333FF"]

plt.figure(figsize=(9, 7))
plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
for i, species in enumerate(data.target_names):
    idx = y2_train == i
    plt.scatter(
        X2_train_scaled[idx, 0], X2_train_scaled[idx, 1],
        c=cmap_bold[i], label=f"{species} (train)", edgecolor="k", s=40
    )
    idx_t = y2_test == i
    plt.scatter(
        X2_test_scaled[idx_t, 0], X2_test_scaled[idx_t, 1],
        c=cmap_bold[i], marker="^", edgecolor="k", s=70, label=f"{species} (test)"
    )
plt.xlabel(f"{feat_x} (standardized)")
plt.ylabel(f"{feat_y} (standardized)")
plt.title(f"KNN Decision Boundaries (K={int(best_k)}) — Petal Length vs Width")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("decision_boundaries.png")
plt.close()
print("Saved plot -> decision_boundaries.png")

# ---------------------------------------------------------------
# 6. Save metrics summary
# ---------------------------------------------------------------
summary = pd.DataFrame(
    {"Metric": ["Best K", "Test Accuracy (4 features)", "Test Accuracy (2 features, for viz)"],
     "Value": [int(best_k), acc, acc2]}
)
summary.to_csv("metrics_summary.csv", index=False)
print("\nSaved metrics -> metrics_summary.csv")
print("\nDone.")
