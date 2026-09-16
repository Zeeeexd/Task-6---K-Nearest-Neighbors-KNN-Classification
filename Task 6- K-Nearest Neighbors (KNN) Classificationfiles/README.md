# Task 6: K-Nearest Neighbors (KNN) Classification

**AI & ML Internship — Elevate Labs**
Objective: Understand and implement KNN for classification problems.

## Dataset
**Iris dataset** — loaded via `sklearn.datasets.load_iris()` (150 samples, 4
numeric features: sepal length/width, petal length/width). Perfectly balanced,
3-class target: `setosa`, `versicolor`, `virginica` (50 samples each). No
missing values.

## What was done
1. **Chose the dataset and normalized features** — 80/20 stratified split,
   then `StandardScaler` fit on the training set only (essential for KNN,
   since it's a distance-based algorithm).
2. **Fit a baseline `KNeighborsClassifier`** (K=5).
3. **Experimented with K = 1 to 30**, tracking test accuracy for each.
4. **Evaluated** the chosen model with accuracy and a confusion matrix.
5. **Visualized decision boundaries** using the two most discriminative
   features (petal length & width) in 2D.

## Results

### Choosing K
Accuracy vs. K (`k_tuning.png` / `k_tuning.csv`) shows a **stable plateau of
96.7% accuracy from K=9 to K=21**, with K=1 also tying at 96.7% purely by
chance on this small 30-sample test set. K=1 was **not** picked despite
tying for best, since a 1-nearest-neighbor model has zero smoothing and is
the most sensitive to noise/outliers of any K — it just happened to get lucky
on this particular split. Instead, **K=15** was chosen from the middle of the
stable plateau, which is a far more robust choice in practice. Accuracy drops
off sharply past K≈23 as the neighborhood becomes too large and starts
crossing into other classes' regions (**underfitting**).

### Final model (K=15)
| Metric | Value |
|---|---|
| Test Accuracy | 0.967 |

**Confusion matrix:**
|  | Pred: setosa | Pred: versicolor | Pred: virginica |
|---|---|---|---|
| **Actual: setosa** | 10 | 0 | 0 |
| **Actual: versicolor** | 0 | 9 | 1 |
| **Actual: virginica** | 0 | 0 | 10 |

Only one misclassification (a versicolor predicted as virginica) — expected,
since those two species overlap slightly in petal measurements, while setosa
is linearly separable from both (visible clearly in `decision_boundaries.png`).

### Decision boundaries
`decision_boundaries.png` plots the KNN decision regions over standardized
petal length vs. petal width, with train points as circles and test points as
triangles. Setosa (red) is cleanly separated; versicolor (green) and
virginica (blue) have a soft, slightly wavy boundary between them, reflecting
their natural overlap in this feature space — this 2-feature model alone
still reaches 93.3% test accuracy.

## Files in this repo
- `knn_classification.py` — full script (load → normalize → baseline → K sweep → evaluate → visualize)
- `k_tuning.png` / `.csv` — accuracy vs K
- `confusion_matrix.png` — confusion matrix at the chosen K
- `decision_boundaries.png` — 2D decision boundary visualization
- `metrics_summary.csv` — final summary metrics

---

## Interview Questions

**1. How does the KNN algorithm work?**
KNN is an **instance-based (lazy) learner** — it doesn't build an explicit
model during training; it just stores the training data. To classify a new
point, it computes the distance (usually Euclidean) from that point to every
training point, finds the **K closest** ones, and assigns the **majority
class** among those K neighbors (for regression, it would average their
values instead).

**2. How do you choose the right K?**
Try a range of K values and evaluate on a validation set or via
cross-validation, then pick the K that gives the best generalization
accuracy — as done here by sweeping K from 1 to 30 and plotting accuracy.
General heuristics: very small K (like 1) is highly sensitive to noise and
overfits; very large K oversmooths and can underfit by pulling in points from
other classes. Using an **odd K** avoids tied votes in binary classification.
A common starting point is `K ≈ √n` (n = number of training samples), then
refine around that with cross-validation.

**3. Why is normalization important in KNN?**
KNN relies entirely on **distance calculations**, and features with larger
numeric ranges dominate the distance metric even if they're not actually more
important. For example, without scaling, a feature ranging 0–10,000 would
completely overwhelm one ranging 0–1 in the Euclidean distance formula.
Standardizing (or min-max scaling) puts all features on a comparable scale so
each contributes fairly to the distance — this project fits `StandardScaler`
on the training set only, then applies it to the test set, to avoid data
leakage.

**4. What is the time complexity of KNN?**
Training is essentially **O(1)** — it just stores the data (that's what
"lazy learning" means). Prediction is expensive: for a naive
implementation, computing distances to all `n` training points with `d`
features costs **O(n·d)** per query, then finding the K smallest distances
adds roughly O(n log K). This makes plain KNN slow at prediction time on
large datasets, though spatial indexing structures like KD-trees or Ball
Trees (which scikit-learn uses automatically for lower-dimensional data) can
reduce average-case query time to roughly O(log n).

**5. What are pros and cons of KNN?**
**Pros:** simple and intuitive; no training phase (instant to "fit"); naturally
handles multi-class problems; makes no assumptions about the underlying data
distribution (non-parametric); decision boundaries can be arbitrarily complex/non-linear.
**Cons:** slow prediction on large datasets (must scan all training points);
requires feature scaling; sensitive to irrelevant/noisy features and the
curse of dimensionality (distances become less meaningful in high dimensions);
sensitive to class imbalance (majority class can dominate neighbor votes);
needs to store the entire training set in memory.

**6. Is KNN sensitive to noise?**
Yes — especially at small K. With K=1, a single mislabeled or outlier point
can flip the prediction for any nearby query, since there's no averaging to
smooth it out. Larger K reduces sensitivity to individual noisy points by
"voting" over a bigger neighborhood, but too large a K can then blur genuine
class boundaries. This is exactly the bias-variance trade-off visible in this
project's `k_tuning.png`: very small K = high variance (noise-sensitive),
very large K = high bias (oversmoothed).

**7. How does KNN handle multi-class problems?**
Natively and without any modification — it simply takes a **majority vote**
among the K nearest neighbors' labels, whatever the number of classes. This
is unlike algorithms such as plain logistic regression or SVM, which need
extensions (One-vs-Rest, One-vs-One, or softmax) to go beyond binary
classification. In this project, KNN handled all 3 Iris species directly.

**8. What's the role of distance metrics in KNN?**
The distance metric defines what "nearest" means, so it directly determines
which points are considered neighbors and therefore the predictions.
**Euclidean distance** (straight-line distance, scikit-learn's default) is
the most common; **Manhattan distance** (sum of absolute differences) can be
more robust in high dimensions or grid-like data; **Minkowski distance**
generalizes both (Euclidean = Minkowski with p=2, Manhattan = p=1); and other
metrics like **cosine distance** are used for text/high-dimensional sparse
data where direction matters more than magnitude. Choosing the right metric
for the data's structure can meaningfully change model performance.
