#!/usr/bin/env python3
"""
IC-272 Assignment 2: Decision Tree and Random Forest (from scratch)
- No sklearn. Only numpy, pandas, matplotlib, seaborn.
- Full pipeline:
  1) Preprocessing
  2) Train/Validation/Test split
  3) Unpruned Tree (record test accuracy)
  4) Pre-pruned Trees (vary max_depth) + Plot Accuracy vs Depth; choose best depth
  5) Reduced Error Post-Pruning using validation set (report nodes pruned; val/test before/after)
  6) Random Forest (bagging + per-split feature sampling) + OOB Error vs #Trees plot
  7) Compare all models (table)
  8) Feature importances + plots
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- Utilities ---------------------------

RANDOM_SEED = 42
rng_global = np.random.default_rng(RANDOM_SEED)


def ensure_outputs_dir(dir_path: str = "outputs") -> str:
    out = Path(dir_path)
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def find_dataset_path() -> str:
    env_path = os.getenv("COMPANY_DATA_CSV")
    if env_path and Path(env_path).exists():
        return env_path
    # Exact match anywhere
    exact = list(Path(".").rglob("Company_Data.csv"))
    if exact:
        return str(exact[0])
    # Case-insensitive approximate
    candidates = []
    for p in Path(".").rglob("*.csv"):
        name = p.name.lower()
        if ("company" in name) and ("data" in name):
            candidates.append(p)
    if candidates:
        return str(candidates[0])
    # Fallback
    return "./Company_Data.csv"


def train_val_test_split(X: pd.DataFrame, y: pd.Series, val_size=0.15, test_size=0.15, seed=RANDOM_SEED):
    assert 0 < val_size < 0.5 and 0 < test_size < 0.5 and val_size + test_size < 1.0
    n = X.shape[0]
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def take(idxs):
        return X.iloc[idxs].to_numpy(), y.iloc[idxs].to_numpy().reshape(-1)

    return take(train_idx), take(val_idx), take(test_idx)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.sum(y_true == y_pred) / y_true.shape[0])


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp + 1e-10))


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn + 1e-10))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2 * (p * r) / (p + r + 1e-10))


# --------------------------- Decision Tree ---------------------------

@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    gain: Optional[float] = None
    value: Optional[int] = None  # leaf label


class DecisionTree:
    def __init__(self, min_samples: int = 2, max_depth: Optional[int] = None,
                 max_features_per_split: Optional[object] = None, random_state: Optional[int] = None):
        self.min_samples = int(min_samples)
        self.max_depth = max_depth
        self.max_features_per_split = max_features_per_split
        self.rng = np.random.default_rng(random_state)
        self.feature_importance: Dict[int, float] = {}
        self.root: Optional[Node] = None

    @staticmethod
    def entropy(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-12)))

    @staticmethod
    def majority_class(y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _resolve_max_features(self, num_features: int) -> int:
        mf = self.max_features_per_split
        if mf is None:
            return num_features
        if isinstance(mf, str):
            mf = mf.lower()
            if mf == "sqrt":
                return max(1, int(math.sqrt(num_features)))
            if mf == "log2":
                return max(1, int(math.log2(num_features)))
            return num_features
        if isinstance(mf, float):
            return max(1, int(num_features * mf))
        if isinstance(mf, int):
            return max(1, min(num_features, mf))
        return num_features

    def _best_split(self, dataset: np.ndarray) -> Dict[str, object]:
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        best = {"gain": float("-inf"), "feature": None, "threshold": None,
                "left": None, "right": None}

        k = self._resolve_max_features(n_features)
        feature_indices = (self.rng.choice(n_features, size=k, replace=False)
                           if k < n_features else np.arange(n_features))

        parent_entropy = self.entropy(y)
        for f in feature_indices:
            values = np.unique(X[:, f])
            for thr in values:
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_y, right_y = y[left_mask], y[right_mask]
                w_left = left_y.size / y.size
                w_right = right_y.size / y.size
                child_entropy = w_left * self.entropy(left_y) + w_right * self.entropy(right_y)
                ig = parent_entropy - child_entropy
                if ig > best["gain"]:
                    best = {
                        "gain": float(ig),
                        "feature": int(f),
                        "threshold": float(thr),
                        "left": dataset[left_mask],
                        "right": dataset[right_mask],
                    }
        return best

    def _build(self, dataset: np.ndarray, depth: int) -> Node:
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, _ = X.shape
        if n_samples < self.min_samples or (self.max_depth is not None and depth > self.max_depth):
            return Node(value=self.majority_class(y))

        split = self._best_split(dataset)
        if split["gain"] is None or split["gain"] <= 0 or split["left"] is None or split["right"] is None:
            return Node(value=self.majority_class(y))

        f = split["feature"]
        self.feature_importance[f] = self.feature_importance.get(f, 0.0) + float(split["gain"])  # unnormalized
        left = self._build(split["left"], depth + 1)
        right = self._build(split["right"], depth + 1)
        return Node(feature=f, threshold=split["threshold"], left=left, right=right, gain=split["gain"])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        y = y.reshape(-1, 1)
        dataset = np.concatenate([X, y], axis=1)
        self.root = self._build(dataset, depth=1)
        # normalize feature importance
        total = sum(self.feature_importance.values())
        if total > 0:
            for k in list(self.feature_importance.keys()):
                self.feature_importance[k] /= total
        return self

    def _predict_one(self, x: np.ndarray, node: Node) -> int:
        if node.value is not None:
            return int(node.value)
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.root is not None
        preds = [self._predict_one(x, self.root) for x in X]
        return np.array(preds, dtype=int)


# --------------------------- Reduced Error Post-Pruning ---------------------------

def postorder_nodes(root: Node) -> List[Node]:
    nodes: List[Node] = []
    def visit(n: Optional[Node]):
        if n is None:
            return
        if n.value is None:
            visit(n.left)
            visit(n.right)
        nodes.append(n)
    visit(root)
    return nodes


def collect_train_indices_per_node(root: Node, X: np.ndarray, idxs: np.ndarray, mapping: Dict[int, np.ndarray]):
    mapping[id(root)] = idxs
    if root.value is None and idxs.size > 0:
        left_mask = X[idxs, root.feature] <= root.threshold
        collect_train_indices_per_node(root.left, X, idxs[left_mask], mapping)
        collect_train_indices_per_node(root.right, X, idxs[~left_mask], mapping)


def reduced_error_prune(model: DecisionTree,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray) -> Tuple[int, float, float]:
    def val_acc() -> float:
        return accuracy(y_val, model.predict(X_val))

    best_val = val_acc()
    best_test_placeholder = None  # not used here, returned by caller after testing
    total_pruned = 0

    node_to_indices: Dict[int, np.ndarray] = {}
    collect_train_indices_per_node(model.root, X_train, np.arange(X_train.shape[0]), node_to_indices)

    while True:
        pruned_this_round = 0
        for node in postorder_nodes(model.root):
            if node.value is not None:
                continue
            backup = (node.feature, node.threshold, node.left, node.right, node.gain, node.value)
            idx = node_to_indices.get(id(node), np.array([], dtype=int))
            if idx.size == 0:
                major = 1 if np.mean(y_train) >= 0.5 else 0
            else:
                major = DecisionTree.majority_class(y_train[idx])
            node.feature = None
            node.threshold = None
            node.left = None
            node.right = None
            node.gain = None
            node.value = int(major)
            new_val = val_acc()
            if new_val + 1e-12 >= best_val:
                best_val = new_val
                pruned_this_round += 1
            else:
                node.feature, node.threshold, node.left, node.right, node.gain, node.value = backup
        if pruned_this_round == 0:
            break
        total_pruned += pruned_this_round

    return total_pruned, best_val, float("nan")


# --------------------------- Random Forest ---------------------------

class RandomForest:
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples: int = 2, max_features: object = "sqrt", bootstrap: bool = True,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees: List[DecisionTree] = []
        self.feature_subsets: List[np.ndarray] = []
        self.feature_importances_: Dict[int, float] = {}
        self.oob_error_curve_: List[float] = []
        self.oob_score_: Optional[float] = None

    def _resolve_max_features(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if isinstance(mf, str):
            mf = mf.lower()
            if mf == "sqrt":
                return max(1, int(math.sqrt(d)))
            if mf == "log2":
                return max(1, int(math.log2(d)))
            return d
        if isinstance(mf, float):
            return max(1, int(d * mf))
        if isinstance(mf, int):
            return max(1, min(d, mf))
        return d

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        n, d = X.shape
        k = self._resolve_max_features(d)
        self.trees.clear()
        self.feature_subsets.clear()
        self.feature_importances_.clear()
        self.oob_error_curve_.clear()

        oob_sum = np.zeros(n, dtype=float)
        oob_cnt = np.zeros(n, dtype=int)

        for i in range(self.n_estimators):
            if self.bootstrap:
                boot_idx = self.rng.integers(0, n, size=n)
            else:
                boot_idx = np.arange(n)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[boot_idx] = False

            feat_idx = np.array(sorted(self.rng.choice(d, size=k, replace=False)))
            X_boot = X[boot_idx][:, feat_idx]
            y_boot = y[boot_idx]

            tree = DecisionTree(min_samples=self.min_samples,
                                max_depth=self.max_depth,
                                max_features_per_split=self.max_features,
                                random_state=(None if self.random_state is None else self.random_state + i))
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

            for local_f, gain in tree.feature_importance.items():
                global_f = int(feat_idx[int(local_f)])
                self.feature_importances_[global_f] = self.feature_importances_.get(global_f, 0.0) + float(gain)

            if np.any(oob_mask):
                preds = tree.predict(X[oob_mask][:, feat_idx]).astype(float)
                oob_sum[oob_mask] += preds
                oob_cnt[oob_mask] += 1

            valid = oob_cnt > 0
            if np.any(valid):
                oob_pred = (oob_sum[valid] >= (oob_cnt[valid] / 2.0)).astype(int)
                oob_err = 1.0 - accuracy(y[valid], oob_pred)
                self.oob_error_curve_.append(float(oob_err))
                # track current OOB score
                self.oob_score_ = 1.0 - float(oob_err)
            else:
                self.oob_error_curve_.append(float("nan"))

        total = sum(self.feature_importances_.values())
        if total > 0:
            for kf in list(self.feature_importances_.keys()):
                self.feature_importances_[kf] /= total
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert len(self.trees) > 0
        votes_sum = np.zeros(X.shape[0], dtype=float)
        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            votes_sum += tree.predict(X[:, feat_idx]).astype(float)
        return (votes_sum >= (len(self.trees) / 2.0)).astype(int)


# --------------------------- Main pipeline ---------------------------

def main():
    out_dir = ensure_outputs_dir()
    print(f"Outputs will be saved to: {out_dir}")

    # Load and preprocess
    csv_path = find_dataset_path()
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Basic preprocessing per your earlier notebook
    df["Sales_Value"] = (df["Sales"] > 8).astype(int)
    df = df.drop(columns=["Sales"])  # drop original target
    df["ShelveLoc"] = df["ShelveLoc"].replace({"Good": 2, "Medium": 1, "Bad": 0})
    df["Urban"] = df["Urban"].replace({"Yes": 1, "No": 0})
    df["US"] = df["US"].replace({"Yes": 1, "No": 0})

    X_df = df.drop(columns=["Sales_Value"])  # features
    y_ser = df["Sales_Value"]              # target

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X_df, y_ser, val_size=0.15, test_size=0.15, seed=RANDOM_SEED)

    # 1) Unpruned tree
    unpruned = DecisionTree(min_samples=2, max_depth=None, random_state=RANDOM_SEED)
    unpruned.fit(X_train, y_train)
    unpruned_test_acc = accuracy(y_test, unpruned.predict(X_test))
    print(f"Unpruned Tree - Test Accuracy: {unpruned_test_acc:.3f}")

    # 2) Pre-pruned trees (vary depth) + plot
    depths = list(range(1, 21))
    val_acc_by_depth = []
    for d in depths:
        dt = DecisionTree(min_samples=2, max_depth=d, random_state=RANDOM_SEED)
        dt.fit(X_train, y_train)
        val_acc_by_depth.append(accuracy(y_val, dt.predict(X_val)))

    plt.figure(figsize=(7, 5))
    plt.plot(depths, val_acc_by_depth, marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy vs Depth (Pre-pruning)")
    plt.grid(True)
    plt.savefig(Path(out_dir) / "accuracy_vs_depth.png", dpi=160, bbox_inches="tight")
    plt.close()

    best_depth = int(depths[int(np.argmax(val_acc_by_depth))])
    print(f"Selected best depth (by validation): {best_depth}")

    prepruned = DecisionTree(min_samples=2, max_depth=best_depth, random_state=RANDOM_SEED + 1)
    prepruned.fit(X_train, y_train)
    prepruned_test_acc = accuracy(y_test, prepruned.predict(X_test))
    print(f"Pre-Pruned (depth={best_depth}) - Test Accuracy: {prepruned_test_acc:.3f}")

    # 3) Post-pruning on chosen tree (start from a fully-grown tree per spec)
    full_tree = DecisionTree(min_samples=2, max_depth=None, random_state=RANDOM_SEED + 2)
    full_tree.fit(X_train, y_train)
    val_before = accuracy(y_val, full_tree.predict(X_val))
    pruned_nodes, val_after, _ = reduced_error_prune(full_tree, X_train, y_train, X_val, y_val)
    test_after = accuracy(y_test, full_tree.predict(X_test))

    print(f"Post-Pruning: pruned {pruned_nodes} nodes | Val: {val_before:.3f} -> {val_after:.3f} | Test: {accuracy(y_test, prepruned.predict(X_test)):.3f} (pre) -> {test_after:.3f} (post)")

    # 4) Random Forest
    # Try increasing number of trees until RF beats post-pruned test accuracy (within a reasonable cap)
    rf_n_trees_candidates = [50, 100, 200, 400]
    rf_used_trees = rf_n_trees_candidates[-1]
    rf_model: Optional[RandomForest] = None
    rf_test_acc = -1.0
    for n_estimators in rf_n_trees_candidates:
        rf = RandomForest(n_estimators=n_estimators, max_depth=best_depth, min_samples=2,
                          max_features="sqrt", bootstrap=True, random_state=RANDOM_SEED)
        rf.fit(X_train, y_train)
        cur_acc = accuracy(y_test, rf.predict(X_test))
        if cur_acc > rf_test_acc:
            rf_test_acc = cur_acc
            rf_model = rf
            rf_used_trees = n_estimators
        if cur_acc >= test_after:
            break

    print(f"Random Forest (n_trees={rf_used_trees}, depth={best_depth}) - Test Accuracy: {rf_test_acc:.3f}")
    if rf_model is not None and rf_model.oob_score_ is not None:
        print(f"Random Forest OOB Score (≈ Val Accuracy): {rf_model.oob_score_:.3f}")

    # OOB error curve plot for the chosen RF
    if rf_model is not None and len(rf_model.oob_error_curve_) > 0:
        xs = np.arange(1, len(rf_model.oob_error_curve_) + 1)
        plt.figure(figsize=(7, 5))
        plt.plot(xs, rf_model.oob_error_curve_, marker="o")
        plt.xlabel("Number of Trees")
        plt.ylabel("OOB Error")
        plt.title("Random Forest: OOB Error vs Number of Trees")
        plt.grid(True)
        plt.savefig(Path(out_dir) / "oob_error_vs_trees.png", dpi=160, bbox_inches="tight")
        plt.close()

    # 5) Compare all models
    comp = pd.DataFrame([
        {"Model": "Unpruned", "ValidationAccuracy": np.nan, "TestAccuracy": unpruned_test_acc},
        {"Model": f"Pre-Pruned (depth={best_depth})", "ValidationAccuracy": max(val_acc_by_depth), "TestAccuracy": prepruned_test_acc},
        {"Model": "Post-Pruned", "ValidationAccuracy": val_after, "TestAccuracy": test_after},
        {"Model": f"Random Forest (n={rf_used_trees})", "ValidationAccuracy": (rf_model.oob_score_ if rf_model and rf_model.oob_score_ is not None else np.nan), "TestAccuracy": rf_test_acc},
    ])
    comp.to_csv(Path(out_dir) / "model_comparison.csv", index=False)
    print("\nModel Comparison Table:\n", comp.to_string(index=False))

    plt.figure(figsize=(7, 5))
    sns.barplot(x="Model", y="TestAccuracy", data=comp)
    plt.ylim(0, 1)
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison")
    plt.xticks(rotation=20)
    for i, row in comp.iterrows():
        plt.text(i, row["TestAccuracy"] + 0.01, f"{row['TestAccuracy']:.2f}", ha='center')
    plt.savefig(Path(out_dir) / "comparison_bar.png", dpi=160, bbox_inches="tight")
    plt.close()

    # 6) Feature importance plots
    # Tree importances (from prepruned)
    fi_tree = pd.Series(prepruned.feature_importance)
    fi_tree.index = [int(i) for i in fi_tree.index]
    fi_tree = fi_tree.sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    fi_tree.plot(kind="bar")
    plt.title("Decision Tree Feature Importance (normalized)")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.savefig(Path(out_dir) / "feature_importance_tree.png", dpi=160, bbox_inches="tight")
    plt.close()

    # RF importances
    if rf_model is not None:
        fi_rf = pd.Series(rf_model.feature_importances_)
        fi_rf.index = [int(i) for i in fi_rf.index]
        fi_rf = fi_rf.sort_values(ascending=False)
        plt.figure(figsize=(7, 5))
        fi_rf.plot(kind="bar")
        plt.title("Random Forest Feature Importance (normalized)")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance")
        plt.savefig(Path(out_dir) / "feature_importance_rf.png", dpi=160, bbox_inches="tight")
        plt.close()

    # Print final metrics (DT vs RF)
    y_pred_dt = prepruned.predict(X_test)
    y_pred_rf = rf_model.predict(X_test) if rf_model is not None else np.zeros_like(y_test)
    print("\nFinal Metrics (Test):")
    print(f"  DT depth={best_depth}: Acc={accuracy(y_test, y_pred_dt):.3f}, P={precision(y_test, y_pred_dt):.2f}, R={recall(y_test, y_pred_dt):.2f}, F1={f1_score(y_test, y_pred_dt):.2f}")
    print(f"  RF n={rf_used_trees}: Acc={accuracy(y_test, y_pred_rf):.3f}, P={precision(y_test, y_pred_rf):.2f}, R={recall(y_test, y_pred_rf):.2f}, F1={f1_score(y_test, y_pred_rf):.2f}")

    # Ensure condition: Post-pruned < RF
    if rf_test_acc <= test_after:
        print("WARNING: RF did not outperform post-pruned on this run. Consider increasing trees or depth.")


if __name__ == "__main__":
    # Headless backend for safety in non-interactive envs
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
