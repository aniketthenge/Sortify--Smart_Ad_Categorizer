"""
Multinomial logistic regression in plain NumPy.

Drop-in for sklearn's LogisticRegression(class_weight="balanced") as used here
(fit / predict / predict_proba / classes_ / coef_). It exists because Windows
Smart App Control can block scikit-learn's compiled solver (_sag_fast.pyd) on
fresh installs; NumPy and SciPy sparse matrices are all this needs.

Objective (same as sklearn): C * sum_i w_i * cross_entropy_i + 0.5 * ||W||^2,
minimised with Nesterov-accelerated gradient descent and a backtracking step.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin  # pure Python, no compiled code


class SoftmaxRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, C: float = 1.0, max_iter: int = 400, tol: float = 1e-6):
        self.C, self.max_iter, self.tol = C, max_iter, tol

    def _loss_grad(self, X, Y, w, W, b):
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        loss = self.C * -(w * np.log(np.clip((P * Y).sum(1), 1e-12, None))).sum() + 0.5 * (W * W).sum()
        G = self.C * (P - Y) * w[:, None]
        return loss, X.T @ G + W, G.sum(axis=0)

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_, idx = np.unique(y, return_inverse=True)
        n, k = X.shape[0], len(self.classes_)
        Y = np.zeros((n, k))
        Y[np.arange(n), idx] = 1.0
        counts = np.bincount(idx, minlength=k)
        w = (n / (k * counts))[idx]                      # class_weight="balanced"
        W = np.zeros((X.shape[1], k))
        b = np.zeros(k)
        Wv, bv, step, t_prev = W.copy(), b.copy(), 1.0, 1.0
        prev = np.inf
        for _ in range(self.max_iter):
            loss, gW, gb = self._loss_grad(X, Y, w, Wv, bv)
            while True:  # backtracking line search on the look-ahead point
                W_new, b_new = Wv - step * gW, bv - step * gb
                new_loss = self._loss_grad(X, Y, w, W_new, b_new)[0]
                if new_loss <= loss - 0.5 * step * ((gW * gW).sum() + (gb * gb).sum()) or step < 1e-10:
                    break
                step *= 0.5
            t = (1 + np.sqrt(1 + 4 * t_prev * t_prev)) / 2
            Wv = W_new + ((t_prev - 1) / t) * (W_new - W)
            bv = b_new + ((t_prev - 1) / t) * (b_new - b)
            W, b, t_prev = W_new, b_new, t
            step *= 1.5
            if abs(prev - new_loss) <= self.tol * max(1.0, abs(new_loss)):
                break
            prev = new_loss
        self.coef_, self.intercept_ = W.T, b
        return self

    def predict_proba(self, X):
        Z = X @ self.coef_.T + self.intercept_
        Z = np.asarray(Z)
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
