"""Supervised classifier: TF-IDF (word 1-2 + char_wb 3-5) + categorical tokens -> LogisticRegression,
temperature-calibrated on an internal holdout. Used for label-trained models (A, B seeds) and for
distillation on LLM-labelled units (sample weights + synthetic leaf-definition documents).
Deterministic; tolerant of missing columns, single-sample classes and unseen leaves."""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .features import GROUPS, cat_series, text_series

log = logging.getLogger("spendcube.supervised")


def _softmax_T(logp: np.ndarray, T: float) -> np.ndarray:
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SupervisedModel:
    def __init__(self, groups: Sequence[str] = GROUPS, C: float = 15.0, calibrate: bool = True,
                 doc_weight: float = 1.0, max_word: int = 80000, max_char: int = 80000,
                 cat_scale: float = 1.0, min_cal: int = 300, seed: int = 0, max_iter: int = 300):
        self.groups = tuple(groups)
        self.C, self.calibrate, self.doc_weight = C, calibrate, doc_weight
        self.max_word, self.max_char, self.cat_scale = max_word, max_char, cat_scale
        self.min_cal, self.seed, self.max_iter = min_cal, seed, max_iter
        self.T = 1.0
        self.classes_: np.ndarray = np.array([])
        self._fitted = False
        self._const = False

    # ---- feature plumbing --------------------------------------------------
    def _new_vectorizers(self):
        self.vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=self.max_word,
                                  token_pattern=r"[A-Z0-9][A-Z0-9\.\-/]*", lowercase=False, dtype=np.float32)
        self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2,
                                  max_features=self.max_char, lowercase=False, dtype=np.float32)
        self.vk = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, binary=True,
                                  dtype=np.float32)

    def _matrix(self, text: pd.Series, cats: pd.Series, fit: bool):
        blocks = []
        if "text" in self.groups:
            blocks.append(self.vw.fit_transform(text) if fit else self.vw.transform(text))
            blocks.append(self.vc.fit_transform(text) if fit else self.vc.transform(text))
        if any(g != "text" for g in self.groups):
            k = self.vk.fit_transform(cats) if fit else self.vk.transform(cats)
            blocks.append(k * self.cat_scale)
        if not blocks:
            raise ValueError("no feature groups selected")
        return sp.hstack(blocks, format="csr")

    # ---- fit -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame, leaf_ids, weights=None, extra_docs: Optional[list[tuple[str, str]]] = None):
        text = text_series(df).reset_index(drop=True)
        cats = cat_series(df, self.groups).reset_index(drop=True)
        y = pd.Series(np.asarray(leaf_ids, dtype=object)).astype(str)
        w = pd.Series(np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)).clip(lower=1e-6)
        if extra_docs:
            ed = pd.DataFrame(list(extra_docs), columns=["t", "y"])
            text = pd.concat([text, ed["t"].astype(str).str.upper()], ignore_index=True)
            cats = pd.concat([cats, pd.Series([""] * len(ed))], ignore_index=True)
            y = pd.concat([y, ed["y"].astype(str)], ignore_index=True)
            w = pd.concat([w, pd.Series(np.full(len(ed), float(self.doc_weight)))], ignore_index=True)
        # collapse identical (text, cats, label) rows: equivalent for LR with summed weights, much faster
        key = text + "\x1f" + cats + "\x1f" + y
        codes, _ = pd.factorize(key)
        first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
        wsum = w.groupby(codes).sum().to_numpy()
        T_ = text.iloc[first].reset_index(drop=True)
        C_ = cats.iloc[first].reset_index(drop=True)
        Y_ = y.iloc[first].to_numpy()
        n = len(Y_)
        self.T = 1.0
        classes = np.unique(Y_)
        self._const = False
        if len(classes) < 2:
            self.classes_, self._const, self._fitted = classes, True, True
            return self
        rng = np.random.RandomState(self.seed)
        if self.calibrate and n >= self.min_cal:
            perm = rng.permutation(n)
            ho = perm[: max(50, int(0.15 * n))]
            tr = np.setdiff1d(np.arange(n), ho)
            if len(np.unique(Y_[tr])) >= 2:
                self._new_vectorizers()
                Xtr = self._matrix(T_.iloc[tr], C_.iloc[tr], fit=True)
                m = LogisticRegression(C=self.C, max_iter=self.max_iter)
                m.fit(Xtr, Y_[tr], sample_weight=wsum[tr])
                Xho = self._matrix(T_.iloc[ho], C_.iloc[ho], fit=False)
                P = np.clip(m.predict_proba(Xho), 1e-9, 1)
                known = np.isin(Y_[ho], m.classes_)      # holdout labels the sub-model never saw: skip
                if known.sum() >= 30:
                    idx = np.searchsorted(m.classes_, Y_[ho][known])
                    lp, ww = np.log(P[known]), wsum[ho][known]
                    best, bT = np.inf, 1.0
                    for T in np.arange(0.5, 6.01, 0.1):
                        pr = np.clip(_softmax_T(lp, T)[np.arange(len(idx)), idx], 1e-9, 1)
                        nll = -(ww * np.log(pr)).sum() / ww.sum()
                        if nll < best:
                            best, bT = nll, T
                    self.T = float(bT)
        self._new_vectorizers()
        X = self._matrix(T_, C_, fit=True)
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self.model.fit(X, Y_, sample_weight=wsum)
        self.classes_ = self.model.classes_
        self._fitted = True
        log.info("supervised fit: %d unique rows, %d classes, T=%.2f, features=%d", n, len(self.classes_), self.T, X.shape[1])
        return self

    # ---- predict ------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self._const:
            return np.ones((len(df), 1))
        X = self._matrix(text_series(df).reset_index(drop=True), cat_series(df, self.groups).reset_index(drop=True), fit=False)
        P = np.clip(self.model.predict_proba(X), 1e-9, 1)
        return _softmax_T(np.log(P), self.T) if self.T != 1.0 else P

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("model not fitted")
        if len(df) == 0:
            return pd.DataFrame({"leaf_id": [], "p": [], "top3": []}, index=df.index)
        P = self.predict_proba(df)
        k = min(3, P.shape[1])
        top = np.argsort(-P, axis=1)[:, :k]
        cls = self.classes_
        return pd.DataFrame({
            "leaf_id": cls[top[:, 0]],
            "p": P[np.arange(len(P)), top[:, 0]].astype(float),
            "top3": [[(cls[j], float(P[i, j])) for j in top[i]] for i in range(len(P))],
        }, index=df.index)
