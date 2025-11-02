from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
from sklearn.model_selection import KFold, StratifiedKFold

from autogluon.core.constants import BINARY, MULTICLASS, QUANTILE, REGRESSION

_MISSING_TOKEN = "__AGTE_MISSING__"


@dataclass
class _SingleTargetEncoderState:
    """Holds the statistics required to transform a single categorical feature."""

    encoded_column_names: List[str]
    count_per_category: pd.Series
    sum_per_category: Dict[int | str, pd.Series]
    global_mean: Dict[int | str, float]


class CrossFoldTargetEncoder:
    """Cross-fold target encoding with additive smoothing.

    The encoder computes out-of-fold target-encoded features for training data to avoid target leakage,
    and stores the smoothed category statistics to transform new data consistently during inference.
    """

    def __init__(
        self,
        *,
        columns: Optional[Iterable[str]] = None,
        problem_type: str = REGRESSION,
        num_classes: Optional[int] = None,
        n_folds: int = 5,
        smoothing: float = 10.0,
        min_samples_leaf: int = 1,
        noise: float = 0.0,
        keep_original: bool = True,
        random_state: Optional[int] = None,
        dtype=np.float32,
    ):
        if n_folds < 1:
            raise ValueError(f"n_folds must be >=1, but was {n_folds}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >=1, but was {min_samples_leaf}")
        if smoothing < 0:
            raise ValueError(f"smoothing must be >=0, but was {smoothing}")

        self.columns_requested = list(columns) if columns is not None else None
        self.problem_type = problem_type
        self.num_classes = num_classes
        self.n_folds = n_folds
        self.smoothing = float(smoothing)
        self.min_samples_leaf = int(min_samples_leaf)
        self.noise = float(noise)
        self.keep_original = keep_original
        self.random_state = random_state
        self.dtype = dtype

        self.columns_: List[str] = []
        self.encoded_feature_names_: List[str] = []
        self._states: Dict[str, _SingleTargetEncoderState] = {}
        self._classes_: Optional[List[int]] = None
        self._fitted = False

    # -------------------- public API --------------------
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        X = X.copy()
        y = self._ensure_series(y, index=X.index, name="target")
        sample_weight = self._ensure_optional_series(sample_weight, index=X.index, name="sample_weight")

        if self.problem_type not in {BINARY, MULTICLASS, REGRESSION, QUANTILE}:
            # Softclass and other advanced modes not currently supported.
            self._fitted = True
            self.columns_ = []
            self.encoded_feature_names_ = []
            self._states = {}
            return X

        self._classes_ = self._infer_classes(y=y)
        columns_to_encode = self._select_columns(X)
        self.columns_ = columns_to_encode
        if not columns_to_encode:
            self._fitted = True
            self.encoded_feature_names_ = []
            self._states = {}
            return X

        rng = np.random.default_rng(self.random_state) if self.noise > 0 else None

        encoded_frames: List[pd.DataFrame] = []
        for column in columns_to_encode:
            feature_series = X[column]
            encoded_values, state = self._fit_transform_single_column(
                feature=feature_series,
                y=y,
                sample_weight=sample_weight,
                rng=rng,
            )
            encoded_df = pd.DataFrame(encoded_values, index=X.index, columns=state.encoded_column_names)
            encoded_frames.append(encoded_df.astype(self.dtype))
            self._states[column] = state

        if encoded_frames:
            encoded_concat = pd.concat(encoded_frames, axis=1)
            self.encoded_feature_names_ = list(encoded_concat.columns)
            if not self.keep_original:
                X = X.drop(columns=columns_to_encode)
            for col in encoded_concat.columns:
                X[col] = encoded_concat[col]
        else:
            self.encoded_feature_names_ = []

        self._fitted = True
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or not self.encoded_feature_names_:
            return X

        X = X.copy()
        encoded_frames: List[pd.DataFrame] = []
        for column in self.columns_:
            state = self._states.get(column)
            if state is None:
                continue
            if column in X.columns:
                feature_series = X[column]
            else:
                feature_series = pd.Series([np.nan] * len(X), index=X.index, name=column)
            encoded_values = self._transform_single_column(feature_series, state)
            encoded_df = pd.DataFrame(encoded_values, index=X.index, columns=state.encoded_column_names)
            encoded_frames.append(encoded_df.astype(self.dtype))

        if encoded_frames:
            encoded_concat = pd.concat(encoded_frames, axis=1)
            if not self.keep_original:
                X = X.drop(columns=[col for col in self.columns_ if col in X.columns])
            for col in encoded_concat.columns:
                X[col] = encoded_concat[col]

        # Guarantee encoded columns appear even if transform skipped due to missing original features
        for col in self.encoded_feature_names_:
            if col not in X.columns:
                fallback_val = self._global_value_for_column(col)
                X[col] = np.full(len(X), fallback_val, dtype=self.dtype)

        return X

    def needs_augmentation(self, X: pd.DataFrame) -> bool:
        if not self._fitted or not self.encoded_feature_names_:
            return False
        return any(col not in X.columns for col in self.encoded_feature_names_)

    # -------------------- helpers --------------------
    @staticmethod
    def _ensure_series(series, *, index: pd.Index, name: str) -> pd.Series:
        if series is None:
            raise ValueError(f"{name} must not be None for target encoding")
        if not isinstance(series, pd.Series):
            series = pd.Series(series, name=name)
        if not series.index.equals(index):
            series = series.reindex(index)
        return series

    @staticmethod
    def _ensure_optional_series(series, *, index: pd.Index, name: str) -> Optional[pd.Series]:
        if series is None:
            return None
        if not isinstance(series, pd.Series):
            series = pd.Series(series, name=name)
        if not series.index.equals(index):
            series = series.reindex(index)
        return series

    def _select_columns(self, X: pd.DataFrame) -> List[str]:
        if self.columns_requested is not None:
            candidates = [col for col in self.columns_requested if col in X.columns]
        else:
            candidates = [
                col
                for col in X.columns
                if pd_types.is_categorical_dtype(X[col]) or pd_types.is_object_dtype(X[col])
            ]
        # Filter out columns with no variation (will not benefit from encoding)
        filtered = []
        for col in candidates:
            if X[col].nunique(dropna=False) > 1:
                filtered.append(col)
        return filtered

    def _infer_classes(self, y: pd.Series) -> Optional[List[int]]:
        if self.problem_type == MULTICLASS:
            unique = sorted(set(y.dropna().unique()))
            if self.num_classes is not None and len(unique) != self.num_classes:
                # Trust num_classes ordering if provided
                return list(range(self.num_classes))
            return [int(cls) for cls in unique]
        if self.problem_type == BINARY:
            return [1]
        return None

    def _fit_transform_single_column(
        self,
        *,
        feature: pd.Series,
        y: pd.Series,
        sample_weight: Optional[pd.Series],
        rng: Optional[np.random.Generator],
    ) -> tuple[np.ndarray, _SingleTargetEncoderState]:
        feature_prepared = self._prepare_feature(feature)
        cv_splits = self._get_cv_splits(y=y)

        output_dim = len(self._classes_) if self._classes_ is not None else 1
        oof_encoded = np.zeros((len(feature_prepared), output_dim), dtype=np.float64)

        if cv_splits is None:
            mapping = self._compute_mapping(feature=feature_prepared, y=y, sample_weight=sample_weight)
            oof_encoded = self._apply_mapping(
                feature=feature_prepared,
                mapping=mapping,
                add_noise=True,
                rng=rng,
            )
        else:
            for train_idx, val_idx in cv_splits:
                mapping = self._compute_mapping(
                    feature=feature_prepared.iloc[train_idx],
                    y=y.iloc[train_idx],
                    sample_weight=None if sample_weight is None else sample_weight.iloc[train_idx],
                )
                fold_encoded = self._apply_mapping(
                    feature=feature_prepared.iloc[val_idx],
                    mapping=mapping,
                    add_noise=True,
                    rng=rng,
                )
                oof_encoded[val_idx, :] = fold_encoded

        # Fit on the entire data for inference
        full_mapping = self._compute_mapping(feature=feature_prepared, y=y, sample_weight=sample_weight)
        state = _SingleTargetEncoderState(
            encoded_column_names=self._encoded_column_names(feature.name),
            count_per_category=full_mapping.count_per_category,
            sum_per_category=full_mapping.sum_per_category,
            global_mean=full_mapping.global_mean,
        )

        return oof_encoded, state

    def _transform_single_column(self, feature: pd.Series, state: _SingleTargetEncoderState) -> np.ndarray:
        feature_prepared = self._prepare_feature(feature)
        return self._apply_mapping(feature=feature_prepared, mapping=state, add_noise=False, rng=None)

    def _prepare_feature(self, feature: pd.Series) -> pd.Series:
        if pd_types.is_categorical_dtype(feature):
            feature = feature.astype("object")
        feature = feature.fillna(_MISSING_TOKEN).astype("object")
        # Convert to string to avoid issues with unhashable objects
        feature = feature.astype(str)
        feature.name = feature.name
        return feature

    def _get_cv_splits(self, y: pd.Series):
        n_samples = len(y)
        n_splits = min(self.n_folds, n_samples)
        if n_splits < 2:
            return None
        if self.problem_type in {BINARY, MULTICLASS}:
            unique_class_count = y.nunique()
            if unique_class_count < 2:
                return None
            min_class_count = y.value_counts().min()
            n_splits = min(n_splits, unique_class_count, min_class_count)
            if n_splits < 2:
                return None
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            return list(splitter.split(np.zeros(n_samples), y))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return list(splitter.split(np.zeros(n_samples)))

    def _compute_mapping(
        self,
        *,
        feature: pd.Series,
        y: pd.Series,
        sample_weight: Optional[pd.Series],
    ) -> _SingleTargetEncoderState:
        counts = self._compute_counts(feature=feature, sample_weight=sample_weight)
        sum_per_category: Dict[int | str, pd.Series] = {}
        global_mean: Dict[int | str, float] = {}

        if self._classes_ is not None:
            for class_idx in self._classes_:
                target = (y == class_idx).astype(float)
                sum_series, global_avg = self._compute_weighted_sum(feature=feature, target=target, sample_weight=sample_weight)
                sum_per_category[class_idx] = sum_series
                global_mean[class_idx] = global_avg
        else:
            target = y.astype(float)
            sum_series, global_avg = self._compute_weighted_sum(feature=feature, target=target, sample_weight=sample_weight)
            sum_per_category["__target__"] = sum_series
            global_mean["__target__"] = global_avg

        return _SingleTargetEncoderState(
            encoded_column_names=self._encoded_column_names(feature.name),
            count_per_category=counts,
            sum_per_category=sum_per_category,
            global_mean=global_mean,
        )

    @staticmethod
    def _compute_counts(*, feature: pd.Series, sample_weight: Optional[pd.Series]) -> pd.Series:
        if sample_weight is not None:
            df = pd.DataFrame({"feature": feature, "weight": sample_weight})
            counts = df.groupby("feature", dropna=False)["weight"].sum()
        else:
            counts = feature.value_counts(dropna=False)
        return counts.astype(float)

    @staticmethod
    def _compute_weighted_sum(*, feature: pd.Series, target: pd.Series, sample_weight: Optional[pd.Series]) -> tuple[pd.Series, float]:
        if sample_weight is not None:
            df = pd.DataFrame({"feature": feature, "target": target, "weight": sample_weight})
            df["target_weighted"] = df["target"] * df["weight"]
            sum_series = df.groupby("feature", dropna=False)["target_weighted"].sum()
            total_weight = df["weight"].sum()
        else:
            df = pd.DataFrame({"feature": feature, "target": target})
            sum_series = df.groupby("feature", dropna=False)["target"].sum()
            total_weight = len(target)
        global_mean = sum_series.sum() / total_weight if total_weight else 0.0
        return sum_series.astype(float), float(global_mean)

    def _apply_mapping(
        self,
        *,
        feature: pd.Series,
        mapping: _SingleTargetEncoderState,
        add_noise: bool,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        counts = mapping.count_per_category.reindex(feature).fillna(0.0).to_numpy(dtype=np.float64)
        smoothing = self.smoothing
        denom = counts + smoothing
        if self._classes_ is not None:
            encoded = np.zeros((len(feature), len(self._classes_)), dtype=np.float64)
            fallback = np.array([mapping.global_mean[class_idx] for class_idx in self._classes_], dtype=np.float64)
            for idx, class_idx in enumerate(self._classes_):
                class_sum = mapping.sum_per_category[class_idx].reindex(feature).fillna(0.0).to_numpy(dtype=np.float64)
                numer = class_sum + smoothing * mapping.global_mean[class_idx]
                encoded[:, idx] = numer / np.where(denom == 0.0, 1.0, denom)
            encoded = self._enforce_min_samples(counts, encoded, fallback)
        else:
            class_sum = mapping.sum_per_category["__target__"].reindex(feature).fillna(0.0).to_numpy(dtype=np.float64)
            fallback = mapping.global_mean["__target__"]
            encoded = (class_sum + smoothing * fallback) / np.where(denom == 0.0, 1.0, denom)
            encoded = self._enforce_min_samples(counts, encoded, fallback)
            encoded = encoded[:, None]

        if add_noise and self.noise > 0 and rng is not None:
            encoded += rng.normal(loc=0.0, scale=self.noise, size=encoded.shape)

        return encoded.astype(np.float64)

    def _enforce_min_samples(self, counts: np.ndarray, encoded: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
        mask = counts < self.min_samples_leaf
        if not np.any(mask):
            return encoded
        encoded = encoded.copy()
        if np.isscalar(fallback):
            encoded[mask] = fallback
        else:
            encoded[mask, :] = fallback
        return encoded

    def _encoded_column_names(self, column: Optional[str]) -> List[str]:
        base = column if column is not None else "feature"
        if self._classes_ is not None:
            if self.problem_type == BINARY and len(self._classes_) == 1:
                return [f"{base}__te"]
            return [f"{base}__te_class{cls}" for cls in self._classes_]
        return [f"{base}__te"]

    def _global_value_for_column(self, encoded_column: str) -> float:
        for column, state in self._states.items():
            if encoded_column in state.encoded_column_names:
                idx = state.encoded_column_names.index(encoded_column)
                if self._classes_ is not None:
                    class_idx = self._classes_[idx]
                    return float(state.global_mean[class_idx])
                return float(state.global_mean["__target__"])
        return 0.0
