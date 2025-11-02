import numpy as np
import pandas as pd
import pytest

from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.utils.target_encoder import CrossFoldTargetEncoder


def test_cross_fold_target_encoder_binary_basic():
    X = pd.DataFrame({
        "cat": ["a", "b", "a", "c", "b", "d"],
        "num": [1, 2, 3, 4, 5, 6],
    })
    y = pd.Series([0, 1, 1, 0, 1, 0], name="label")

    encoder = CrossFoldTargetEncoder(
        columns=["cat"],
        problem_type=BINARY,
        n_folds=3,
        smoothing=1.0,
        noise=0.0,
        random_state=0,
    )

    transformed = encoder.fit_transform(X, y)

    assert "cat__te" in transformed.columns
    assert encoder.encoded_feature_names_ == ["cat__te"]
    assert transformed["cat__te"].isna().sum() == 0
    # Original column retained by default
    assert "cat" in transformed.columns

    new_data = pd.DataFrame({"cat": ["a", "e"], "num": [0, 0]})
    transformed_new = encoder.transform(new_data)
    assert "cat__te" in transformed_new.columns
    assert pytest.approx(transformed_new.loc[1, "cat__te"], rel=1e-6) == y.mean()


def test_cross_fold_target_encoder_binary_drop_original():
    X = pd.DataFrame({
        "cat": ["x", "y", "x", "z"],
        "num": [1, 2, 3, 4],
    })
    y = pd.Series([0, 1, 1, 0], name="label")

    encoder = CrossFoldTargetEncoder(
        columns=["cat"],
        problem_type=BINARY,
        n_folds=2,
        smoothing=1.0,
        noise=0.0,
        keep_original=False,
        random_state=0,
    )

    transformed = encoder.fit_transform(X, y)

    assert "cat" not in transformed.columns
    assert "cat__te" in transformed.columns

    transformed_new = encoder.transform(pd.DataFrame({"cat": ["x", "q"], "num": [0, 0]}))
    assert "cat" not in transformed_new.columns
    assert "cat__te" in transformed_new.columns


def test_cross_fold_target_encoder_multiclass():
    X = pd.DataFrame({
        "cat": ["m", "n", "m", "o", "n", "p"],
        "num": [1, 2, 3, 4, 5, 6],
    })
    y = pd.Series([0, 1, 2, 0, 1, 2], name="label")

    encoder = CrossFoldTargetEncoder(
        columns=["cat"],
        problem_type=MULTICLASS,
        num_classes=3,
        n_folds=3,
        smoothing=2.0,
        noise=0.0,
        random_state=1,
    )

    transformed = encoder.fit_transform(X, y)
    expected_columns = ["cat__te_class0", "cat__te_class1", "cat__te_class2"]

    for col in expected_columns:
        assert col in transformed.columns
        assert transformed[col].between(0, 1).all()

    new_rows = pd.DataFrame({"cat": ["m", "r"], "num": [0, 0]})
    transformed_new = encoder.transform(new_rows)

    for col in expected_columns:
        assert col in transformed_new.columns
        assert transformed_new[col].between(0, 1).all()

    # Unseen category should revert to global class proportions
    global_means = np.array([
        (y == cls).mean() for cls in range(3)
    ])
    fallback_values = transformed_new.loc[1, expected_columns].to_numpy()
    assert np.allclose(fallback_values, global_means, atol=1e-6)
