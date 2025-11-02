from __future__ import annotations

import pandas as pd

from autogluon.core.models.dummy.dummy_model import DummyModel


def test_target_encoding_resolves_auto_seed_to_default():
    X = pd.DataFrame(
        {
            "cat": ["a", "b", "c", "a", "b", "c"],
            "num": [0, 1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    model = DummyModel(hyperparameters={"ag.target_encoding": True})
    model.fit(X=X, y=y)

    encoder = model._target_encoder
    assert encoder is not None
    assert encoder.encoded_feature_names_
    assert encoder.random_state == model.random_seed
    assert encoder.random_state == model.default_random_seed
