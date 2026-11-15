"""Fold-fitted preprocessing and fixed candidate model definitions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_model_pipeline(
    frame: pd.DataFrame,
    features: list[str],
    model_name: str,
    parameters: dict[str, Any],
    seed: int = 42,
) -> Pipeline:
    """Build preprocessing and estimator together so each fold fits separately."""
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise ValueError(f"Pipeline features are missing from the frame: {missing}")

    categorical = [
        feature
        for feature in features
        if (
            pd.api.types.is_object_dtype(frame[feature].dtype)
            or pd.api.types.is_string_dtype(frame[feature].dtype)
            or isinstance(frame[feature].dtype, pd.CategoricalDtype)
        )
    ]
    numeric = [feature for feature in features if feature not in categorical]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))
    if numeric:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "logistic":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("numeric", numeric_pipeline, numeric))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    if model_name not in {"logistic", "random_forest", "hist_gradient_boosting"}:
        raise ValueError(f"Unsupported candidate model: {model_name}")
    model_parameters: dict[str, Any]
    if model_name == "logistic":
        model_parameters = {"C": 0.1, "max_iter": 500, "random_state": seed}
        model_parameters.update(parameters)
        estimator = LogisticRegression(**model_parameters)
    elif model_name == "random_forest":
        model_parameters = {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_leaf": 50,
            "n_jobs": -1,
            "random_state": seed,
        }
        model_parameters.update(parameters)
        estimator = RandomForestClassifier(**model_parameters)
    else:
        model_parameters = {
            "max_iter": 150,
            "learning_rate": 0.06,
            "max_leaf_nodes": 12,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            "random_state": seed,
        }
        model_parameters.update(parameters)
        estimator = HistGradientBoostingClassifier(**model_parameters)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def candidate_specs(feature_sets: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Return the fixed comparison and small compact-set tuning grid."""
    specs: list[dict[str, Any]] = []
    defaults = {
        "logistic": {"C": 0.1},
        "random_forest": {"max_depth": 8},
        "hist_gradient_boosting": {"max_leaf_nodes": 12},
    }
    for feature_set in feature_sets:
        for model_name, parameters in defaults.items():
            specs.append(
                {
                    "name": f"{model_name}__{feature_set}",
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "parameters": parameters,
                }
            )

    tuning_grid = {
        "logistic": [{"C": 0.03}, {"C": 0.3}],
        "random_forest": [{"max_depth": 5}, {"max_depth": 12}],
        "hist_gradient_boosting": [
            {"max_leaf_nodes": 8},
            {"max_leaf_nodes": 16},
        ],
    }
    for model_name, parameter_sets in tuning_grid.items():
        for parameters in parameter_sets:
            value = next(iter(parameters.values()))
            parameter_name = next(iter(parameters))
            specs.append(
                {
                    "name": f"{model_name}__compact_22__{parameter_name}_{value}",
                    "model_name": model_name,
                    "feature_set": "compact_22",
                    "parameters": parameters,
                }
            )
    return specs
