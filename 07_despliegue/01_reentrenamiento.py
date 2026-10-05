"""Train the reproducible PD, EAD and LGD credit-risk pipelines.

Run from the project root:
    .venv\\Scripts\\python.exe 07_despliegue\\01_reentrenamiento.py
"""

import sys
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "04_scripts"))

import sector_empleo  # noqa: E402

# Embed the module's code in the artefact so the API does not need 04_scripts to load it.
cloudpickle.register_pickle_by_value(sector_empleo)
TRAIN_PATH = PROJECT_ROOT / "02_datos" / "03_Entrenamiento" / "train.pkl"
ARTEFACT_PATH = PROJECT_ROOT / "07_despliegue" / "artefacto_pipeline.pkl"
# The API (and Render) load this copy; it must always match ARTEFACT_PATH.
API_ARTEFACT_PATH = PROJECT_ROOT / "07_despliegue" / "api" / "artefacto_pipeline.pkl"
RANDOM_STATE = 42

CATEGORICAL_INPUT_COLUMNS = ["ingresos_verificados", "vivienda", "finalidad", "num_cuotas"]
# sector_empleo is derived from the raw job title (`empleo`) inside clean_raw_input.
OHE_COLUMNS = CATEGORICAL_INPUT_COLUMNS + ["sector_empleo"]
ORDINAL_COLUMNS = ["antigüedad_empleo", "rating"]
# Skewed numerics (|skew| > 0.75 in training) get Yeo-Johnson; the rest only standard scaling.
SKEWED_NUMERIC_COLUMNS = ["ingresos", "dti", "num_lineas_credito", "principal", "imp_cuota"]
SYMMETRIC_NUMERIC_COLUMNS = ["porc_uso_revolving", "tipo_interes"]
NUMERIC_COLUMNS = SKEWED_NUMERIC_COLUMNS + SYMMETRIC_NUMERIC_COLUMNS
BINARIZED_COLUMNS = ["num_derogatorios"]
RAW_INPUT_COLUMNS = CATEGORICAL_INPUT_COLUMNS + ORDINAL_COLUMNS + NUMERIC_COLUMNS + BINARIZED_COLUMNS + ["empleo"]
DEFAULT_STATUSES = {
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
    "Default",
}
# Loans without a final outcome cannot be labelled as good or bad payers.
UNRESOLVED_STATUSES = {"Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"}
DTI_MISSING_CODE = 999
EMPLOYMENT_ORDER = [
    "desconocido", "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years",
]
RATING_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
# Categories seen fewer times than this share one infrequent column (e.g. vivienda ANY/NONE/OTHER).
MIN_CATEGORY_FREQUENCY = 200


def clean_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only deterministic, row-preserving preparation used by all models."""
    missing = sorted(set(RAW_INPUT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    result = df.loc[:, RAW_INPUT_COLUMNS].copy()
    result["antigüedad_empleo"] = result["antigüedad_empleo"].fillna("desconocido")
    result["sector_empleo"] = sector_empleo.asignar_sector(result.pop("empleo"))
    result["dti"] = result["dti"].clip(0, 100)
    result["porc_uso_revolving"] = result["porc_uso_revolving"].clip(0, 100)
    return result


def make_preprocessor() -> Pipeline:
    """Mirror of the preprocessor designed in Notebooks/04 (see 01_Documentos/Diseño_Transformaciones.md)."""
    categorical = OneHotEncoder(
        drop="first",
        min_frequency=MIN_CATEGORY_FREQUENCY,
        handle_unknown="infrequent_if_exist",
        sparse_output=False,
    )
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("encode", OrdinalEncoder(categories=[EMPLOYMENT_ORDER, RATING_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scale", StandardScaler()),
    ])
    skewed = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    symmetric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)), ("binarize", Binarizer(threshold=0))])
    columns = ColumnTransformer([
        ("ohe", categorical, OHE_COLUMNS),
        ("oe", ordinal, ORDINAL_COLUMNS),
        ("yj", skewed, SKEWED_NUMERIC_COLUMNS),
        ("ss", symmetric, SYMMETRIC_NUMERIC_COLUMNS),
        ("bin", binary, BINARIZED_COLUMNS),
    ], remainder="drop")
    return Pipeline([("clean", FunctionTransformer(clean_raw_input, validate=False)), ("columns", columns)])


def make_model_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("model", model)])


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result = result.loc[result["ingresos"].fillna(0) <= 400000].copy()
    result = result.loc[result["dti"] != DTI_MISSING_CODE]
    result = result.loc[~result["estado"].isin(UNRESOLVED_STATUSES)].copy()
    result["target_pd"] = result["estado"].isin(DEFAULT_STATUSES).astype(int)
    pending = result["principal"] - result["imp_amortizado"]
    result["target_ead"] = (pending / result["principal"]).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    result["target_lgd"] = (1 - (result["imp_recuperado"] / pending)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 1)
    return result


raw = pd.read_pickle(TRAIN_PATH)
dataset = build_targets(raw)
train_frame, test_frame = train_test_split(
    dataset, test_size=0.30, stratify=dataset["target_pd"], random_state=RANDOM_STATE
)

# Best configuration from the grid search in Notebooks/05 (DEC-008, DEC-010): Ridge with C = 10**-0.5 (~0.316).
pd_pipeline = make_model_pipeline(LogisticRegression(solver="saga", l1_ratio=0.0, C=10 ** -0.5, max_iter=3000, random_state=RANDOM_STATE))
ead_pipeline = make_model_pipeline(HistGradientBoostingRegressor(learning_rate=0.1, max_iter=100, max_depth=5, min_samples_leaf=50, l2_regularization=1.0, random_state=RANDOM_STATE))
lgd_pipeline = make_model_pipeline(HistGradientBoostingRegressor(learning_rate=0.01, max_iter=100, max_depth=5, min_samples_leaf=50, l2_regularization=0.5, random_state=RANDOM_STATE))

pd_pipeline.fit(train_frame, train_frame["target_pd"])
default_train = train_frame.loc[train_frame["target_pd"] == 1]
default_test = test_frame.loc[test_frame["target_pd"] == 1]
ead_pipeline.fit(default_train, default_train["target_ead"])
lgd_pipeline.fit(default_train, default_train["target_lgd"])

pd_auc = roc_auc_score(test_frame["target_pd"], pd_pipeline.predict_proba(test_frame)[:, 1])
ead_mae = mean_absolute_error(default_test["target_ead"], np.clip(ead_pipeline.predict(default_test), 0, 1))
lgd_mae = mean_absolute_error(default_test["target_lgd"], np.clip(lgd_pipeline.predict(default_test), 0, 1))

# Refit the released artefacts on every available training row after evaluation.
pd_pipeline.fit(dataset, dataset["target_pd"])
default_dataset = dataset.loc[dataset["target_pd"] == 1]
ead_pipeline.fit(default_dataset, default_dataset["target_ead"])
lgd_pipeline.fit(default_dataset, default_dataset["target_lgd"])

artefact = {
    "version": "1.3.0",
    "raw_input_columns": RAW_INPUT_COLUMNS,
    "models": {"pd": pd_pipeline, "ead": ead_pipeline, "lgd": lgd_pipeline},
    "evaluation": {"pd_roc_auc": float(pd_auc), "ead_mae": float(ead_mae), "lgd_mae": float(lgd_mae)},
    "training_rows": {"pd": int(len(dataset)), "ead_lgd": int(len(default_dataset))},
}
for path in (ARTEFACT_PATH, API_ARTEFACT_PATH):
    with path.open("wb") as stream:
        cloudpickle.dump(artefact, stream)
    print(f"Artefact written to: {path}")
print(f"PD ROC-AUC: {pd_auc:.4f}")
print(f"EAD MAE: {ead_mae:.4f}")
print(f"LGD MAE: {lgd_mae:.4f}")
