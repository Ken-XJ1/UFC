"""Entrena y compara modelos UFC con datos previos y cuotas históricas."""

from __future__ import annotations

import argparse
import io
import re
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw" / "external"
RAW_PATH = RAW_DIR / "ultimate_ufc_dataset.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
SOURCE_URL = (
    "https://raw.githubusercontent.com/shortlikeafox/"
    "ultimate_ufc_dataset/main/ufc-master.csv"
)
REQUIRED_COLUMNS = {
    "R_fighter",
    "B_fighter",
    "R_odds",
    "B_odds",
    "date",
    "Winner",
}
POST_FIGHT_COLUMNS = {
    "Winner",
    "finish",
    "finish_details",
    "finish_round",
    "finish_round_time",
    "total_fight_time_secs",
    "date",
    "R_fighter",
    "B_fighter",
}
RANDOM_STATE = 42
METHOD_LABELS = {
    "KO/TKO": "KO/TKO",
    "SUB": "Sumisión",
    "U-DEC": "Decisión",
    "S-DEC": "Decisión",
    "M-DEC": "Decisión",
}


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def pair_key(first: object, second: object) -> tuple[str, str]:
    return tuple(sorted((normalize_name(first), normalize_name(second))))


def refresh_source() -> None:
    """Descarga y valida el CSV público antes de reemplazar la copia local."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "ufc-predictor-data-refresh"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    if not payload or not payload.startswith(b"R_fighter,B_fighter,"):
        raise ValueError("La descarga no parece ser el CSV UFC esperado.")
    preview = pd.read_csv(io.BytesIO(payload), nrows=2)
    absent = REQUIRED_COLUMNS - set(preview.columns)
    if absent:
        raise ValueError(f"Faltan columnas requeridas: {sorted(absent)}")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix="_ufc_dataset_", suffix=".csv",
        dir=RAW_DIR, delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    temporary_path.replace(RAW_PATH)
    print("Dataset de UFC con cuotas actualizado desde GitHub.")


def load_source() -> pd.DataFrame:
    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"No se encontró {RAW_PATH}. Ejecuta este script con --refresh."
        )
    source = pd.read_csv(RAW_PATH, low_memory=False).copy()
    absent = REQUIRED_COLUMNS - set(source.columns)
    if absent:
        raise ValueError(f"Faltan columnas requeridas: {sorted(absent)}")
    source["event_date"] = pd.to_datetime(
        source["date"].astype("string").str.slice(0, 10), errors="coerce"
    )
    source["pair_key"] = [
        pair_key(first, second)
        for first, second in zip(source["R_fighter"], source["B_fighter"])
    ]
    source["red_key"] = source["R_fighter"].map(normalize_name)
    source["blue_key"] = source["B_fighter"].map(normalize_name)
    source = source.loc[
        source["event_date"].notna()
        & source["Winner"].isin(["Red", "Blue"])
    ].copy()
    if source.duplicated(["event_date", "pair_key"]).any():
        raise ValueError("Hay peleas duplicadas por fecha y participantes.")
    source["target_red"] = source["Winner"].eq("Red").astype("int64")
    return source.reset_index(drop=True)


def validate_labels_against_ufcstats(source: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Descarta discrepancias detectables contra el resultado de UFCStats."""
    processed_path = PROCESSED_DIR / "ufc_features_historicas.csv"
    if not processed_path.is_file():
        raise FileNotFoundError(
            "Se necesita data/processed/ufc_features_historicas.csv para "
            "verificar las etiquetas antes de entrenar. Ejecuta primero "
            "src/train_updated_model.py."
        )
    local = pd.read_csv(processed_path, low_memory=False)
    needed = {
        "event_date",
        "fighter1",
        "fighter2",
        "winner_side",
    }
    absent = needed - set(local.columns)
    if absent:
        raise ValueError(
            f"El archivo procesado de UFCStats no tiene columnas: {sorted(absent)}"
        )
    local["event_date"] = pd.to_datetime(local["event_date"], errors="coerce")
    local["pair_key"] = [
        pair_key(first, second)
        for first, second in zip(local["fighter1"], local["fighter2"])
    ]
    local["fighter1_key"] = local["fighter1"].map(normalize_name)
    local["fighter2_key"] = local["fighter2"].map(normalize_name)
    local["local_winner_key"] = np.where(
        local["winner_side"].eq(1),
        local["fighter1_key"],
        local["fighter2_key"],
    )
    local_check = local[["event_date", "pair_key", "local_winner_key"]]
    if local_check.duplicated(["event_date", "pair_key"]).any():
        raise ValueError("UFCStats tiene claves fecha/pareja ambiguas.")
    local_check = local_check.drop_duplicates(["event_date", "pair_key"])
    audited = source.merge(
        local_check,
        on=["event_date", "pair_key"],
        how="left",
        validate="one_to_one",
    )
    matched = audited["local_winner_key"].notna()
    expected_winner = np.where(
        audited["target_red"].eq(1),
        audited["red_key"],
        audited["blue_key"],
    )
    conflicts = matched & (expected_winner != audited["local_winner_key"])
    conflict_count = int(conflicts.sum())
    verified_count = int(matched.sum())
    audited = audited.loc[~conflicts].drop(columns="local_winner_key")
    return audited.reset_index(drop=True), verified_count, conflict_count


def feature_columns(source: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    features = [
        column
        for column in source.columns
        if column not in POST_FIGHT_COLUMNS
        and column
        not in {
            "event_date",
            "pair_key",
            "red_key",
            "blue_key",
            "target_red",
        }
    ]
    categorical = [
        column
        for column in features
        if pd.api.types.is_string_dtype(source[column])
        or pd.api.types.is_bool_dtype(source[column])
    ]
    numeric = [column for column in features if column not in categorical]
    leakage = set(features) & POST_FIGHT_COLUMNS
    if leakage:
        raise ValueError(f"Se detectaron variables con posible fuga: {sorted(leakage)}")
    return features, numeric, categorical


def preprocessor(
    numeric: list[str], categorical: list[str], scale_numeric: bool
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = make_pipeline(*(step for _, step in numeric_steps))
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def models(
    numeric: list[str], categorical: list[str]
) -> dict[str, object]:
    return {
        "Regresión logística + cuotas": make_pipeline(
            preprocessor(numeric, categorical, scale_numeric=True),
            LogisticRegression(max_iter=3000, C=0.2, random_state=RANDOM_STATE),
        ),
        "Random Forest + cuotas": make_pipeline(
            preprocessor(numeric, categorical, scale_numeric=False),
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "XGBoost + cuotas": make_pipeline(
            preprocessor(numeric, categorical, scale_numeric=False),
            XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=5,
                reg_lambda=5,
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def add_metrics(
    rows: list[dict[str, object]],
    name: str,
    truth: pd.Series,
    predicted: np.ndarray | pd.Series,
) -> None:
    rows.append(
        {
            "model": name,
            "accuracy": accuracy_score(truth, predicted),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "n_test": int(len(truth)),
        }
    )


def save_comparison_figure(metrics: pd.DataFrame) -> Path:
    plot_rows = metrics.loc[
        metrics["model"].isin(
            [
                "Regresión logística + cuotas",
                "Random Forest + cuotas",
                "XGBoost + cuotas",
                "Random Forest sin cuotas",
                "Baseline: clase más frecuente",
                "Favorito del mercado",
            ]
        )
    ].copy()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(plot_rows))
    width = 0.36
    ax.bar(
        positions - width / 2,
        plot_rows["accuracy"],
        width,
        label="Accuracy",
        color="#4C78A8",
    )
    ax.bar(
        positions + width / 2,
        plot_rows["balanced_accuracy"],
        width,
        label="Balanced accuracy",
        color="#F28E2B",
    )
    ax.set_xticks(positions, plot_rows["model"], rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proporción de aciertos")
    ax.set_title("Modelos con cuotas — prueba cronológica")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    ax.legend(loc="upper left")
    for bars in ax.containers:
        ax.bar_label(
            bars,
            labels=[f"{value:.1%}" for value in bars.datavalues],
            padding=3,
            fontsize=8,
        )
    fig.tight_layout()
    output = FIGURES_DIR / "17_comparacion_modelos_con_cuotas.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def save_confusion_figure(truth: pd.Series, predicted: np.ndarray) -> Path:
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Azul", "Rojo"],
        yticklabels=["Azul", "Rojo"],
        xlabel="Predicción",
        ylabel="Resultado real",
        title="Matriz de confusión — Random Forest con cuotas",
    )
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    output = FIGURES_DIR / "18_matriz_confusion_con_cuotas.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def train_method_model(
    source: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
) -> dict[str, object]:
    """Entrena probabilidades para esquina ganadora y método del resultado."""
    eligible = source.copy()
    finish = eligible["finish"].fillna("").astype("string").str.upper().str.strip()
    eligible["method"] = finish.map(METHOD_LABELS)
    eligible = eligible.loc[eligible["method"].notna()].copy()
    eligible["target_method_outcome"] = (
        eligible["Winner"].map({"Red": "Esquina roja", "Blue": "Esquina azul"})
        + " — "
        + eligible["method"]
    )

    dates = pd.DatetimeIndex(
        eligible["event_date"].drop_duplicates().sort_values()
    )
    if len(dates) < 5:
        raise ValueError("No hay suficientes fechas para modelar los métodos.")
    split_date = dates[int(len(dates) * 0.8)]
    train_mask = eligible["event_date"] < split_date
    test_mask = ~train_mask
    y_train = eligible.loc[train_mask, "target_method_outcome"]
    y_test = eligible.loc[test_mask, "target_method_outcome"]
    if y_train.nunique() != 6 or y_test.nunique() != 6:
        raise ValueError(
            "Se requieren los seis resultados (esquina y método) "
            "en train y test."
        )

    model = make_pipeline(
        preprocessor(numeric, categorical, scale_numeric=False),
        RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    )
    model.fit(eligible.loc[train_mask, features], y_train)
    x_test = eligible.loc[test_mask, features]
    predicted = model.predict(x_test)
    probabilities = model.predict_proba(x_test)
    classifier = model.steps[-1][1]
    class_labels = list(classifier.classes_)

    majority_class = y_train.mode().iloc[0]
    majority_prediction = np.full(len(y_test), majority_class)
    metrics = pd.DataFrame(
        [
            {
                "model": "Random Forest — esquina y método",
                "accuracy": accuracy_score(y_test, predicted),
                "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                "macro_f1": f1_score(
                    y_test,
                    predicted,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                ),
                "n_test": len(y_test),
                "train_end": eligible.loc[train_mask, "event_date"].max(),
                "test_start": eligible.loc[test_mask, "event_date"].min(),
                "test_end": eligible.loc[test_mask, "event_date"].max(),
            },
            {
                "model": "Baseline — clase más frecuente",
                "accuracy": accuracy_score(y_test, majority_prediction),
                "balanced_accuracy": balanced_accuracy_score(
                    y_test, majority_prediction
                ),
                "macro_f1": f1_score(
                    y_test,
                    majority_prediction,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                ),
                "n_test": len(y_test),
                "train_end": eligible.loc[train_mask, "event_date"].max(),
                "test_start": eligible.loc[test_mask, "event_date"].min(),
                "test_end": eligible.loc[test_mask, "event_date"].max(),
            },
        ]
    )
    metrics.to_csv(
        REPORTS_DIR / "metricas_metodo_victoria.csv", index=False
    )
    report = classification_report(
        y_test,
        predicted,
        labels=class_labels,
        zero_division=0,
    )
    (REPORTS_DIR / "classification_report_metodo_victoria.txt").write_text(
        report, encoding="utf-8"
    )

    prediction_rows = eligible.loc[
        test_mask,
        ["event_date", "R_fighter", "B_fighter", "Winner", "finish", "target_method_outcome"],
    ].copy()
    prediction_rows["predicted_outcome"] = predicted
    for index, label in enumerate(class_labels):
        key = (
            label.lower()
            .replace("esquina ", "")
            .replace(" — ", "_")
            .replace("/", "_")
            .replace("ó", "o")
        )
        prediction_rows[f"probability_{key}"] = probabilities[:, index]
    prediction_rows.to_csv(
        REPORTS_DIR / "predicciones_metodo_victoria.csv", index=False
    )

    matrix = confusion_matrix(y_test, predicted, labels=class_labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(row_percentages, cmap="Blues", vmin=0, vmax=1)
    short_labels = [
        label.replace(" — ", "\n") for label in class_labels
    ]
    ax.set(
        xticks=np.arange(len(class_labels)),
        yticks=np.arange(len(class_labels)),
        xticklabels=short_labels,
        yticklabels=short_labels,
        xlabel="Lo que predijo el modelo",
        ylabel="Lo que ocurrió realmente",
        title=(
            "Resultado real vs. predicción\n"
            "Cada fila suma 100%; la celda muestra % y número de peleas"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for row in range(len(class_labels)):
        for column in range(len(class_labels)):
            percentage = row_percentages[row, column]
            color = "white" if percentage >= 0.55 else "black"
            ax.text(
                column,
                row,
                f"{percentage:.0%}\n(n={matrix[row, column]})",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    colorbar = fig.colorbar(image, ax=ax, format=plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    colorbar.set_label("Porcentaje dentro del resultado real de esa fila")
    fig.tight_layout()
    confusion_path = FIGURES_DIR / "19_matriz_confusion_metodo_victoria.png"
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    dump(
        {
            "model": model,
            "feature_columns": features,
            "classes": class_labels,
            "target_definition": (
                "Esquina roja/azul y método: KO/TKO, sumisión o decisión."
            ),
            "excluded_finish_categories": [
                "DQ",
                "CNC",
                "Overturned",
                "unknown",
            ],
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_end": eligible.loc[train_mask, "event_date"].max(),
            "test_start": eligible.loc[test_mask, "event_date"].min(),
            "test_end": eligible.loc[test_mask, "event_date"].max(),
            "metrics": metrics.to_dict(orient="records"),
            "source": SOURCE_URL,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "random_forest_ufc_method.joblib",
    )
    return {
        "metrics": metrics,
        "rows": len(eligible),
        "excluded_rows": len(source) - len(eligible),
        "split_date": split_date,
        "class_counts": eligible["target_method_outcome"].value_counts(),
        "confusion_path": confusion_path,
    }


def save_report(
    metrics: pd.DataFrame,
    dataset: pd.DataFrame,
    split_date: pd.Timestamp,
    verified_count: int,
    conflict_count: int,
    comparison_path: Path,
    confusion_path: Path,
) -> None:
    document = Document()
    document.core_properties.title = "Predicción UFC con cuotas históricas"
    document.add_heading("Predicción de peleas UFC con cuotas", 0)
    document.add_paragraph(
        f"Datos de {dataset['event_date'].min().date()} a "
        f"{dataset['event_date'].max().date()}; "
        f"{len(dataset):,} peleas con resultado Red/Blue."
    )
    document.add_paragraph(
        f"Se entrenó con fechas anteriores al {split_date.date()} y se probó "
        "con las posteriores, manteniendo juntas las peleas de cada fecha."
    )
    document.add_paragraph(
        f"Se verificaron {verified_count:,} etiquetas frente al historial de "
        f"UFCStats y se descartaron {conflict_count:,} discrepancias. El "
        "conjunto fuente identifica las esquinas roja y azul."
    )
    table = document.add_table(rows=1, cols=4)
    for cell, title in zip(
        table.rows[0].cells,
        ["Modelo", "Accuracy", "Balanced accuracy", "Peleas de prueba"],
    ):
        cell.text = title
    for _, row in metrics.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["model"])
        cells[1].text = f"{row['accuracy']:.1%}"
        cells[2].text = f"{row['balanced_accuracy']:.1%}"
        cells[3].text = f"{int(row['n_test']):,}"
    document.add_picture(str(comparison_path), width=Inches(6.2))
    document.add_heading("Matriz de confusión", level=1)
    document.add_picture(str(confusion_path), width=Inches(4.8))
    document.add_heading("Límites y fuente", level=1)
    document.add_paragraph(
        "Se excluyeron del entrenamiento el ganador y todas las variables "
        "posteriores al combate (método, ronda y duración). Las cuotas son "
        "líneas históricas de cierre descritas por la fuente; una evaluación "
        "con cuotas no equivale a predecir sin conocer el mercado. La prueba "
        "es histórica y no garantiza resultados futuros."
    )
    document.add_paragraph(
        "Fuente: Ultimate UFC Dataset, de Matt Dabbert y "
        "shortlikeafox; dataset publicado con licencia CC BY 4.0. "
        "https://github.com/shortlikeafox/ultimate_ufc_dataset"
    )
    document.save(REPORTS_DIR / "Informe_UFC_con_cuotas.docx")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena modelos UFC con odds y estadísticas históricas."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Descarga la versión más reciente del dataset público.",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh_source()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    source = load_source()
    source, verified_count, conflict_count = validate_labels_against_ufcstats(
        source
    )
    if source.empty:
        raise ValueError("No quedan filas válidas tras validar los resultados.")

    unique_dates = pd.DatetimeIndex(
        source["event_date"].drop_duplicates().sort_values()
    )
    if len(unique_dates) < 5:
        raise ValueError("Se necesitan al menos cinco fechas para evaluar.")
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_mask = source["event_date"] < split_date
    test_mask = ~train_mask
    if source.loc[train_mask, "target_red"].nunique() != 2:
        raise ValueError("El conjunto de entrenamiento debe contener ambas clases.")
    if source.loc[test_mask, "target_red"].nunique() != 2:
        raise ValueError("El conjunto de prueba debe contener ambas clases.")

    features, numeric, categorical = feature_columns(source)
    no_odds_features = [
        column
        for column in features
        if "odds" not in column.lower()
        and not column.lower().endswith("_ev")
    ]
    no_odds_numeric = [
        column for column in no_odds_features if column in numeric
    ]
    no_odds_categorical = [
        column for column in no_odds_features if column in categorical
    ]
    x_train = source.loc[train_mask, features]
    x_test = source.loc[test_mask, features]
    y_train = source.loc[train_mask, "target_red"]
    y_test = source.loc[test_mask, "target_red"]
    metrics_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    classification_reports: list[str] = []

    for name, model in models(numeric, categorical).items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        predictions[name] = predicted
        fitted[name] = model
        add_metrics(metrics_rows, name, y_test, predicted)
        classification_reports.append(
            f"{name}\n"
            + classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["Esquina azul", "Esquina roja"],
                zero_division=0,
            )
        )

    no_odds_name = "Random Forest sin cuotas"
    no_odds_model = models(no_odds_numeric, no_odds_categorical)[
        "Random Forest + cuotas"
    ]
    no_odds_model.fit(
        source.loc[train_mask, no_odds_features], y_train
    )
    no_odds_prediction = no_odds_model.predict(
        source.loc[test_mask, no_odds_features]
    )
    add_metrics(metrics_rows, no_odds_name, y_test, no_odds_prediction)
    classification_reports.append(
        f"{no_odds_name}\n"
        + classification_report(
            y_test,
            no_odds_prediction,
            labels=[0, 1],
            target_names=["Esquina azul", "Esquina roja"],
            zero_division=0,
        )
    )

    majority_class = int(y_train.mode().iloc[0])
    add_metrics(
        metrics_rows,
        "Baseline: clase más frecuente",
        y_test,
        np.full(len(y_test), majority_class),
    )
    red_odds = pd.to_numeric(source.loc[test_mask, "R_odds"], errors="coerce")
    blue_odds = pd.to_numeric(source.loc[test_mask, "B_odds"], errors="coerce")
    valid_odds = red_odds.notna() & blue_odds.notna()
    favorite_prediction = red_odds.loc[valid_odds].lt(
        blue_odds.loc[valid_odds]
    ).astype("int64")
    add_metrics(
        metrics_rows,
        "Favorito del mercado",
        y_test.loc[valid_odds],
        favorite_prediction.to_numpy(),
    )

    prediction_rows = source.loc[
        test_mask,
        ["event_date", "R_fighter", "B_fighter", "Winner", "target_red"],
    ].copy()
    rf_name = "Random Forest + cuotas"
    rf_model = fitted[rf_name]
    rf_prediction = predictions[rf_name]
    rf_probability = rf_model.predict_proba(x_test)[:, 1]
    prediction_rows["predicted_red"] = rf_prediction
    prediction_rows["probability_red"] = rf_probability
    prediction_rows["probability_blue"] = 1 - rf_probability
    prediction_rows.to_csv(
        REPORTS_DIR / "predicciones_prueba_con_cuotas.csv", index=False
    )

    old_predictions_path = REPORTS_DIR / "predicciones_prueba_ufcstats.csv"
    comparison_n = 0
    if old_predictions_path.is_file():
        old = pd.read_csv(old_predictions_path, low_memory=False)
        old["event_date"] = pd.to_datetime(old["event_date"], errors="coerce")
        old["pair_key"] = [
            pair_key(first, second)
            for first, second in zip(old["fighter1"], old["fighter2"])
        ]
        old["fighter1_key"] = old["fighter1"].map(normalize_name)
        old["fighter2_key"] = old["fighter2"].map(normalize_name)
        comparison = source.loc[
            test_mask,
            [
                "event_date",
                "pair_key",
                "red_key",
                "target_red",
                "R_odds",
                "B_odds",
            ],
        ].merge(
            old[
                [
                    "event_date",
                    "pair_key",
                    "fighter1_key",
                    "fighter2_key",
                    "predicted_side",
                ]
            ],
            on=["event_date", "pair_key"],
            how="inner",
            validate="one_to_one",
        )
        predicted_winner_key = np.where(
            comparison["predicted_side"].eq(1),
            comparison["fighter1_key"],
            comparison["fighter2_key"],
        )
        old_prediction_red = (
            predicted_winner_key == comparison["red_key"]
        ).astype("int64")
        comparison_n = len(comparison)
        add_metrics(
            metrics_rows,
            "Modelo anterior (mismas peleas)",
            comparison["target_red"],
            old_prediction_red,
        )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(
        REPORTS_DIR / "metricas_modelos_con_cuotas.csv", index=False
    )
    (REPORTS_DIR / "classification_report_con_cuotas.txt").write_text(
        "\n\n".join(classification_reports), encoding="utf-8"
    )
    processed_columns = [
        "event_date",
        "R_fighter",
        "B_fighter",
        "Winner",
        "target_red",
        *features,
    ]
    source[processed_columns].to_csv(
        PROCESSED_DIR / "ufc_features_con_cuotas.csv", index=False
    )

    preprocessor_step = rf_model.steps[0][1]
    feature_importance = pd.DataFrame(
        {
            "feature": preprocessor_step.get_feature_names_out(),
            "importance": rf_model.steps[-1][1].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        REPORTS_DIR / "importancia_variables_con_cuotas.csv", index=False
    )
    figure_path = save_comparison_figure(metrics)
    confusion_path = save_confusion_figure(y_test, rf_prediction)
    save_report(
        metrics,
        source,
        split_date,
        verified_count,
        conflict_count,
        figure_path,
        confusion_path,
    )
    dump(
        {
            "model": rf_model,
            "feature_columns": features,
            "target_definition": "1 = gana esquina roja; 0 = gana esquina azul",
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "train_end": source.loc[train_mask, "event_date"].max(),
            "test_start": source.loc[test_mask, "event_date"].min(),
            "test_end": source.loc[test_mask, "event_date"].max(),
            "metrics": metrics.to_dict(orient="records"),
            "verified_rows_against_ufcstats": verified_count,
            "discarded_label_conflicts": conflict_count,
            "source": SOURCE_URL,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "random_forest_ufc_odds.joblib",
    )

    method_result = train_method_model(
        source,
        features,
        numeric,
        categorical,
    )

    print(f"Filas binarias utilizables: {len(source):,}")
    print(
        f"Etiquetas comprobadas contra UFCStats: {verified_count:,}; "
        f"discrepancias descartadas: {conflict_count:,}"
    )
    print(f"Variables previas al combate: {len(features)}")
    print(
        f"Train: {int(train_mask.sum()):,} peleas hasta "
        f"{source.loc[train_mask, 'event_date'].max().date()}"
    )
    print(
        f"Test: {int(test_mask.sum()):,} peleas desde "
        f"{source.loc[test_mask, 'event_date'].min().date()} hasta "
        f"{source.loc[test_mask, 'event_date'].max().date()}"
    )
    print("\nMétricas:")
    print(
        metrics.to_string(
            index=False,
            formatters={
                "accuracy": "{:.3f}".format,
                "balanced_accuracy": "{:.3f}".format,
            },
        )
    )
    if comparison_n:
        print(f"\nModelo anterior comparado en {comparison_n:,} peleas comunes.")
    print(f"\nInforme: {REPORTS_DIR / 'Informe_UFC_con_cuotas.docx'}")
    method_metrics = method_result["metrics"].iloc[0]
    print(
        "\nPredicción adicional de esquina y método:"
        f"\nPeleas con método clasificado: {method_result['rows']:,}; "
        f"excluidas por método desconocido/no compatible: "
        f"{method_result['excluded_rows']:,}"
        f"\nTrain hasta {method_metrics['train_end'].date()}; "
        f"test desde {method_metrics['test_start'].date()} hasta "
        f"{method_metrics['test_end'].date()} "
        f"({int(method_metrics['n_test']):,} peleas)"
        f"\nAccuracy: {method_metrics['accuracy']:.3f}; "
        f"balanced accuracy: {method_metrics['balanced_accuracy']:.3f}; "
        f"macro F1: {method_metrics['macro_f1']:.3f}"
        f"\nMétricas: {REPORTS_DIR / 'metricas_metodo_victoria.csv'}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error al entrenar con cuotas: {error}", file=sys.stderr)
        raise
