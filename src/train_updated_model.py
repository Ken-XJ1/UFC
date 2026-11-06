"""Prepara UFCStats, crea variables previas a cada pelea y evalúa modelos."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
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
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw" / "ufcstats"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_DIR / "models"
GITHUB_RAW = (
    "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"
)
SOURCE_FILES = (
    "ufc_event_details.csv",
    "ufc_fight_details.csv",
    "ufc_fight_results.csv",
    "ufc_fight_stats.csv",
    "ufc_fighter_tott.csv",
    "ufc_fighter_details.csv",
)
NUMERIC_FEATURES = [
    "fighter1_age",
    "fighter2_age",
    "fighter1_height_cm",
    "fighter2_height_cm",
    "fighter1_reach_cm",
    "fighter2_reach_cm",
    "fighter1_previous_fights",
    "fighter2_previous_fights",
    "fighter1_previous_wins",
    "fighter2_previous_wins",
    "fighter1_previous_losses",
    "fighter2_previous_losses",
    "fighter1_previous_win_rate",
    "fighter2_previous_win_rate",
    "fighter1_current_streak",
    "fighter2_current_streak",
    "fighter1_previous_stats_fights",
    "fighter2_previous_stats_fights",
    "fighter1_avg_sig_strikes_landed",
    "fighter2_avg_sig_strikes_landed",
    "fighter1_avg_sig_strikes_accuracy",
    "fighter2_avg_sig_strikes_accuracy",
    "fighter1_avg_takedowns_landed",
    "fighter2_avg_takedowns_landed",
    "fighter1_avg_knockdowns",
    "fighter2_avg_knockdowns",
]
CATEGORICAL_FEATURES = [
    "fighter1_stance",
    "fighter2_stance",
    "weightclass",
]
DIFFERENCE_METRICS = [
    "age",
    "height_cm",
    "reach_cm",
    "previous_fights",
    "previous_wins",
    "previous_losses",
    "previous_win_rate",
    "current_streak",
    "avg_sig_strikes_landed",
    "avg_sig_strikes_accuracy",
    "avg_takedowns_landed",
    "avg_knockdowns",
]
NUMERIC_FEATURES.extend(
    f"difference_{metric}" for metric in DIFFERENCE_METRICS
)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def normalize_name(value: object) -> str:
    """Normaliza texto para unir nombres sin depender de mayúsculas o signos."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def normalize_event(value: object) -> str:
    """Unifica diferencias de prefijo entre UFC Fight Night y Noche UFC."""
    normalized = normalize_name(value)
    for prefix in ("ufc fight night ", "noche ufc "):
        if normalized.startswith(prefix):
            return normalized[len(prefix):].strip()
    return normalized


def parse_fraction(value: object) -> tuple[float, float]:
    """Lee valores de UFCStats con formato 'acertados de intentos'."""
    if pd.isna(value):
        return 0.0, 0.0
    match = re.search(r"(\d+)\s+of\s+(\d+)", str(value), flags=re.IGNORECASE)
    if not match:
        return 0.0, 0.0
    return float(match.group(1)), float(match.group(2))


def parse_height_cm(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+)\s*'\s*(\d+)?", str(value))
    if not match:
        return np.nan
    feet = int(match.group(1))
    inches = int(match.group(2) or 0)
    return (feet * 12 + inches) * 2.54


def parse_reach_cm(value: object) -> float:
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) * 2.54 if match else np.nan


def parse_dob(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value) or str(value).strip() in {"", "--"}:
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def refresh_source_files() -> None:
    """Descarga de GitHub los seis CSV y reemplaza los locales tras validarlos."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="_download_staging_", dir=RAW_DIR
    ) as staging_name:
        staging = Path(staging_name)
        for filename in SOURCE_FILES:
            url = f"{GITHUB_RAW}/{filename}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "ufc-predictor-data-refresh"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if not payload or not payload.startswith(
                (b"EVENT,", b"FIGHTER,", b"FIRST,")
            ):
                raise ValueError(f"Descarga vacía o CSV inesperado: {filename}")
            (staging / filename).write_bytes(payload)
            pd.read_csv(staging / filename, nrows=2)
        for filename in SOURCE_FILES:
            (staging / filename).replace(RAW_DIR / filename)
        print("CSV actualizados desde el repositorio público.")


def load_source_tables() -> tuple[pd.DataFrame, ...]:
    missing = [name for name in SOURCE_FILES if not (RAW_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan CSV de UFCStats en "
            f"{RAW_DIR}: {', '.join(missing)}"
        )
    events, details, results, stats, fighters, _fighter_details = (
        pd.read_csv(RAW_DIR / filename, low_memory=False)
        for filename in SOURCE_FILES
    )
    required = {
        "events": (events, {"EVENT", "DATE"}),
        "details": (details, {"EVENT", "BOUT", "URL"}),
        "results": (results, {"EVENT", "BOUT", "URL", "OUTCOME"}),
        "stats": (stats, {"EVENT", "BOUT", "FIGHTER", "ROUND"}),
        "fighters": (fighters, {"FIGHTER", "HEIGHT", "REACH", "STANCE", "DOB"}),
    }
    for name, (frame, columns) in required.items():
        absent = columns - set(frame.columns)
        if absent:
            raise ValueError(f"Faltan columnas en {name}: {sorted(absent)}")
    return events, details, results, stats, fighters


def prepare_fight_rows(
    events: pd.DataFrame, details: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, int, int]:
    event_dates = events[["EVENT", "DATE"]].copy()
    event_dates["event_key"] = event_dates["EVENT"].map(normalize_event)
    event_dates["event_date"] = pd.to_datetime(
        event_dates["DATE"], errors="coerce"
    )
    event_dates = event_dates.drop_duplicates("event_key")
    date_lookup = event_dates.set_index("event_key")["event_date"]

    fight_details = details[["EVENT", "BOUT", "URL"]].copy()
    fight_details["event_key"] = fight_details["EVENT"].map(normalize_event)
    fight_details["bout_key"] = fight_details["BOUT"].map(normalize_name)
    fight_details["event_has_date"] = fight_details["event_key"].isin(
        date_lookup.index
    )
    fight_details = (
        fight_details.sort_values("event_has_date", ascending=False)
        .drop_duplicates("URL")
    )

    fights = results.copy()
    fights["event_key_result"] = fights["EVENT"].map(normalize_event)
    fights["event_has_date"] = fights["event_key_result"].isin(date_lookup.index)
    fights = (
        fights.sort_values("event_has_date", ascending=False)
        .drop_duplicates("URL")
        .merge(
            fight_details[["URL", "EVENT", "BOUT", "event_key", "bout_key"]],
            on="URL",
            how="left",
            suffixes=("_result", "_details"),
            validate="one_to_one",
        )
    )
    fights["event_date"] = fights["event_key"].map(date_lookup)
    fights["bout_key"] = fights["BOUT_result"].map(normalize_name)
    fights["participants"] = fights["BOUT_result"].astype("string").str.split(
        r"\s+vs\.?\s+", regex=True
    )
    valid_pair = fights["participants"].map(
        lambda pair: isinstance(pair, list) and len(pair) == 2
    )
    fights["fighter1"] = fights["participants"].map(
        lambda pair: pair[0].strip() if isinstance(pair, list) and len(pair) == 2 else ""
    )
    fights["fighter2"] = fights["participants"].map(
        lambda pair: pair[1].strip() if isinstance(pair, list) and len(pair) == 2 else ""
    )
    fights["fighter1_key"] = fights["fighter1"].map(normalize_name)
    fights["fighter2_key"] = fights["fighter2"].map(normalize_name)
    fights["winner_side"] = fights["OUTCOME"].map(
        {"W/L": 1, "L/W": 0}
    )
    invalid_pairs = int((~valid_pair).sum())
    missing_dates = int(fights["event_date"].isna().sum())
    fights = fights.loc[valid_pair & fights["event_date"].notna()].copy()
    fights["weightclass"] = fights["WEIGHTCLASS"].astype("string").str.strip()
    return fights, invalid_pairs, missing_dates


def summarize_round_stats(stats: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, float]]:
    rows = stats.copy()
    rows["event_key"] = rows["EVENT"].map(normalize_event)
    rows["bout_key"] = rows["BOUT"].map(normalize_name)
    rows["fighter_key"] = rows["FIGHTER"].map(normalize_name)
    rows = rows.loc[rows["fighter_key"] != ""].copy()
    for column in ["KD", "SIG.STR.", "TOTAL STR.", "TD"]:
        if column not in rows:
            rows[column] = np.nan
    rows["knockdowns"] = pd.to_numeric(rows["KD"], errors="coerce").fillna(0)
    parsed = rows["SIG.STR."].map(parse_fraction)
    rows["sig_landed"] = parsed.map(lambda pair: pair[0])
    rows["sig_attempts"] = parsed.map(lambda pair: pair[1])
    parsed = rows["TOTAL STR."].map(parse_fraction)
    rows["total_landed"] = parsed.map(lambda pair: pair[0])
    parsed = rows["TD"].map(parse_fraction)
    rows["takedowns_landed"] = parsed.map(lambda pair: pair[0])
    columns = [
        "knockdowns", "sig_landed", "sig_attempts",
        "total_landed", "takedowns_landed",
    ]
    grouped = rows.groupby(
        ["event_key", "bout_key", "fighter_key"], dropna=False
    )[columns].sum()
    return grouped.to_dict(orient="index")


def prepare_fighter_profiles(fighters: pd.DataFrame) -> dict[str, dict[str, object]]:
    profiles = fighters.copy()
    profiles["fighter_key"] = profiles["FIGHTER"].map(normalize_name)
    profiles["height_cm"] = profiles["HEIGHT"].map(parse_height_cm)
    profiles["reach_cm"] = profiles["REACH"].map(parse_reach_cm)
    profiles["dob"] = profiles["DOB"].map(parse_dob)
    profiles["stance"] = profiles["STANCE"].replace(
        {"--": np.nan, "": np.nan}
    )
    profiles = profiles.drop_duplicates("fighter_key", keep="first")
    return profiles.set_index("fighter_key")[
        ["height_cm", "reach_cm", "dob", "stance"]
    ].to_dict(orient="index")


def fighter_snapshot(
    fighter_key: str,
    event_date: pd.Timestamp,
    history: dict[str, dict[str, object]],
    profiles: dict[str, dict[str, object]],
) -> dict[str, object]:
    record = history.get(fighter_key, {})
    profile = profiles.get(fighter_key, {})
    previous_fights = int(record.get("fights", 0))
    previous_wins = int(record.get("wins", 0))
    previous_losses = int(record.get("losses", 0))
    decisive_fights = previous_wins + previous_losses
    stats_fights = int(record.get("stats_fights", 0))

    age = np.nan
    dob = profile.get("dob", pd.NaT)
    if pd.notna(dob):
        age = (event_date - dob).days / 365.2425

    values: dict[str, object] = {
        "age": age,
        "height_cm": profile.get("height_cm", np.nan),
        "reach_cm": profile.get("reach_cm", np.nan),
        "stance": profile.get("stance", np.nan),
        "previous_fights": previous_fights,
        "previous_wins": previous_wins,
        "previous_losses": previous_losses,
        "previous_win_rate": (
            previous_wins / decisive_fights if decisive_fights else np.nan
        ),
        "current_streak": int(record.get("streak", 0)),
        "previous_stats_fights": stats_fights,
    }
    for source_name, feature_name in [
        ("sig_landed", "avg_sig_strikes_landed"),
        ("takedowns_landed", "avg_takedowns_landed"),
        ("knockdowns", "avg_knockdowns"),
    ]:
        total = float(record.get(source_name, 0.0))
        values[feature_name] = total / stats_fights if stats_fights else np.nan
    attempts = float(record.get("sig_attempts", 0.0))
    landed = float(record.get("sig_landed", 0.0))
    values["avg_sig_strikes_accuracy"] = landed / attempts if attempts else np.nan
    return values


def create_pre_fight_dataset(
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    fighters: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    stats_lookup = summarize_round_stats(stats)
    profiles = prepare_fighter_profiles(fighters)
    fights = fights.sort_values(
        ["event_date", "URL"], kind="mergesort"
    ).reset_index(drop=True)
    histories: dict[str, dict[str, object]] = {}
    records: list[dict[str, object]] = []
    missing_profile_count = 0
    for _, date_group in fights.groupby("event_date", sort=True):
        date_updates: list[tuple[str, str, int | None, dict[str, float] | None]] = []
        rows_for_date: list[dict[str, object]] = []
        for _, fight in date_group.iterrows():
            fighter1_key = str(fight["fighter1_key"])
            fighter2_key = str(fight["fighter2_key"])
            if fighter1_key not in profiles:
                missing_profile_count += 1
            if fighter2_key not in profiles:
                missing_profile_count += 1
            stats1 = stats_lookup.get(
                (fight["event_key"], fight["bout_key"], fighter1_key)
            )
            stats2 = stats_lookup.get(
                (fight["event_key"], fight["bout_key"], fighter2_key)
            )
            side1 = fighter_snapshot(
                fighter1_key, fight["event_date"], histories, profiles
            )
            side2 = fighter_snapshot(
                fighter2_key, fight["event_date"], histories, profiles
            )
            row: dict[str, object] = {
                "event_date": fight["event_date"],
                "event": fight["EVENT_details"],
                "bout": fight["BOUT_result"],
                "fighter1": fight["fighter1"],
                "fighter2": fight["fighter2"],
                "fight_url": fight["URL"],
                "weightclass": fight["weightclass"],
                "winner_side": fight["winner_side"],
            }
            for feature_name, value in side1.items():
                row[f"fighter1_{feature_name}"] = value
            for feature_name, value in side2.items():
                row[f"fighter2_{feature_name}"] = value
            for feature_name in DIFFERENCE_METRICS:
                value1 = side1[feature_name]
                value2 = side2[feature_name]
                row[f"difference_{feature_name}"] = (
                    value1 - value2
                    if pd.notna(value1) and pd.notna(value2)
                    else np.nan
                )
            rows_for_date.append(row)
            date_updates.append(
                (fighter1_key, fighter2_key, fight["winner_side"], stats1)
            )
            if stats2 is not None:
                date_updates[-1] = (
                    fighter1_key,
                    fighter2_key,
                    fight["winner_side"],
                    {"fighter1_stats": stats1, "fighter2_stats": stats2},  # type: ignore[dict-item]
                )
        records.extend(rows_for_date)
        for fighter1_key, fighter2_key, winner_side, bundled_stats in date_updates:
            history1 = histories.setdefault(
                fighter1_key,
                {
                    "fights": 0, "wins": 0, "losses": 0, "streak": 0,
                    "stats_fights": 0, "sig_landed": 0.0,
                    "sig_attempts": 0.0, "total_landed": 0.0,
                    "takedowns_landed": 0.0,
                    "knockdowns": 0.0,
                },
            )
            history2 = histories.setdefault(
                fighter2_key,
                {
                    "fights": 0, "wins": 0, "losses": 0, "streak": 0,
                    "stats_fights": 0, "sig_landed": 0.0,
                    "sig_attempts": 0.0, "total_landed": 0.0,
                    "takedowns_landed": 0.0,
                    "knockdowns": 0.0,
                },
            )
            history1["fights"] = int(history1["fights"]) + 1
            history2["fights"] = int(history2["fights"]) + 1
            if winner_side == 1:
                history1["wins"] = int(history1["wins"]) + 1
                history2["losses"] = int(history2["losses"]) + 1
                history1["streak"] = max(1, int(history1["streak"]) + 1)
                history2["streak"] = min(-1, int(history2["streak"]) - 1)
            elif winner_side == 0:
                history2["wins"] = int(history2["wins"]) + 1
                history1["losses"] = int(history1["losses"]) + 1
                history2["streak"] = max(1, int(history2["streak"]) + 1)
                history1["streak"] = min(-1, int(history1["streak"]) - 1)
            else:
                history1["streak"] = 0
                history2["streak"] = 0

            if bundled_stats is not None:
                stats1 = bundled_stats.get("fighter1_stats") if isinstance(
                    bundled_stats, dict
                ) else bundled_stats
                stats2 = bundled_stats.get("fighter2_stats") if isinstance(
                    bundled_stats, dict
                ) else None
                if stats1 is not None:
                    for key, value in stats1.items():
                        history1[key] = float(history1[key]) + float(value)
                    history1["stats_fights"] = int(history1["stats_fights"]) + 1
                if stats2 is not None:
                    for key, value in stats2.items():
                        history2[key] = float(history2[key]) + float(value)
                    history2["stats_fights"] = int(history2["stats_fights"]) + 1

    dataset = pd.DataFrame(records)
    dataset = dataset.loc[dataset["winner_side"].notna()].copy()
    dataset["winner_side"] = dataset["winner_side"].astype("int64")
    dataset = dataset.drop_duplicates("fight_url").reset_index(drop=True)
    return dataset, missing_profile_count


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = make_pipeline(*(step[1] for step in numeric_steps))
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_models() -> dict[str, object]:
    return {
        "Regresión logística": make_pipeline(
            make_preprocessor(scale_numeric=True),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Random Forest": make_pipeline(
            make_preprocessor(scale_numeric=False),
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            make_preprocessor(scale_numeric=False),
            XGBClassifier(
                n_estimators=350,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def make_report(
    metrics: pd.DataFrame,
    split_date: pd.Timestamp,
    dataset: pd.DataFrame,
    examples: pd.DataFrame,
    feature_importance: pd.DataFrame,
    confusion_path: Path,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_name = str(metrics.sort_values("accuracy", ascending=False).iloc[0]["model"])
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(
        metrics["model"],
        metrics["accuracy"] * 100,
        color=[
            "#4C78A8" if name != best_name else "#F28E2B"
            for name in metrics["model"]
        ],
    )
    plt.ylabel("Accuracy en prueba (%)")
    plt.title("Comparación de modelos con peleas recientes")
    plt.xticks(rotation=10, ha="right")
    plt.ylim(0, 100)
    for bar, score in zip(bars, metrics["accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{score:.1%}",
            ha="center",
        )
    plt.tight_layout()
    comparison_path = FIGURES_DIR / "14_comparacion_modelos_ufcstats.png"
    plt.savefig(comparison_path, dpi=160)
    plt.close()

    document = Document()
    document.core_properties.title = "Avance con datos UFCStats actualizados"
    document.add_heading("Predicción de resultados de peleas UFC", 0)
    subtitle = document.add_paragraph("Avance del modelo con datos actualizados")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_heading("Qué hicimos", level=1)
    document.add_paragraph(
        f"Usamos {len(dataset):,} peleas con ganador y fecha válida, con "
        f"combates desde {dataset['event_date'].min().date()} hasta "
        f"{dataset['event_date'].max().date()}. Para cada pelea comparamos "
        "edad, estatura, alcance, postura, categoría y estadísticas históricas "
        "de los dos participantes."
    )
    document.add_paragraph(
        "Los promedios, récord y racha se calculan antes de cada combate. Las "
        "estadísticas por ronda de la pelea actual solo se agregan al historial "
        "después de crear sus variables, para evitar que el modelo vea el resultado."
    )
    document.add_heading("Cómo se evaluó", level=1)
    document.add_paragraph(
        f"Se ordenaron las peleas por fecha. El entrenamiento termina antes "
        f"del {split_date.date()} y la prueba usa fechas posteriores. Los "
        "combates de una misma fecha permanecen en el mismo grupo."
    )
    table = document.add_table(rows=1, cols=3)
    for cell, text in zip(
        table.rows[0].cells,
        ["Modelo", "Accuracy", "Balanced accuracy"],
    ):
        cell.text = text
    for _, row in metrics.iterrows():
        cells = table.add_row().cells
        cells[0].text = str(row["model"])
        cells[1].text = f"{row['accuracy']:.1%}"
        cells[2].text = f"{row['balanced_accuracy']:.1%}"
    document.add_picture(str(comparison_path), width=Inches(6.2))
    paragraph = document.add_paragraph(
        "La accuracy mide la proporción total de aciertos. La balanced accuracy "
        "promedia el acierto en ambas clases, para que una clase mayoritaria no "
        "oculte errores en la otra."
    )
    paragraph.paragraph_format.space_after = Pt(8)
    document.add_heading("Matriz de confusión: Random Forest", level=1)
    document.add_paragraph(
        "Las filas representan el ganador real y las columnas la predicción. "
        "0 = segundo peleador del listado; 1 = primero."
    )
    document.add_picture(str(confusion_path), width=Inches(4.8))
    document.add_heading("Variables más utilizadas", level=1)
    if not feature_importance.empty:
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "Variable"
        table.rows[0].cells[1].text = "Importancia"
        for _, row in feature_importance.head(8).iterrows():
            cells = table.add_row().cells
            cells[0].text = str(row["feature"])
            cells[1].text = f"{row['importance']:.3f}"
    document.add_heading("Ejemplos en el conjunto de prueba", level=1)
    if examples.empty:
        document.add_paragraph(
            "No se encontraron en este corte temporal los enfrentamientos "
            "conocidos elegidos como ejemplos."
        )
    else:
        table = document.add_table(rows=1, cols=5)
        for cell, text in zip(
            table.rows[0].cells,
            ["Fecha", "Pelea", "Prob. peleador 1", "Predicción", "Ganador"],
        ):
            cell.text = text
        for _, row in examples.iterrows():
            cells = table.add_row().cells
            cells[0].text = str(row["event_date"].date())
            cells[1].text = f"{row['fighter1']} vs. {row['fighter2']}"
            cells[2].text = f"{row['probability_fighter1']:.1%}"
            cells[3].text = str(row["predicted_winner"])
            cells[4].text = str(row["actual_winner"])
    document.add_heading("Conclusión y límites", level=1)
    document.add_paragraph(
        f"El mejor accuracy en esta prueba fue {metrics['accuracy'].max():.1%}. "
        "Es un resultado de evaluación, no una garantía de acierto. Las "
        "probabilidades también deben comprobarse con calibración antes de "
        "interpretarlas como probabilidades reales."
    )
    document.add_paragraph(
        "Fuente de datos: UFCStats, recopilada por "
        "Greco1899/scrape_ufc_stats. El repositorio indica que actualiza sus "
        "CSV diariamente; este entrenamiento usa la copia local y solo se "
        "actualiza al volver a ejecutar el proceso."
    )
    document.add_paragraph(
        "Referencia: https://github.com/Greco1899/scrape_ufc_stats "
        "(código publicado bajo licencia GPL-3.0)."
    )
    document.save(REPORTS_DIR / "Informe_UFC_Datos_Actualizados.docx")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara datos UFCStats y reentrena modelos cronológicos."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Descarga los CSV más recientes del repositorio antes de entrenar.",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh_source_files()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    events, details, results, stats, fighters = load_source_tables()
    fights, invalid_pairs, missing_dates = prepare_fight_rows(
        events, details, results
    )
    dataset, missing_profiles = create_pre_fight_dataset(
        fights, stats, fighters
    )
    if dataset.empty:
        raise ValueError("No hay peleas utilizables después de limpiar los datos.")

    dates = pd.DatetimeIndex(dataset["event_date"].drop_duplicates().sort_values())
    if len(dates) < 5:
        raise ValueError("Se necesitan al menos cinco fechas distintas.")
    split_position = min(max(int(len(dates) * 0.8), 1), len(dates) - 1)
    split_date = dates[split_position]
    train_mask = dataset["event_date"] < split_date
    test_mask = dataset["event_date"] >= split_date
    X = dataset[MODEL_FEATURES].copy()
    y = dataset["winner_side"].copy()
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "El corte temporal debe incluir victorias de ambos lados en train y test."
        )

    metrics_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}
    prediction_map: dict[str, np.ndarray] = {}
    classification_reports: list[str] = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        fitted_models[name] = model
        prediction_map[name] = predicted
        metrics_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predicted),
                "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
            }
        )
        classification_reports.append(
            f"{name}\n"
            + classification_report(
                y_test,
                predicted,
                labels=[0, 1],
                target_names=["Segundo peleador", "Primer peleador"],
                zero_division=0,
            )
        )

    majority_class = int(y_train.mode().iloc[0])
    baseline = np.full(len(y_test), majority_class)
    metrics_rows.append(
        {
            "model": "Regla: clase más frecuente en train",
            "accuracy": accuracy_score(y_test, baseline),
            "balanced_accuracy": balanced_accuracy_score(y_test, baseline),
        }
    )
    metrics = pd.DataFrame(metrics_rows).sort_values(
        "accuracy", ascending=False
    )
    metrics.to_csv(REPORTS_DIR / "metricas_modelos_ufcstats.csv", index=False)
    (REPORTS_DIR / "classification_report_ufcstats.txt").write_text(
        "\n\n".join(classification_reports), encoding="utf-8"
    )
    dataset.to_csv(
        PROCESSED_DIR / "ufc_features_historicas.csv", index=False
    )

    forest = fitted_models["Random Forest"]
    forest_predictions = prediction_map["Random Forest"]
    forest_probabilities = forest.predict_proba(X_test)
    confusion = confusion_matrix(y_test, forest_predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion, cmap="Blues")
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Segundo", "Primero"],
        yticklabels=["Segundo", "Primero"],
        xlabel="Predicción",
        ylabel="Resultado real",
        title="Matriz de confusión — Random Forest",
    )
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(confusion[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    confusion_path = FIGURES_DIR / "15_matriz_confusion_random_forest_ufcstats.png"
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)
    results_test = dataset.loc[test_mask].copy()
    prediction_output = results_test[
        ["event_date", "event", "bout", "fighter1", "fighter2", "winner_side"]
    ].copy()
    prediction_output["predicted_side"] = forest_predictions
    prediction_output["probability_fighter1"] = forest_probabilities[:, 1]
    prediction_output["probability_fighter2"] = forest_probabilities[:, 0]
    prediction_output.to_csv(
        REPORTS_DIR / "predicciones_prueba_ufcstats.csv", index=False
    )

    examples = prediction_output.copy()
    examples["predicted_winner"] = np.where(
        examples["predicted_side"] == 1,
        examples["fighter1"],
        examples["fighter2"],
    )
    examples["actual_winner"] = np.where(
        examples["winner_side"] == 1,
        examples["fighter1"],
        examples["fighter2"],
    )
    examples["pair_key"] = examples.apply(
        lambda row: frozenset(
            [normalize_name(row["fighter1"]), normalize_name(row["fighter2"])]
        ),
        axis=1,
    )
    known_pairs = [
        ("Conor McGregor", "Donald Cerrone"),
        ("Amanda Nunes", "Irene Aldana"),
        ("Daniel Cormier", "Stipe Miocic"),
        ("Conor McGregor", "Dustin Poirier"),
        ("Khabib Nurmagomedov", "Justin Gaethje"),
        ("Israel Adesanya", "Alex Pereira"),
    ]
    examples_list: list[pd.Series] = []
    used_pairs: set[frozenset[str]] = set()
    for name1, name2 in known_pairs:
        key = frozenset([normalize_name(name1), normalize_name(name2)])
        if key in used_pairs:
            continue
        match = examples.loc[examples["pair_key"] == key]
        if not match.empty:
            examples_list.append(match.iloc[0])
            used_pairs.add(key)
        if len(examples_list) == 3:
            break
    examples_frame = (
        pd.DataFrame(examples_list).drop(columns=["pair_key"], errors="ignore")
        if examples_list
        else pd.DataFrame()
    )
    if not examples_frame.empty:
        examples_frame.to_csv(
            REPORTS_DIR / "ejemplos_historicos_ufcstats.csv", index=False
        )
    else:
        examples_frame = pd.DataFrame(
            columns=[
                "event_date", "fighter1", "fighter2",
                "probability_fighter1", "predicted_winner", "actual_winner",
            ]
        )

    preprocessor = forest.steps[0][1]
    transformed_names = preprocessor.get_feature_names_out()
    feature_importance = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": forest.steps[-1][1].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        REPORTS_DIR / "importancia_variables_ufcstats.csv", index=False
    )

    dump(
        {
            "model": forest,
            "feature_columns": MODEL_FEATURES,
            "target_definition": "1 = first fighter in BOUT won; 0 = second fighter won",
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_end": dataset.loc[train_mask, "event_date"].max(),
            "test_start": dataset.loc[test_mask, "event_date"].min(),
            "test_end": dataset.loc[test_mask, "event_date"].max(),
            "metrics": metrics.to_dict(orient="records"),
            "data_source": "Greco1899/scrape_ufc_stats",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        MODELS_DIR / "random_forest_ufcstats.joblib",
    )
    make_report(
        metrics,
        split_date,
        dataset,
        examples_frame,
        feature_importance,
        confusion_path,
    )

    print(f"Eventos: {len(events):,}; peleas únicas con resultado: {len(results.drop_duplicates('URL')):,}")
    print(f"Peleas con fecha válida y resultado binario: {len(dataset):,}")
    print(f"Fechas sin correspondencia: {missing_dates:,}; parejas mal formadas: {invalid_pairs:,}")
    print(f"Perfiles de peleador no encontrados: {missing_profiles:,}")
    print(
        f"Train: {len(X_train):,} peleas hasta "
        f"{dataset.loc[train_mask, 'event_date'].max().date()}"
    )
    print(
        f"Test: {len(X_test):,} peleas desde "
        f"{dataset.loc[test_mask, 'event_date'].min().date()}"
    )
    print("\nMétricas:")
    print(metrics.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format,
        "balanced_accuracy": "{:.3f}".format,
    }))
    print("\nReporte de clasificación — Random Forest:")
    print(
        classification_report(
            y_test,
            forest_predictions,
            labels=[0, 1],
            target_names=["Segundo peleador", "Primer peleador"],
            zero_division=0,
        )
    )
    print(f"\nDatos procesados: {PROCESSED_DIR / 'ufc_features_historicas.csv'}")
    print(f"Modelo guardado: {MODELS_DIR / 'random_forest_ufcstats.joblib'}")
    print(f"Informe: {REPORTS_DIR / 'Informe_UFC_Datos_Actualizados.docx'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error al preparar o entrenar el modelo: {error}", file=sys.stderr)
        raise
