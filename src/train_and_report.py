"""Entrena modelos cronológicos de UFC y genera un informe DOCX reproducible."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR / "data" / "raw" / "data.csv"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURES_DIR = REPORT_DIR / "figures"
FEATURE_PAIRS = {
    "edad": ("B_Age", "R_Age"),
    "estatura": ("B_Height", "R_Height"),
    "peso": ("B_Weight", "R_Weight"),
    "peleas_previas": ("BPrev", "RPrev"),
    "racha_previa": ("BStreak", "RStreak"),
}
EXAMPLE_MATCHUPS = {
    ("Eddie Alvarez", "Conor McGregor"),
    ("Amanda Nunes", "Ronda Rousey"),
    ("Daniel Cormier", "Anthony Johnson"),
}
RANDOM_STATE = 42


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construye variables únicamente a partir de columnas pre-pelea verificadas."""
    features = pd.DataFrame(index=frame.index)
    for name, (blue_column, red_column) in FEATURE_PAIRS.items():
        if blue_column not in frame or red_column not in frame:
            continue
        blue = pd.to_numeric(frame[blue_column], errors="coerce")
        red = pd.to_numeric(frame[red_column], errors="coerce")
        features[f"azul_{name}"] = blue
        features[f"rojo_{name}"] = red
        features[f"diferencia_{name}_azul_menos_rojo"] = blue - red
    if features.empty:
        raise ValueError("El CSV no contiene pares reconocidos de variables pre-pelea.")
    return features


def make_models() -> dict[str, object]:
    """Devuelve los tres modelos y su preprocesamiento."""
    return {
        "Regresión logística": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "XGBoost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    }


def style_table(table) -> None:
    """Aplica un formato legible a una tabla de Word."""
    table.style = "Light Shading Accent 1"
    for cell in table.rows[0].cells:
        for run in cell.paragraphs[0].runs:
            run.bold = True


def add_figure(document: Document, path: Path, caption: str) -> None:
    document.add_picture(str(path), width=Inches(6.1))
    paragraph = document.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in paragraph.runs:
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(90, 90, 90)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {RAW_PATH}")

    raw = pd.read_csv(RAW_PATH)
    if not {"Date", "winner"}.issubset(raw.columns):
        raise KeyError("El CSV debe tener las columnas Date y winner.")
    frame = raw.drop_duplicates().copy()
    frame["Date"] = pd.to_datetime(
        frame["Date"], format="%m/%d/%Y", errors="coerce"
    )
    if frame["Date"].isna().any():
        other_dates = pd.to_datetime(raw.loc[frame.index, "Date"], errors="coerce")
        frame["Date"] = frame["Date"].fillna(other_dates)

    features = build_features(frame)
    outcome = frame["winner"].astype("string").str.strip().str.lower()
    usable = outcome.isin(["red", "blue"]) & frame["Date"].notna()
    X = features.loc[usable].copy()
    y = outcome.loc[usable].map({"red": 1, "blue": 0}).astype("int64")
    dates = frame.loc[usable, "Date"]

    order = dates.sort_values(kind="mergesort").index
    X, y, dates = X.loc[order], y.loc[order], dates.loc[order]
    if len(X) < 5:
        raise ValueError("Se necesitan al menos cinco peleas válidas para entrenar.")
    split = min(max(int(len(X) * 0.8), 1), len(X) - 1)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    test_dates = dates.iloc[split:]

    models = make_models()
    scores: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        predictions[name] = estimator.predict(X_test)
        scores[name] = accuracy_score(y_test, predictions[name])

    rf = models["Random Forest"]
    rf_predictions = predictions["Random Forest"]
    rf_probabilities = rf.predict_proba(X_test)
    baseline_accuracy = float((y_test == 1).mean())
    report = classification_report(
        y_test,
        rf_predictions,
        labels=[0, 1],
        target_names=["Azul", "Rojo"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, rf_predictions, labels=[0, 1])

    examples: list[dict[str, object]] = []
    test_indices = X_test.index
    for position, row_index in enumerate(test_indices):
        red_name = str(frame.loc[row_index, "R_Name"]) if "R_Name" in frame else ""
        blue_name = str(frame.loc[row_index, "B_Name"]) if "B_Name" in frame else ""
        if (red_name, blue_name) not in EXAMPLE_MATCHUPS:
            continue
        examples.append(
            {
                "date": dates.loc[row_index],
                "red": red_name,
                "blue": blue_name,
                "actual": str(outcome.loc[row_index]).upper(),
                "predicted": "RED" if rf_predictions[position] == 1 else "BLUE",
                "p_red": float(rf_probabilities[position, 1]),
                "p_blue": float(rf_probabilities[position, 0]),
            }
        )
    if len(examples) < 3:
        raise ValueError(
            "No se encontraron los tres enfrentamientos de ejemplo en el tramo de prueba."
        )

    dump(
        {
            "model": rf,
            "feature_columns": list(X.columns),
            "target_encoding": {"blue": 0, "red": 1},
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_end": dates.iloc[split - 1],
            "test_start": dates.iloc[split],
            "test_end": dates.iloc[-1],
            "test_accuracy": scores["Random Forest"],
        },
        MODEL_DIR / "random_forest_ufc.joblib",
    )

    test_predictions = pd.DataFrame(
        {
            "Date": test_dates.to_numpy(),
            "y_real": y_test.to_numpy(),
            "y_predicho": rf_predictions,
            "probabilidad_azul": rf_probabilities[:, 0],
            "probabilidad_rojo": rf_probabilities[:, 1],
        },
        index=test_indices,
    )
    test_predictions.to_csv(REPORT_DIR / "predicciones_prueba.csv", index_label="fila_csv")
    pd.DataFrame(examples).to_csv(
        REPORT_DIR / "ejemplos_historicos.csv", index=False
    )

    sns.set_theme(style="whitegrid")
    outcome_counts = outcome.value_counts().reindex(
        ["red", "blue", "draw", "no contest"], fill_value=0
    )
    plt.figure(figsize=(7, 4.5))
    sns.barplot(
        x=outcome_counts.index,
        y=outcome_counts.values,
        hue=outcome_counts.index,
        palette={
            "red": "#E45756",
            "blue": "#4C78A8",
            "draw": "#72B7B2",
            "no contest": "#B279A2",
        },
        legend=False,
    )
    plt.xlabel("Resultado registrado")
    plt.ylabel("Número de combates")
    plt.title("Distribución de resultados en el CSV")
    plt.tight_layout()
    outcomes_path = FIGURES_DIR / "11_distribucion_resultados_informe.png"
    plt.savefig(outcomes_path, dpi=160)
    plt.close()

    eda_specs = [
        ("edad", "Edad", ("B_Age", "R_Age")),
        ("estatura", "Estatura (cm)", ("B_Height", "R_Height")),
    ]
    eda_paths: dict[str, Path] = {}
    for key, label, columns in eda_specs:
        available = [column for column in columns if column in frame]
        if not available:
            continue
        values = frame[available].melt(
            var_name="esquina", value_name="valor"
        )
        values["valor"] = pd.to_numeric(values["valor"], errors="coerce")
        values["esquina"] = values["esquina"].map(
            {"R_Age": "Rojo", "B_Age": "Azul",
             "R_Height": "Rojo", "B_Height": "Azul"}
        )
        values = values.dropna(subset=["valor"])
        plt.figure(figsize=(7, 4.5))
        sns.histplot(
            data=values,
            x="valor",
            hue="esquina",
            kde=True,
            element="step",
            palette={"Rojo": "#E45756", "Azul": "#4C78A8"},
        )
        plt.xlabel(label)
        plt.ylabel("Frecuencia")
        plt.title(f"Distribución de {label.lower()} por esquina")
        plt.tight_layout()
        figure_path = FIGURES_DIR / f"12_distribucion_{key}_informe.png"
        plt.savefig(figure_path, dpi=160)
        plt.close()
        eda_paths[key] = figure_path

    correlation_columns = [
        column
        for column in [
            "B_Age", "R_Age", "B_Height", "R_Height", "B_Weight",
            "R_Weight", "BPrev", "RPrev", "BStreak", "RStreak",
        ]
        if column in frame
    ]
    correlation_path = None
    if len(correlation_columns) >= 2:
        correlation_data = frame[correlation_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        plt.figure(figsize=(8, 6.5))
        sns.heatmap(
            correlation_data.corr(),
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            square=True,
        )
        plt.title("Correlación entre atributos pre-pelea")
        plt.tight_layout()
        correlation_path = FIGURES_DIR / "13_correlaciones_pre_pelea_informe.png"
        plt.savefig(correlation_path, dpi=160)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    names = list(scores)
    values = [scores[name] * 100 for name in names]
    values.append(baseline_accuracy * 100)
    names.append("Siempre elegir rojo")
    colors = ["#4C78A8", "#F58518", "#54A24B", "#999999"]
    bars = plt.bar(names, values, color=colors)
    plt.ylim(0, 75)
    plt.ylabel("Accuracy (%)")
    plt.title("Modelos frente a una referencia sencilla")
    plt.xticks(rotation=12, ha="right")
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
        )
    plt.tight_layout()
    comparison_path = FIGURES_DIR / "07_comparacion_modelos.png"
    plt.savefig(comparison_path, dpi=160)
    plt.close()

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Azul", "Rojo"],
        yticklabels=["Azul", "Rojo"],
    )
    plt.xlabel("Predicción del modelo")
    plt.ylabel("Resultado real")
    plt.title("Matriz de confusión — Random Forest")
    plt.tight_layout()
    confusion_path = FIGURES_DIR / "08_matriz_confusion_informe.png"
    plt.savefig(confusion_path, dpi=160)
    plt.close()

    forest = rf.steps[-1][1]
    importance = pd.Series(
        forest.feature_importances_, index=X.columns
    ).sort_values().tail(10)
    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="#4C78A8")
    plt.xlabel("Importancia del modelo")
    plt.title("Diez variables más usadas por Random Forest")
    plt.tight_layout()
    importance_path = FIGURES_DIR / "09_importancia_random_forest_informe.png"
    plt.savefig(importance_path, dpi=160)
    plt.close()

    pie_paths: dict[tuple[str, str], Path] = {}
    for example in examples:
        red_name = str(example["red"])
        blue_name = str(example["blue"])
        actual_winner = {"RED": "rojo", "BLUE": "azul"}.get(
            str(example["actual"]), str(example["actual"]).lower()
        )
        file_name = (
            f"10_probabilidades_{red_name}_{blue_name}"
            .lower()
            .replace(" ", "_")
            .replace(".", "")
        )
        pie_path = FIGURES_DIR / f"{file_name}.png"
        figure, axis = plt.subplots(figsize=(6, 4.8))
        wedges, _ = axis.pie(
            [example["p_red"], example["p_blue"]],
            colors=["#E45756", "#4C78A8"],
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        axis.legend(
            wedges,
            [
                f"{red_name} (rojo): {example['p_red']:.1%}",
                f"{blue_name} (azul): {example['p_blue']:.1%}",
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=1,
            frameon=False,
        )
        axis.set_title(
            f"{red_name} vs. {blue_name}\n"
            f"Resultado real: ganó {actual_winner}"
        )
        axis.axis("equal")
        figure.tight_layout()
        figure.savefig(pie_path, dpi=160, bbox_inches="tight")
        plt.close(figure)
        pie_paths[(red_name, blue_name)] = pie_path

    document = Document()
    document.core_properties.title = "Avance: predicción de peleas de UFC"
    document.core_properties.subject = "Avance de proyecto de ciencia de datos"
    document.add_heading("Predicción de peleas de UFC", 0)
    subtitle = document.add_paragraph("Avance del proyecto")
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_heading("Objetivo", level=1)
    document.add_paragraph(
        "Probar si los datos disponibles antes de una pelea ayudan a estimar "
        "qué esquina gana: roja o azul."
    )
    document.add_heading("Datos y método", level=1)
    document.add_paragraph(
        f"El CSV contiene {len(raw):,} peleas, con fechas entre "
        f"{dates.iloc[0].date()} y {dates.iloc[-1].date()}. Para el modelo "
        f"quedaron {len(X):,} peleas con ganador rojo o azul. Se usaron edad, "
        "estatura, peso, peleas previas y racha de ambos luchadores. Los "
        "empates y resultados sin combate se excluyeron."
    )
    document.add_paragraph(
        f"Se entrenó con {len(X_train):,} peleas anteriores y se evaluó con "
        f"{len(X_test):,} más recientes (desde {dates.iloc[split].date()}). "
        "Se compararon regresión logística, Random Forest y XGBoost. Las "
        "estadísticas generadas durante la pelea no se usaron como entrada."
    )

    document.add_heading("Resultados preliminares", level=1)
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Método"
    table.rows[0].cells[1].text = "Aciertos en prueba"
    for name, score in scores.items():
        cells = table.add_row().cells
        cells[0].text = name
        cells[1].text = f"{score:.1%}"
    cells = table.add_row().cells
    cells[0].text = "Elegir siempre rojo"
    cells[1].text = f"{baseline_accuracy:.1%}"
    style_table(table)
    document.add_paragraph(
        f"Random Forest obtuvo el resultado más alto ({scores['Random Forest']:.1%}), "
        f"pero apenas superó la referencia de elegir rojo siempre "
        f"({baseline_accuracy:.1%}). En esta prueba, identificó "
        f"{report['Azul']['recall']:.1%} de las victorias azules y "
        f"{report['Rojo']['recall']:.1%} de las rojas. El resultado es modesto: "
        "el modelo todavía no es confiable para predecir peleas futuras."
    )
    add_figure(
        document,
        comparison_path,
        "Comparación de aciertos en el conjunto de prueba.",
    )
    add_figure(
        document,
        confusion_path,
        "Random Forest: resultado real frente a predicción.",
    )

    document.add_heading("Ejemplos de peleas históricas", level=1)
    document.add_paragraph(
        "Estas peleas estaban en el grupo de prueba. Los porcentajes son "
        "estimaciones del modelo con los datos previos disponibles en el CSV."
    )
    table = document.add_table(rows=1, cols=4)
    for cell, header in zip(
        table.rows[0].cells,
        ["Pelea (rojo vs. azul)", "Probabilidad", "Predijo", "Ganador"],
    ):
        cell.text = header
    style_table(table)
    for example in examples:
        cells = table.add_row().cells
        cells[0].text = f"{example['red']} vs. {example['blue']}"
        cells[1].text = (
            f"Rojo {example['p_red']:.1%} / Azul {example['p_blue']:.1%}"
        )
        cells[2].text = {"RED": "Rojo", "BLUE": "Azul"}[
            str(example["predicted"])
        ]
        cells[3].text = {"RED": "Rojo", "BLUE": "Azul"}[
            str(example["actual"])
        ]
        matchup = (str(example["red"]), str(example["blue"]))
        add_figure(
            document,
            pie_paths[matchup],
            f"{example['red']} (rojo) vs. {example['blue']} (azul), "
            f"{example['date'].date()}.",
        )

    document.add_heading("Límites y siguiente paso", level=1)
    document.add_paragraph(
        "Los datos disponibles cubren 2014–2017 y no incluyen alcance ni "
        "categoría de peso. Los tres ejemplos son combates pasados, no "
        "pronósticos de peleas actuales. Como siguiente paso se propone buscar "
        "estadísticas históricas más completas y volver a evaluar el modelo."
    )
    document.add_paragraph(
        "Referencia de estilo para el análisis exploratorio: Rish Pande, "
        "“UFC: Data Analysis & Visualization (beginner)”, Kaggle. "
        "https://www.kaggle.com/code/rishpande/ufc-data-analysis-visualization-beginner"
    )
    document.add_paragraph(
        "Avance elaborado a partir del CSV disponible y los modelos evaluados "
        "en este proyecto."
    )
    output_path = REPORT_DIR / "Informe_UFC_Prediccion.docx"
    document.save(output_path)

    print(f"Accuracy — regresión logística: {scores['Regresión logística']:.3f}")
    print(f"Accuracy — Random Forest: {scores['Random Forest']:.3f}")
    print(f"Accuracy — XGBoost: {scores['XGBoost']:.3f}")
    print(f"Accuracy — referencia rojo: {baseline_accuracy:.3f}")
    print(f"Fecha de inicio de prueba: {dates.iloc[split].date()}")
    print(f"Modelo guardado: {MODEL_DIR / 'random_forest_ufc.joblib'}")
    print(f"Informe guardado: {output_path}")
    print(f"Ejemplos: {REPORT_DIR / 'ejemplos_historicos.csv'}")


if __name__ == "__main__":
    main()
