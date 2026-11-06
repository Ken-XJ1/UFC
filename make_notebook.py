import json
from pathlib import Path

cells = []


def md(src):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip().splitlines(keepends=True),
    })


def code(src):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.strip().splitlines(keepends=True),
    })


md("""# UFC Fight Predictor — Informe reproducible del proyecto

Este notebook resume de forma ejecutable todo el proyecto: las fuentes de datos, la
ingeniería de variables, el entrenamiento y evaluación de los modelos, y un ejemplo
real de predicción (Tom Nolan vs. Renato Moicano) generado con el mismo pipeline que
usa la aplicación web.

> **Aviso:** las predicciones son estimaciones experimentales sobre datos históricos.
> No garantizan resultados ni constituyen consejo de apuestas.""")

md("""## 1. Objetivo

Responder dos preguntas de clasificación sobre peleas de UFC:

1. **¿Qué esquina ganará?** — modelo binario (esquina roja vs. azul).
2. **¿Cómo terminará?** — modelo de seis clases que combina esquina y método
   (KO/TKO, sumisión o decisión).

La evaluación se hace de forma **cronológica**: entrenar con peleas antiguas y probar
con las más recientes, evitando la fuga de información.""")

md("""## 2. Datos""")
code("""from pathlib import Path
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

_cwd = Path.cwd()
PROJECT_DIR = _cwd if (_cwd / "app.py").exists() else _cwd.parent
if not (PROJECT_DIR / "app.py").exists():
    raise FileNotFoundError("Ejecuta el notebook desde la carpeta ufc-predictor")
PROCESSED = PROJECT_DIR / "data" / "processed" / "ufc_features_con_cuotas.csv"
REPORTS = PROJECT_DIR / "reports"

features = pd.read_csv(PROCESSED, low_memory=False)
print("Filas (peleas):", len(features))
print("Columnas:", len(features.columns))
features[["event_date", "R_fighter", "B_fighter", "weight_class", "R_odds", "B_odds"]].head()""")

code("""features["weight_class"].value_counts().head(10)""")

md("""## 3. Variables predictivas

Cada esquina aporta su perfil histórico más reciente (estadísticas acumuladas antes
de la pelea) y el pipeline calcula **diferencias** entre ambos peleadores, además de
variables del evento: división, título, ubicación, país, número de asaltos, género y
cuotas americanas.""")

code("""# Cargar la lista de variables que usan los modelos entrenados
import joblib
winner_bundle = joblib.load(PROJECT_DIR / "models" / "random_forest_ufc_odds.joblib")
feature_columns = winner_bundle["feature_columns"]
print("Número de variables del modelo:", len(feature_columns))
print("\\nEjemplos de variables de diferencia:")
print([c for c in feature_columns if c.endswith("_dif")][:10])""")

md("""## 4. Evaluación de los modelos

Las métricas se leen de los reportes generados durante el entrenamiento.""")

code("""metrics = pd.read_csv(REPORTS / "metricas_modelos_con_cuotas.csv")
metrics""")

code("""preds = pd.read_csv(REPORTS / "predicciones_prueba_con_cuotas.csv")
print(preds.columns.tolist())
preds.head()""")

code("""from sklearn.metrics import accuracy_score, balanced_accuracy_score

accuracy_rf = accuracy_score(preds["target_red"], preds["predicted_red"])
bacc_rf = balanced_accuracy_score(preds["target_red"], preds["predicted_red"])
print(f"Random Forest con cuotas — accuracy: {accuracy_rf:.1%}")
print(f"Random Forest con cuotas — balanced accuracy: {bacc_rf:.1%}")
print(f"Peleas de prueba: {len(preds)}")""")

md("""## 5. Importancia de variables

Qué tanto contribuye cada variable al modelo de ganador.""")

code("""import matplotlib.pyplot as plt

importance_path = REPORTS / "importancia_variables_con_cuotas.csv"
imp = pd.read_csv(importance_path)
print(imp.columns.tolist())
imp.head()""")

code("""# Top 15 variables más importantes
col_name, val_col = "feature", "importance"
top = imp.nlargest(15, val_col)
plt.figure(figsize=(9, 6))
plt.barh(top[col_name][::-1], top[val_col][::-1], color="#e10600")
plt.title("Importancia de variables — modelo ganador (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()""")

md("""## 6. Ejemplo práctico: Tom Nolan vs. Renato Moicano

Reproducimos la pelea estelar de UFC Fight Night del 31 de octubre de 2026
(Meta Apex, Las Vegas, peso ligero, sin título) usando **exactamente las mismas
funciones que usa la aplicación web** (`app.py`):

- Cuotas ingresadas: **-111** (Nolan) y **-115** (Moicano), formato americano.
- Perfiles: última aparición de cada atleta en el dataset histórico.""")

code("""import sys
sys.path.insert(0, str(PROJECT_DIR))
import app  # en modo "bare" Streamlit solo emite warnings, no bloquea

red_profile = app.load_fighter_profiles().loc[app.normalize_fighter("Tom Nolan")]
blue_profile = app.load_fighter_profiles().loc[app.normalize_fighter("Renato Moicano")]

X = app.build_matchup_features(
    red_profile,
    blue_profile,
    app.feature_columns,
    weight_class="Lightweight",
    title_bout=False,
    location="Las Vegas, Nevada",
    country="USA",
    red_odds=-111.0,
    blue_odds=-115.0,
)

# Modelo de ganador
win_probs = app.winner_model.predict_proba(X[app.feature_columns])[0]
win_classes = list(app.winner_model.steps[-1][1].classes_)
red_win = float(win_probs[win_classes.index(1)])
print(f"Tom Nolan (rojo):  {red_win:.1%}")
print(f"Renato Moicano (azul): {1 - red_win:.1%}")""")

code("""# Modelo de esquina + método de victoria
method_probs = app.method_model.predict_proba(X[app.feature_columns])[0]
method_classes = list(app.method_model.steps[-1][1].classes_)
method_df = pd.DataFrame({
    "Resultado": method_classes,
    "Probabilidad": method_probs,
}).sort_values("Probabilidad", ascending=False)
method_df["Probabilidad"] = method_df["Probabilidad"].map(lambda v: f"{v:.1%}")
method_df""")

code("""# Gráfica de probabilidades de ganador
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(["Tom Nolan", "Renato Moicano"], [red_win, 1 - red_win], color=["#e10600", "#3a3a44"])
ax.set_ylim(0, 1)
ax.set_ylabel("Probabilidad")
ax.set_title("Predicción del modelo — ganador")
for i, v in enumerate([red_win, 1 - red_win]):
    ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""## 7. La aplicación web

La misma predicción está disponible de forma interactiva en `app.py` (Streamlit):

- **Próximas peleas:** consulta la API de UFCalendar y muestra eventos, divisiones,
  récords, horarios por sección en hora de Colombia y cuotas de consenso.
- **Pelea personalizada:** asistente para elegir dos atletas, división, título,
  ubicación y cuotas; devuelve las mismas probabilidades de este notebook.

Para ejecutarla:

```bash
python -m streamlit run app.py
```

## 8. Limitaciones

- Los datos con cuotas llegan hasta el **28 de marzo de 2026**; el modelo no conoce
  peleas ni cambios posteriores.
- Los perfiles personalizados usan la última aparición histórica del atleta y pueden
  estar desactualizados (lesiones, rachas reales, etc.).
- Las cuotas faltantes se imputan con valores típicos del entrenamiento; omitirlas
  no equivale al modelo sin cuotas.
- Las probabilidades **no están calibradas** y el modelo de método tiene menor
  precisión (36,7 % de accuracy en seis clases).
- Todo lo anterior implica que las salidas son estimaciones educativas, no predicciones
  garantizadas.""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/03_informe_reproducible_proyecto.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook guardado:", out)
