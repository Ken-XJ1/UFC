"""Aplicación web local para demostrar los modelos UFC entrenados."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "ufc_features_con_cuotas.csv"
WINNER_PREDICTIONS_PATH = (
    PROJECT_DIR / "reports" / "predicciones_prueba_con_cuotas.csv"
)
METHOD_PREDICTIONS_PATH = (
    PROJECT_DIR / "reports" / "predicciones_metodo_victoria.csv"
)
WINNER_MODEL_PATH = PROJECT_DIR / "models" / "random_forest_ufc_odds.joblib"
METHOD_MODEL_PATH = PROJECT_DIR / "models" / "random_forest_ufc_method.joblib"
FINISH_TO_METHOD = {
    "KO/TKO": "KO/TKO",
    "SUB": "Sumisión",
    "U-DEC": "Decisión",
    "S-DEC": "Decisión",
    "M-DEC": "Decisión",
}
STAT_LABELS = {
    "age": "Edad",
    "Height_cms": "Estatura (cm)",
    "Reach_cms": "Alcance (cm)",
    "Weight_lbs": "Peso (lb)",
    "wins": "Victorias previas",
    "losses": "Derrotas previas",
    "draw": "Empates previos",
    "current_win_streak": "Racha actual de victorias",
    "current_lose_streak": "Racha actual de derrotas",
    "longest_win_streak": "Racha máxima de victorias",
    "avg_SIG_STR_landed": "Golpes significativos conectados por pelea",
    "avg_TD_landed": "Derribos conectados por pelea",
    "Stance": "Guardia",
}

st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: #f5f7fb; border: 1px solid #e6e9ef;
        padding: 1rem; border-radius: 12px;
    }
    .notice {
        padding: 1rem 1.1rem; border-radius: 10px;
        background: #fff7e6; border: 1px solid #f0d48a;
        margin: 0.6rem 0 1.2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_fighter(name: object) -> str:
    if pd.isna(name):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(name).lower())).strip()


@st.cache_resource
def load_models() -> tuple[dict, dict]:
    for model_path in (WINNER_MODEL_PATH, METHOD_MODEL_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo {model_path.name}. "
                "Entrena los modelos con `python src\\train_odds_model.py`."
            )
    return joblib.load(WINNER_MODEL_PATH), joblib.load(METHOD_MODEL_PATH)


@st.cache_data
def load_demo_fights() -> pd.DataFrame:
    required_paths = (
        PROCESSED_PATH,
        WINNER_PREDICTIONS_PATH,
        METHOD_PREDICTIONS_PATH,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Faltan archivos de resultados: {names}. "
            "Ejecuta `python src\\train_odds_model.py` antes de abrir la web."
        )

    data = pd.read_csv(PROCESSED_PATH, low_memory=False)
    winner_test = pd.read_csv(
        WINNER_PREDICTIONS_PATH, parse_dates=["event_date"]
    )
    method_test = pd.read_csv(
        METHOD_PREDICTIONS_PATH, parse_dates=["event_date"]
    )
    data["event_date"] = pd.to_datetime(data["event_date"], errors="coerce")

    data = data.copy()
    winner_test = winner_test.copy()
    method_test = method_test.copy()
    for frame in (data, winner_test, method_test):
        frame["red_key"] = frame["R_fighter"].map(normalize_fighter)
        frame["blue_key"] = frame["B_fighter"].map(normalize_fighter)

    keys = ["event_date", "red_key", "blue_key"]
    winner_keys = winner_test[keys].drop_duplicates()
    method_outcomes = method_test[
        keys + ["finish", "target_method_outcome"]
    ].drop_duplicates()
    shared_test_keys = winner_keys.merge(
        method_outcomes, on=keys, how="inner", validate="one_to_one"
    )
    feature_columns = load_models()[0]["feature_columns"]
    data_columns = keys + ["R_fighter", "B_fighter", "Winner"]
    data_columns += [column for column in feature_columns if column not in data_columns]
    demo = data[data_columns].merge(
        shared_test_keys, on=keys, how="inner", validate="one_to_one"
    )
    if demo.empty:
        raise ValueError(
            "No se encontraron peleas comunes entre los modelos y los datos procesados."
        )
    return demo.sort_values(
        ["event_date", "R_fighter", "B_fighter"], ascending=[False, True, True]
    ).reset_index(drop=True)


@st.cache_data
def load_fighter_profiles() -> pd.DataFrame:
    if not PROCESSED_PATH.is_file():
        raise FileNotFoundError(
            "No se encontró el conjunto procesado necesario para crear "
            "perfiles de peleadores."
        )
    feature_columns = load_models()[0]["feature_columns"]
    data = pd.read_csv(PROCESSED_PATH, low_memory=False).copy()
    data["event_date"] = pd.to_datetime(data["event_date"], errors="coerce")

    profiles: list[pd.DataFrame] = []
    for side in ("R", "B"):
        fighter_column = f"{side}_fighter"
        side_features = [
            column
            for column in feature_columns
            if column.startswith(f"{side}_") and column in data.columns
        ]
        side_data = data[
            ["event_date", fighter_column, *side_features]
        ].dropna(subset=["event_date", fighter_column]).copy()
        side_data["fighter_key"] = side_data[fighter_column].map(
            normalize_fighter
        )
        side_data["fighter_name"] = side_data[fighter_column]
        side_data["profile_date"] = side_data["event_date"]
        side_data = side_data.drop(
            columns=["event_date", fighter_column]
        ).rename(
            columns={
                column: column[2:]
                for column in side_features
            }
        )
        profiles.append(side_data)

    all_profiles = pd.concat(profiles, ignore_index=True, sort=False)
    all_profiles = all_profiles.dropna(subset=["fighter_key"])
    all_profiles = all_profiles.sort_values("profile_date")
    return (
        all_profiles.drop_duplicates("fighter_key", keep="last")
        .set_index("fighter_key")
        .sort_index()
    )


def build_matchup_features(
    red_profile: pd.Series,
    blue_profile: pd.Series,
    feature_columns: list[str],
    *,
    weight_class: str,
    title_bout: bool,
    location: str,
    country: str,
    red_odds: float | None,
    blue_odds: float | None,
) -> pd.DataFrame:
    """Construye las entradas de una pelea con perfiles previos de cada atleta."""
    row: dict[str, Any] = {}
    for column in feature_columns:
        if column.startswith("R_"):
            row[column] = red_profile.get(column[2:], np.nan)
        elif column.startswith("B_"):
            row[column] = blue_profile.get(column[2:], np.nan)
        else:
            row[column] = np.nan

    row.update(
        {
            "R_odds": red_odds if red_odds is not None else np.nan,
            "B_odds": blue_odds if blue_odds is not None else np.nan,
            "R_ev": odds_to_ev(red_odds),
            "B_ev": odds_to_ev(blue_odds),
            "location": location,
            "country": country,
            "title_bout": title_bout,
            "weight_class": weight_class,
            "gender": "FEMALE" if weight_class.startswith("Women's") else "MALE",
            "no_of_rounds": 5 if title_bout else 3,
            "empty_arena": False,
        }
    )

    difference_pairs = {
        "lose_streak_dif": ("current_lose_streak", "current_lose_streak"),
        "win_streak_dif": ("current_win_streak", "current_win_streak"),
        "longest_win_streak_dif": ("longest_win_streak", "longest_win_streak"),
        "win_dif": ("wins", "wins"),
        "loss_dif": ("losses", "losses"),
        "total_round_dif": ("total_rounds_fought", "total_rounds_fought"),
        "total_title_bout_dif": ("total_title_bouts", "total_title_bouts"),
        "ko_dif": ("win_by_KO/TKO", "win_by_KO/TKO"),
        "sub_dif": ("win_by_Submission", "win_by_Submission"),
        "height_dif": ("Height_cms", "Height_cms"),
        "reach_dif": ("Reach_cms", "Reach_cms"),
        "age_dif": ("age", "age"),
        "sig_str_dif": ("avg_SIG_STR_landed", "avg_SIG_STR_landed"),
        "avg_sub_att_dif": ("avg_SUB_ATT", "avg_SUB_ATT"),
        "avg_td_dif": ("avg_TD_landed", "avg_TD_landed"),
    }
    for difference, (red_feature, blue_feature) in difference_pairs.items():
        if difference in feature_columns:
            red_value = pd.to_numeric(
                pd.Series([red_profile.get(red_feature)]), errors="coerce"
            ).iloc[0]
            blue_value = pd.to_numeric(
                pd.Series([blue_profile.get(blue_feature)]), errors="coerce"
            ).iloc[0]
            row[difference] = (
                blue_value - red_value
                if pd.notna(red_value) and pd.notna(blue_value)
                else np.nan
            )

    red_rank = rank_for_division(red_profile, weight_class)
    blue_rank = rank_for_division(blue_profile, weight_class)
    if "R_match_weightclass_rank" in feature_columns:
        row["R_match_weightclass_rank"] = red_rank
    if "B_match_weightclass_rank" in feature_columns:
        row["B_match_weightclass_rank"] = blue_rank
    if (
        "better_rank" in feature_columns
        and pd.notna(red_rank)
        and pd.notna(blue_rank)
    ):
        row["better_rank"] = (
            "Red"
            if red_rank < blue_rank
            else "Blue"
            if blue_rank < red_rank
            else np.nan
        )

    return pd.DataFrame([row], columns=feature_columns)


def rank_for_division(profile: pd.Series, weight_class: str) -> float:
    rank_column = f"{weight_class}_rank"
    if rank_column not in profile.index:
        return np.nan
    rank = pd.to_numeric(pd.Series([profile[rank_column]]), errors="coerce").iloc[0]
    return float(rank) if pd.notna(rank) else np.nan


def odds_to_ev(odds: float | None) -> float:
    if odds is None:
        return np.nan
    return 10000 / abs(odds) if odds < 0 else odds


def parse_american_odds(value: str, label: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        odds = float(value)
    except ValueError as error:
        raise ValueError(
            f"La cuota de {label} debe ser un número, por ejemplo -150 o +130."
        ) from error
    if not np.isfinite(odds):
        raise ValueError(f"La cuota de {label} debe ser un número finito.")
    if odds == 0:
        raise ValueError(f"La cuota de {label} no puede ser cero.")
    if odds > 0 and odds < 100:
        raise ValueError(
            f"La cuota positiva de {label} normalmente debe ser 100 o mayor."
        )
    if odds < 0 and odds > -100:
        raise ValueError(
            f"La cuota negativa de {label} normalmente debe ser -100 o menor."
        )
    return odds


def display_stat(value: object, suffix: str = "") -> str:
    if pd.isna(value):
        return "Sin dato"
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return f"{int(value)}{suffix}"
        return f"{value:.1f}{suffix}"
    return str(value)


def fighter_stats(row: pd.Series, side: str) -> pd.Series:
    stats: dict[str, str] = {}
    for field, label in STAT_LABELS.items():
        column = f"{side}_{field}"
        if column in row.index:
            suffix = " cm" if field in {"Height_cms", "Reach_cms"} else ""
            suffix = " lb" if field == "Weight_lbs" else suffix
            stats[label] = display_stat(row[column], suffix)
    return pd.Series(stats, name=row[f"{side}_fighter"])


st.title("🥊 UFC Fight Predictor")
st.caption("Demostración interactiva de los modelos de ganador y método de victoria")

st.markdown(
    """
    <div class="notice">
    <strong>Demostración histórica, no predicción en vivo.</strong>
    Esta aplicación usa peleas reservadas para prueba que ya ocurrieron. La fuente más reciente
    disponible en este proyecto llega al <strong>28 de marzo de 2026</strong>; no incluye
    información actualizada de peleas posteriores.
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    winner_bundle, method_bundle = load_models()
    demo_fights = load_demo_fights()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

winner_model = winner_bundle["model"]
method_model = method_bundle["model"]
winner_features = winner_bundle["feature_columns"]
method_features = method_bundle["feature_columns"]
if winner_features != method_features:
    st.error("Los modelos no tienen la misma definición de variables.")
    st.stop()
feature_columns = winner_features

demo_features = demo_fights.loc[:, feature_columns]
demo_winner_predictions = winner_model.predict(demo_features)
demo_method_predictions = method_model.predict(demo_features)
demo_winner_actual = demo_fights["Winner"].eq("Red").astype(int)
demo_winner_accuracy = demo_winner_actual.eq(demo_winner_predictions).mean()
demo_method_accuracy = demo_fights["target_method_outcome"].eq(
    demo_method_predictions
).mean()

summary_columns = st.columns(4)
summary_columns[0].metric(
    "Accuracy: ganador",
    f"{demo_winner_accuracy:.1%}",
)
summary_columns[1].metric(
    "Accuracy: esquina + método",
    f"{demo_method_accuracy:.1%}",
)
summary_columns[2].metric("Peleas para explorar", f"{len(demo_fights):,}")
summary_columns[3].metric(
    "Última fecha en la fuente",
    f"{demo_fights['event_date'].max():%d %b %Y}",
)
st.caption(
    f"Accuracy calculado por la aplicación para las mismas {len(demo_fights):,} "
    "peleas históricas disponibles en el selector."
)

mode = st.radio(
    "Modo de demostración",
    ["Elegir peleadores", "Explorar pelea histórica"],
    horizontal=True,
)
is_historical = mode == "Explorar pelea histórica"

if is_historical:
    options = {
        (
            f"{row.event_date:%Y-%m-%d}  |  "
            f"Roja: {row.R_fighter}  vs  Azul: {row.B_fighter}"
        ): index
        for index, row in demo_fights.iterrows()
    }
    selection = st.selectbox(
        "Elige una pelea histórica del conjunto de prueba",
        options=list(options),
        help=(
            "Son combates posteriores al periodo de entrenamiento. "
            "Los modelos calcularán su predicción al seleccionar una pelea."
        ),
    )
    fight = demo_fights.iloc[options[selection]].copy()
else:
    try:
        profiles = load_fighter_profiles()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    profile_names = profiles["fighter_name"].astype(str).tolist()
    red_default = (
        profile_names.index("Alexa Grasso")
        if "Alexa Grasso" in profile_names
        else 0
    )
    blue_default_name = "Maycee Barber"
    red_col, blue_col = st.columns(2)
    with red_col:
        red_name = st.selectbox(
            "Esquina roja · busca un peleador",
            profile_names,
            index=red_default,
            key="custom_red_fighter",
            help="Empieza a escribir para filtrar los nombres disponibles.",
        )
    red_key = normalize_fighter(red_name)
    blue_names = [
        name for name in profile_names if normalize_fighter(name) != red_key
    ]
    blue_default = (
        blue_names.index(blue_default_name)
        if blue_default_name in blue_names
        else 0
    )
    with blue_col:
        blue_name = st.selectbox(
            "Esquina azul · busca un peleador",
            blue_names,
            index=blue_default,
            key="custom_blue_fighter",
            help="Empieza a escribir para filtrar los nombres disponibles.",
        )

    weight_classes = sorted(
        str(value)
        for value in demo_fights["weight_class"].dropna().unique()
    )
    preferred_class = (
        "Lightweight" if "Lightweight" in weight_classes else weight_classes[0]
    )
    context_col, venue_col = st.columns(2)
    with context_col:
        weight_class = st.selectbox(
            "División del combate",
            weight_classes,
            index=weight_classes.index(preferred_class),
        )
        title_bout = st.checkbox("Combate por título (5 asaltos)")
    location_countries = (
        demo_fights.dropna(subset=["location", "country"])
        .groupby("location")["country"]
        .agg(lambda values: values.mode().iloc[0])
        .to_dict()
    )
    locations = sorted(location_countries)
    default_location = (
        "Las Vegas, Nevada"
        if "Las Vegas, Nevada" in locations
        else locations[0]
    )
    with venue_col:
        location = st.selectbox(
            "Ubicación (contexto)",
            locations,
            index=locations.index(default_location),
        )
        st.caption(f"País/región registrado: {location_countries[location]}")

    st.markdown("**Cuotas americanas (opcional)**")
    st.caption(
        "Escribe, por ejemplo, -150 para favorito o +130 para no favorito. "
        "Déjalas vacías si no tienes cuotas."
    )
    red_odds_text, blue_odds_text = st.columns(2)
    with red_odds_text:
        red_odds_input = st.text_input(
            f"Cuota roja · {red_name}", placeholder="-150 o +130"
        )
    with blue_odds_text:
        blue_odds_input = st.text_input(
            f"Cuota azul · {blue_name}", placeholder="+130 o -150"
        )
    try:
        red_odds = parse_american_odds(red_odds_input, red_name)
        blue_odds = parse_american_odds(blue_odds_input, blue_name)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    red_profile = profiles.loc[red_key]
    blue_profile = profiles.loc[normalize_fighter(blue_name)]
    matchup_features = build_matchup_features(
        red_profile,
        blue_profile,
        feature_columns,
        weight_class=weight_class,
        title_bout=title_bout,
        location=location,
        country=location_countries[location],
        red_odds=red_odds,
        blue_odds=blue_odds,
    )
    fight = matchup_features.iloc[0].copy()
    fight["R_fighter"] = red_name
    fight["B_fighter"] = blue_name
    fight["R_profile_date"] = red_profile["profile_date"]
    fight["B_profile_date"] = blue_profile["profile_date"]

    st.caption(
        f"Perfiles deportivos disponibles hasta: "
        f"roja {pd.Timestamp(red_profile['profile_date']):%d-%m-%Y}; "
        f"azul {pd.Timestamp(blue_profile['profile_date']):%d-%m-%Y}. "
        "No son necesariamente las estadísticas actuales."
    )
    if red_odds is None or blue_odds is None:
        st.info(
            "Falta al menos una cuota. El modelo fue entrenado con cuotas; "
            "los valores vacíos se imputarán con valores típicos del entrenamiento. "
            "Esto no equivale al modelo sin cuotas."
        )

red_name = fight["R_fighter"]
blue_name = fight["B_fighter"]
st.subheader(f"{red_name}  vs  {blue_name}")
if is_historical:
    st.write(
        f"**Fecha:** {fight['event_date']:%d-%m-%Y}  ·  "
        f"**Esquina roja:** {red_name}  ·  **Esquina azul:** {blue_name}"
    )
else:
    st.write(
        f"**División:** {fight['weight_class']}  ·  "
        f"**Esquina roja:** {red_name}  ·  **Esquina azul:** {blue_name}"
    )

input_row = fight.loc[feature_columns].to_frame().T
winner_probabilities = winner_model.predict_proba(input_row)[0]
winner_classes = list(winner_model.steps[-1][1].classes_)
red_class_index = winner_classes.index(1)
red_probability = float(winner_probabilities[red_class_index])
blue_probability = 1.0 - red_probability
predicted_side = "Red" if red_probability >= 0.5 else "Blue"
predicted_fighter = red_name if predicted_side == "Red" else blue_name

st.markdown("### ¿A quién eligió el modelo?")
winner_columns = st.columns([1, 2, 1])
winner_columns[0].metric(f"Esquina roja · {red_name}", f"{red_probability:.1%}")
winner_columns[1].progress(
    red_probability,
    text=f"Probabilidad estimada para {red_name} (rojo)",
)
winner_columns[2].metric(f"Esquina azul · {blue_name}", f"{blue_probability:.1%}")
st.success(f"Predicción del modelo: gana **{predicted_fighter}** ({'roja' if predicted_side == 'Red' else 'azul'}).")

method_probabilities = method_model.predict_proba(input_row)[0]
method_classes = list(method_model.steps[-1][1].classes_)
method_table = pd.DataFrame(
    {
        "Resultado posible": method_classes,
        "Probabilidad estimada": method_probabilities,
    }
).sort_values("Probabilidad estimada", ascending=False)
method_table["Método"] = method_table["Resultado posible"].str.split(" — ").str[-1]
method_totals = (
    method_table.groupby("Método", as_index=False)["Probabilidad estimada"]
    .sum()
    .sort_values("Probabilidad estimada", ascending=False)
)
method_name = method_table.iloc[0]["Resultado posible"]

st.markdown("### ¿Cómo podría ganar?")
st.info(
    f"El resultado combinado más probable es **{method_name}** "
    f"({method_table.iloc[0]['Probabilidad estimada']:.1%} según este modelo). "
    "No es una garantía y el modelo de método tiene menor precisión."
)
method_chart, outcome_chart = st.columns(2)
with method_chart:
    st.write("**Método, sumando ambas esquinas**")
    st.bar_chart(
        method_totals.set_index("Método")[["Probabilidad estimada"]],
        horizontal=True,
    )
with outcome_chart:
    st.write("**Seis resultados posibles**")
    st.dataframe(
        method_table[["Resultado posible", "Probabilidad estimada"]].style.format(
            {"Probabilidad estimada": "{:.1%}"}
        ),
        width="stretch",
        hide_index=True,
    )

with st.expander("Ver comparación de estadísticas previas de ambos peleadores"):
    red_stats = fighter_stats(fight, "R")
    blue_stats = fighter_stats(fight, "B")
    comparison = pd.concat([red_stats, blue_stats], axis=1)
    comparison.columns = [f"Roja · {red_name}", f"Azul · {blue_name}"]
    st.dataframe(comparison, width="stretch")
    if is_historical:
        st.caption(
            "Son las estadísticas históricas guardadas para esa pelea, no el estado "
            "actual de los atletas. Las cuotas son una señal del mercado, no una certeza."
        )
    else:
        st.caption(
            "Se usa para cada atleta su aparición más reciente en la fuente disponible. "
            "Los registros pueden ser antiguos y no reflejan lesiones ni cambios recientes."
        )
    odds_columns = st.columns(2)
    odds_columns[0].metric(
        f"Cuota roja · {red_name}", display_stat(fight.get("R_odds"))
    )
    odds_columns[1].metric(
        f"Cuota azul · {blue_name}", display_stat(fight.get("B_odds"))
    )

if is_historical:
    with st.expander("Revelar el resultado real y comprobar la predicción"):
        actual_side = fight["Winner"]
        actual_fighter = red_name if actual_side == "Red" else blue_name
        actual_method = fight.get("finish", "Sin dato")
        actual_method_group = FINISH_TO_METHOD.get(str(actual_method).upper().strip())
        actual_method_class = (
            f"Esquina {'roja' if actual_side == 'Red' else 'azul'} — {actual_method_group}"
            if actual_method_group
            else None
        )
        st.write(
            f"**Resultado registrado:** {actual_fighter} "
            f"({'roja' if actual_side == 'Red' else 'azul'}) · {actual_method}."
        )
        winner_hit = predicted_side == actual_side
        method_hit = method_name == actual_method_class
        st.write(
            f"Predicción de ganador: **{'correcta' if winner_hit else 'incorrecta'}**.  \n"
            f"Predicción exacta de esquina + método: "
            f"**{'correcta' if method_hit else 'incorrecta'}**."
        )

st.divider()
st.caption(
    "Proyecto educativo. Los resultados son históricos y experimentales; "
    "las probabilidades no están calibradas, no garantizan resultados y no son consejo de apuestas."
)
