"""Aplicación web local para demostrar los modelos UFC entrenados."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = PROJECT_DIR / "data" / "processed" / "ufc_features_con_cuotas.csv"

# Cargar variables de entorno desde .env local (nunca se sube al repo)
_env_path = PROJECT_DIR / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())
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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html { color-scheme: dark; }

    /* ===== Tema UFC: negro, rojo y dorado ===== */
    /* Ocultar la barra superior clara de Streamlit */
    header[data-testid="stHeader"],
    [data-testid="stHeader"],
    #MainMenu,
    footer { display: none !important; }

    .stApp {
        background-color: #0d0d10;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cg fill='none' stroke='%23ffffff' stroke-opacity='0.11' stroke-width='1.6'%3E%3Cpath d='M60 38l15.6 9v18L60 74l-15.6-9v-18z'/%3E%3Ccircle cx='0' cy='0' r='3' fill='%23ffffff' fill-opacity='0.11' stroke='none'/%3E%3Ccircle cx='120' cy='0' r='3' fill='%23ffffff' fill-opacity='0.11' stroke='none'/%3E%3Ccircle cx='0' cy='120' r='3' fill='%23ffffff' fill-opacity='0.11' stroke='none'/%3E%3Ccircle cx='120' cy='120' r='3' fill='%23ffffff' fill-opacity='0.11' stroke='none'/%3E%3C/g%3E%3C/svg%3E");
        color: #f2f2f2;
        font-family: 'Inter', sans-serif;
    }
    .block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}

    /* Título principal estilo UFC */
    h1 {
        font-family: 'Oswald', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase;
        background: linear-gradient(90deg, #e10600 0%, #ff2a1a 55%, #ffb199 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    h2, h3 {
        color: #f5f5f5 !important;
        letter-spacing: 0.5px;
        font-family: 'Oswald', sans-serif !important;
        text-transform: uppercase;
    }

    /* Texto general legible sobre fondo oscuro */
    .stMarkdown p, .stMarkdown li, label, .stCaption {
        color: #d6d6dd;
    }
    .stCaption { color: #8a8a93 !important; }

    /* Sidebar oscura por si aparece */
    section[data-testid="stSidebar"] {
        background: #141417;
    }
    section[data-testid="stSidebar"] * {
        color: #d6d6dd;
    }

    /* Línea divisoria estilo UFC */
    hr { border: none; border-top: 2px solid #2c2c33; }
    .ufc-divider {
        height: 4px;
        background: linear-gradient(90deg, #e10600 0%, #e10600 30%, #d4af37 30%, #d4af37 34%, transparent 34%);
        border-radius: 2px;
        margin: 0.8rem 0;
    }

    /* Pestañas estilo UFC */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #2a2a2e;
    }
    .stTabs [data-baseweb="tab"] {
        color: #9a9a9f !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        border-bottom: 3px solid #e10600 !important;
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background: #1a1a1f;
        border: 1px solid #2c2c33;
        border-top: 3px solid #e10600;
        padding: 1rem;
        border-radius: 10px;
    }
    [data-testid="stMetric"] label {
        color: #c9c9cf !important;
        font-weight: 600;
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 1.9rem;
    }

    /* Botones oscuros estilo UFC */
    .stButton > button {
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: 0.4px;
        font-family: 'Oswald', sans-serif;
        text-transform: uppercase;
        background-color: #232329 !important;
        color: #f2f2f2 !important;
        border: 1px solid #3a3a44 !important;
        box-shadow: none !important;
    }
    .stButton > button:hover {
        background-color: #2e2e37 !important;
        border-color: #e10600 !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #ff1f14 0%, #c40800 100%) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 2px 10px rgba(225, 6, 0, 0.35) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #ff3527 0%, #e10600 100%) !important;
    }

    /* Selectbox e inputs oscuros */
    .stSelectbox div[data-baseweb="select"],
    .stTextInput input {
        background-color: #1a1a1f !important;
        color: #f2f2f2 !important;
        border: 1px solid #2c2c33 !important;
        border-radius: 8px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a1f;
        border: 1px solid #2c2c33;
        border-radius: 8px;
        color: #f2f2f2;
    }

    /* Alertas de Streamlit en versión oscura */
    div[data-testid="stAlert"] {
        background: #1a1a1f !important;
        border-left: 4px solid #e10600 !important;
        color: #e8e8ec !important;
        border-radius: 8px;
    }
    div[data-testid="stAlert"] * { color: #e8e8ec !important; }
    div[data-testid="stAlert"] svg { fill: #e10600 !important; }

    .notice {
        padding: 1rem 1.1rem; border-radius: 10px;
        background: #241305; border: 1px solid #8a5a1e;
        color: #ffd9a0;
        margin: 0.6rem 0 1.2rem 0;
    }
    .fight-card {
        background: #17171b;
        border: 1px solid #2c2c33;
        border-left: 4px solid #e10600;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .fight-card.title {
        border-left-color: #d4af37;
        background: linear-gradient(90deg, #1c1708 0%, #17171b 60%);
    }
    .fight-card h4 {
        color: #fff; margin: 0 0 0.4rem 0;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        font-size: 1.25rem;
    }
    .fight-card p { color: #a8a8b0; font-size: 0.85rem; margin: 0.15rem 0; }
    .fight-card .vs { color: #e10600; font-weight: 800; }
    .fight-card .gold { color: #d4af37; font-weight: 700; }
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


@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_ufc_events(api_key: str) -> list:
    """Obtiene eventos UFC de UFCalendar API."""
    url = "https://api.ufcalendar.com/v1/events"
    params = {
        "org": "ufc",
        "limit": 10
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        # UFCalendar devuelve "data" no "events"
        return data.get("data", [])
    except requests.RequestException as e:
        st.error(f"Error al obtener eventos: {e}")
        return []


@st.cache_data(ttl=3600)
def fetch_fighter_profile(slug: str) -> dict:
    """Obtiene el perfil de un peleador de UFCalendar API (records, stats)."""
    if not slug:
        return {}
    url = f"https://api.ufcalendar.com/v1/fighters/{slug}"
    headers = {"Authorization": f"Bearer {UFCAL_API_KEY}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("data", {}) or {}
    except requests.RequestException:
        return {}


@st.cache_data(ttl=3600)
def fetch_event_detail(slug: str) -> dict:
    """Obtiene el detalle completo de un evento (incluye card, venue y horarios)."""
    if not slug:
        return {}
    url = f"https://api.ufcalendar.com/v1/events/{slug}"
    headers = {"Authorization": f"Bearer {UFCAL_API_KEY}"}
    for _ in range(2):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json().get("data", {})
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict) and data:
                return data
            return {}
        except requests.RequestException:
            continue
    return {}


@st.cache_data(ttl=3600)
def fetch_event_odds(slug: str) -> dict:
    """Obtiene las cuotas de consenso por pelea de un evento (UFCalendar API).

    Devuelve {fight_id: (cuota_a, cuota_b)} en formato americano.
    """
    if not slug:
        return {}
    url = f"https://api.ufcalendar.com/v1/events/{slug}/odds"
    headers = {"Authorization": f"Bearer {UFCAL_API_KEY}"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json().get("data", {})
        odds_map = {}
        for fight in data.get("fights", []):
            consensus = fight.get("consensus") or {}
            odds_a = (consensus.get("a") or {}).get("american")
            odds_b = (consensus.get("b") or {}).get("american")
            if odds_a is not None and odds_b is not None:
                odds_map[fight.get("fight_id")] = (float(odds_a), float(odds_b))
        return odds_map
    except (requests.RequestException, ValueError):
        return {}


def pro_record(fighter: dict) -> str:
    """Extrae el record profesional MMA (ej. '19-2-0') del perfil del API."""
    profile = fetch_fighter_profile(fighter.get("slug", ""))
    record = (profile.get("records") or {}).get("pro_mma") or {}
    return record.get("value") or "N/A"


def colombia_time(iso_time: str) -> str:
    """Convierte un timestamp ISO a hora de Colombia (America/Bogota)."""
    if not iso_time:
        return "N/A"
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        return dt.astimezone(ZoneInfo("America/Bogota")).strftime("%H:%M")
    except (ValueError, TypeError):
        return "N/A"


COLOMBIA_TZ = ZoneInfo("America/Bogota")
SECTION_LABELS = {
    "early_prelims": "Prelims tempranos",
    "prelims": "Prelims",
    "main": "Main card",
}


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


st.title("UFC Fight Predictor")
st.markdown('<div class="ufc-divider"></div>', unsafe_allow_html=True)
st.caption("Demostración interactiva de los modelos de ganador y método de victoria")

# API key de UFCalendar (variable de entorno; nunca subirla al repo)
UFCAL_API_KEY = os.environ.get("UFCAL_API_KEY", "")

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

with st.expander("Información sobre el modelo"):
    st.markdown("""
    **Métricas de rendimiento (entrenamiento con datos hasta 28-03-2026):**
    
    - **Accuracy ganador:** 69.1% (con cuotas)
    - **Accuracy esquina + método:** 36.7% (experimental)
    
    Las probabilidades del modelo de método no están calibradas y deben interpretarse con cautela.
    """)

# Inicializar estado de sesión
if "step" not in st.session_state:
    st.session_state.step = 1
if "mode" not in st.session_state:
    st.session_state.mode = None
if "red_fighter" not in st.session_state:
    st.session_state.red_fighter = None
if "blue_fighter" not in st.session_state:
    st.session_state.blue_fighter = None
if "weight_class" not in st.session_state:
    st.session_state.weight_class = None
if "title_bout" not in st.session_state:
    st.session_state.title_bout = False
if "location" not in st.session_state:
    st.session_state.location = None
if "red_odds" not in st.session_state:
    st.session_state.red_odds = None
if "blue_odds" not in st.session_state:
    st.session_state.blue_odds = None
if "analysis_shown" not in st.session_state:
    st.session_state.analysis_shown = False

# Paso 1: Menú principal
if st.session_state.step == 1:
    st.markdown("### ¿Qué quieres hacer?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Próximas peleas")
        st.markdown("Ver las peleas más recientes disponibles en el dataset.")
        if st.button("Ver próximas peleas", use_container_width=True):
            st.session_state.mode = "upcoming"
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        st.markdown("#### Pelea personalizada")
        st.markdown("Crea tu propia pelea seleccionando peleadores.")
        if st.button("Crear mi propia pelea", use_container_width=True):
            st.session_state.mode = "custom"
            st.session_state.red_odds = None
            st.session_state.blue_odds = None
            st.session_state.pop("red_odds_input", None)
            st.session_state.pop("blue_odds_input", None)
            st.session_state.step = 3
            st.rerun()

# Paso 2: Próximas peleas
elif st.session_state.step == 2 and st.session_state.mode == "upcoming":
    st.markdown("### Próximas peleas")
    st.markdown("Peleas programadas de UFC (API en tiempo real)")
    
    with st.spinner("Cargando eventos de UFC..."):
        events = fetch_ufc_events(UFCAL_API_KEY)
    
    if not events:
        st.error("No se pudieron cargar los eventos. Intenta más tarde.")
    else:
        # Filtrar eventos de las próximas 2 semanas
        two_weeks_later = datetime.now(timezone.utc) + pd.Timedelta(weeks=2)
        upcoming_events = []
        
        for event in events:
            event_date_str = event.get("starts_at", "")
            if event_date_str:
                try:
                    event_date = datetime.fromisoformat(event_date_str.replace("Z", "+00:00"))
                    if event_date > datetime.now(timezone.utc) and event_date <= two_weeks_later:
                        upcoming_events.append(event)
                except:
                    continue
        
        if not upcoming_events:
            st.info("No hay eventos programados en las próximas 2 semanas.")
        else:
            # Mostrar eventos como tabs (el listado no trae nombre/venue/card;
            # hay que pedir el detalle de cada evento)
            event_details = []
            for event in upcoming_events:
                detail = fetch_event_detail(event.get("slug", ""))
                event_details.append({**event, **detail} if detail else event)

            tabs = st.tabs([
                d.get("title") or d.get("name") or "Evento"
                for d in event_details
            ])

            for tab_idx, tab in enumerate(tabs):
                event = event_details[tab_idx]

                with tab:
                    # Información del evento
                    event_date = (
                        event.get("starts_at") or event.get("date") or "N/A"
                    )
                    if event_date and "T" in event_date:
                        event_date = event_date[:10]

                    venue = event.get("venue", {})
                    if isinstance(venue, dict):
                        venue_name = (
                            venue.get("fullName")
                            or venue.get("name")
                            or "N/A"
                        )
                        if venue.get("city"):
                            venue_name = f"{venue_name} ({venue['city']})"
                    else:
                        venue_name = str(venue) if venue else "N/A"

                    st.markdown(f"**Fecha:** {event_date}  |  **Ubicación:** {venue_name}")
                    st.markdown("---")

                    card = event.get("card", []) or []
                    odds_map = fetch_event_odds(event.get("slug", "")) if card else {}

                    if card:
                        # Horas de cada sección del evento en hora Colombia
                        section_times = {
                            "early_prelims": colombia_time(event.get("early_prelims_at", "")),
                            "prelims": colombia_time(event.get("prelims_at", "")),
                            "main": colombia_time(event.get("main_card_at", "")),
                        }
                        if event.get("starts_at") and all(
                            hour == "N/A" for hour in section_times.values()
                        ):
                            section_times["event"] = colombia_time(event.get("starts_at", ""))
                        time_parts = [
                            f"{SECTION_LABELS.get(section, 'Evento')}: {hour}"
                            for section, hour in section_times.items()
                            if hour != "N/A"
                        ]
                        if time_parts:
                            st.markdown(f"**Horarios (Colombia):** {' · '.join(time_parts)}")
                        st.markdown("---")

                        # Organizar en cuadrícula de 2 columnas
                        for i in range(0, len(card), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i + j < len(card):
                                    fight = card[i + j]
                                    fighter_a = fight.get("fighter_a", {})
                                    fighter_b = fight.get("fighter_b", {})
                                    fighter1 = fighter_a.get("name", "N/A")
                                    fighter2 = fighter_b.get("name", "N/A")
                                    weight_class = fight.get("weight_class", "N/A")
                                    section = fight.get("card_section", "")
                                    fight_time = section_times.get(section, "")
                                    rec1 = pro_record(fighter_a)
                                    rec2 = pro_record(fighter_b)
                                    fight_odds = odds_map.get(fight.get("id"))
                                    odds_label = ""
                                    if fight_odds:
                                        odds_label = f"Cuotas: {fight_odds[0]:g} / {fight_odds[1]:g}"

                                    card_class = "fight-card title" if fight.get("is_title") else "fight-card"
                                    title_html = '<span class="gold"> · TÍTULO</span>' if fight.get("is_title") else ""

                                    with cols[j]:
                                        # Card/cuadro para cada pelea con más info
                                        st.markdown(
                                            f"""
                                            <div class="{card_class}">
                                                <h4>{fighter1} <span class="vs">vs</span> {fighter2}</h4>
                                                <p>División: {weight_class}{title_html}</p>
                                                <p>Record: {rec1} vs {rec2}</p>
                                                <p>{SECTION_LABELS.get(section, section)}{f' · ~{fight_time} (Colombia)' if fight_time != 'N/A' and fight_time else ''}</p>
                                                <p>{odds_label}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True
                                        )

                                        if st.button("Analizar", key=f"fight_{tab_idx}_{i+j}"):
                                            st.session_state.red_fighter = fighter1
                                            st.session_state.blue_fighter = fighter2
                                            st.session_state.weight_class = weight_class
                                            st.session_state.title_bout = bool(fight.get("is_title"))
                                            st.session_state.location = venue_name
                                            st.session_state.country = (
                                                venue.get("country", "")
                                                if isinstance(venue, dict)
                                                else ""
                                            )
                                            if fight_odds:
                                                st.session_state.red_odds = fight_odds[0]
                                                st.session_state.blue_odds = fight_odds[1]
                                            else:
                                                st.session_state.red_odds = None
                                                st.session_state.blue_odds = None
                                            st.session_state.pop("red_odds_input", None)
                                            st.session_state.pop("blue_odds_input", None)
                                            st.session_state.came_from = "upcoming"
                                            st.session_state.mode = "custom"
                                            st.session_state.step = 4
                                            st.rerun()
                    else:
                        st.info("No hay información de peleas para este evento aún.")
    
    st.markdown("---")
    if st.button("Volver al inicio", use_container_width=True):
        st.session_state.step = 1
        st.rerun()

# Paso 3: Selección de peleadores (modo personalizado)
elif st.session_state.step == 3 and st.session_state.mode == "custom":
    # Si los peleadores ya vienen del dashboard, saltar al contexto
    if st.session_state.red_fighter and st.session_state.blue_fighter:
        st.session_state.step = 4
        st.rerun()
    
    try:
        profiles = load_fighter_profiles()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()
    
    st.markdown("### Elige los peleadores")
    st.markdown("Selecciona un atleta para cada esquina. No puedes elegir el mismo peleador en ambas.")
    
    profile_names = profiles["fighter_name"].astype(str).tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Esquina Roja")
        red_default = (
            profile_names.index("Alexa Grasso")
            if "Alexa Grasso" in profile_names
            else 0
        )
        red_name = st.selectbox(
            "Selecciona peleador rojo",
            profile_names,
            index=red_default,
            key="custom_red_fighter",
            help="Empieza a escribir para filtrar los nombres disponibles.",
        )
    
    with col2:
        st.markdown("#### Esquina Azul")
        red_key = normalize_fighter(red_name)
        blue_names = [
            name for name in profile_names if normalize_fighter(name) != red_key
        ]
        blue_default_name = "Maycee Barber"
        blue_default = (
            blue_names.index(blue_default_name)
            if blue_default_name in blue_names
            else 0
        )
        blue_name = st.selectbox(
            "Selecciona peleador azul",
            blue_names,
            index=blue_default,
            key="custom_blue_fighter",
            help="Empieza a escribir para filtrar los nombres disponibles.",
        )
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("Volver", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col_next:
        if st.button("Siguiente", use_container_width=True):
            st.session_state.red_fighter = red_name
            st.session_state.blue_fighter = blue_name
            st.session_state.step = 4
            st.rerun()

# Paso 2: Dashboard de peleas del fin de semana
elif st.session_state.step == 2 and st.session_state.mode == "dashboard":
    st.markdown("### Peleas del fin de semana")
    
    if not api_key:
        st.warning("Por favor, ingresa tu API key de UFCalendar en la barra lateral para ver las peleas.")
        col_back = st.columns([1])
        with col_back[0]:
            if st.button("Volver", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
    else:
        with st.spinner("Cargando eventos de UFC..."):
            events_data = fetch_ufc_events(api_key)
        
        if not events_data or "events" not in events_data:
            st.error("No se pudieron cargar los eventos. Verifica tu API key.")
            col_back = st.columns([1])
            with col_back[0]:
                if st.button("Volver", use_container_width=True):
                    st.session_state.step = 1
                    st.rerun()
        else:
            events = events_data["events"]
            
            if not events:
                st.info("No hay eventos programados en este momento.")
                col_back = st.columns([1])
                with col_back[0]:
                    if st.button("Volver", use_container_width=True):
                        st.session_state.step = 1
                        st.rerun()
            else:
                st.markdown(f"Se encontraron {len(events)} eventos programados:")
                
                for event in events:
                    with st.expander(f"{event.get('name', 'Evento sin nombre')} - {event.get('date', 'Fecha no disponible')}"):
                        st.markdown(f"**Fecha:** {event.get('date', 'N/A')}")
                        st.markdown(f"**Ubicación:** {event.get('venue', 'N/A')}")
                        
                        # Mostrar peleas del evento
                        if "card" in event and event["card"]:
                            st.markdown("**Peleas:**")
                            for fight in event["card"]:
                                fighter1 = fight.get("fighter1", {}).get("name", "N/A")
                                fighter2 = fight.get("fighter2", {}).get("name", "N/A")
                                st.markdown(f"- {fighter1} vs {fighter2}")
                                
                                if st.button(f"Analizar: {fighter1} vs {fighter2}", key=f"fight_{fight.get('id')}"):
                                    # Aquí podríamos cargar los datos para análisis
                                    st.info("Función de análisis automático próximamente disponible.")
                
                col_back = st.columns([1])
                with col_back[0]:
                    if st.button("Volver", use_container_width=True):
                        st.session_state.step = 1
                        st.rerun()

# Paso 4: Contexto de la pelea (modo personalizado)
elif st.session_state.step == 4 and st.session_state.mode == "custom":
    st.markdown("### Define el contexto de la pelea")
    st.markdown("Configura la división, si es por título y la ubicación del evento.")
    
    # Cargar datos procesados para obtener opciones
    if not PROCESSED_PATH.is_file():
        st.error("No se encontró el archivo procesado necesario.")
        st.stop()
    
    processed_data = pd.read_csv(PROCESSED_PATH, low_memory=False)
    
    weight_classes = sorted(
        str(value)
        for value in processed_data["weight_class"].dropna().unique()
    )
    preferred_class = (
        "Lightweight" if "Lightweight" in weight_classes else weight_classes[0]
    )
    
    location_countries = (
        processed_data.dropna(subset=["location", "country"])
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### División")
        weight_class = st.selectbox(
            "División del combate",
            weight_classes,
            index=weight_classes.index(preferred_class),
        )
        title_bout = st.checkbox("Combate por título (5 asaltos)")
    
    with col2:
        st.markdown("#### Ubicación")
        location = st.selectbox(
            "Ubicación del evento",
            locations,
            index=locations.index(default_location),
        )
        st.caption(f"País/región registrado: {location_countries[location]}")
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("Volver", use_container_width=True):
            if st.session_state.get("came_from") == "upcoming":
                st.session_state.mode = "upcoming"
                st.session_state.step = 2
            else:
                st.session_state.step = 3
            st.rerun()
    with col_next:
        if st.button("Siguiente", use_container_width=True):
            st.session_state.weight_class = weight_class
            st.session_state.title_bout = title_bout
            st.session_state.location = location
            st.session_state.country = location_countries[location]
            st.session_state.step = 5
            st.rerun()

# Paso 5: Cuotas (modo personalizado)
elif st.session_state.step == 5 and st.session_state.mode == "custom":
    st.markdown("### Cuotas de apuestas (opcional)")
    
    st.markdown("""
    #### ¿Qué son las cuotas americanas?
    
    Las cuotas americanas indican cuánto puedes ganar por cada apuesta:
    
    - **Negativa (ej. -150):** El peleador es favorito. Necesitas apostar $150 para ganar $100.
    - **Positiva (ej. +130):** El peleador es no favorito. Si apuestas $100, ganas $130.
    
    **¿Dónde encontrarlas?**
    - DraftKings, FanDuel, BetMGM (casas de apuestas)
    - OddsChecker, OddsShark (agregadores de cuotas)
    - Si no tienes cuotas, déjalas vacías y el modelo usará valores promedio.
    """)
    
    col1, col2 = st.columns(2)

    def odds_default(value):
        return "" if value is None else f"{value:g}"

    with col1:
        st.markdown(f"#### Cuota: {st.session_state.red_fighter}")
        red_odds_input = st.text_input(
            "Cuota roja (ej. -150 o +130)",
            value=odds_default(st.session_state.red_odds),
            placeholder="-150 o +130",
            key="red_odds_input"
        )

    with col2:
        st.markdown(f"#### Cuota: {st.session_state.blue_fighter}")
        blue_odds_input = st.text_input(
            "Cuota azul (ej. +130 o -150)",
            value=odds_default(st.session_state.blue_odds),
            placeholder="+130 o -150",
            key="blue_odds_input"
        )

    if st.session_state.red_odds is not None or st.session_state.blue_odds is not None:
        st.info("Las cuotas se cargaron automáticamente desde las casas de apuestas. Puedes editarlas si lo deseas.")
    else:
        st.info("""
        Si no tienes cuotas reales, déjalas vacías. El modelo funcionará pero usará
        valores promedio del entrenamiento, lo cual es menos preciso que usar cuotas reales.
        """)
    
    col_back, col_next = st.columns([1, 1])
    with col_back:
        if st.button("Volver", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col_next:
        if st.button("Comenzar análisis", use_container_width=True):
            try:
                red_odds = parse_american_odds(red_odds_input, st.session_state.red_fighter)
                blue_odds = parse_american_odds(blue_odds_input, st.session_state.blue_fighter)
                st.session_state.red_odds = red_odds
                st.session_state.blue_odds = blue_odds
                st.session_state.analysis_shown = True
                st.rerun()
            except ValueError as error:
                st.error(str(error))

# Análisis con animación
if st.session_state.analysis_shown:
    with st.spinner("Analizando datos y generando predicciones..."):
        import time
        time.sleep(1.5)  # Pequeña pausa para efecto de animación
    
    st.success("Análisis completado")
    
    # Preparar datos (solo modo personalizado ahora)
    try:
        profiles = load_fighter_profiles()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()
    
    red_profile = profiles.loc[normalize_fighter(st.session_state.red_fighter)]
    blue_profile = profiles.loc[normalize_fighter(st.session_state.blue_fighter)]
    matchup_features = build_matchup_features(
        red_profile,
        blue_profile,
        feature_columns,
        weight_class=st.session_state.weight_class,
        title_bout=st.session_state.title_bout,
        location=st.session_state.location,
        country=st.session_state.country,
        red_odds=st.session_state.red_odds,
        blue_odds=st.session_state.blue_odds,
    )
    fight = matchup_features.iloc[0].copy()
    fight["R_fighter"] = st.session_state.red_fighter
    fight["B_fighter"] = st.session_state.blue_fighter
    fight["R_profile_date"] = red_profile["profile_date"]
    fight["B_profile_date"] = blue_profile["profile_date"]
    
    red_name = fight["R_fighter"]
    blue_name = fight["B_fighter"]
    
    # Encabezado de la pelea
    st.markdown("---")
    st.markdown(f"## {red_name}  vs  {blue_name}")
    
    st.info(f"**División:** {fight['weight_class']} {'(Título)' if fight['title_bout'] else ''}")
    st.caption(
        f"Perfiles deportivos disponibles hasta: "
        f"roja {pd.Timestamp(fight['R_profile_date']):%d-%m-%Y}; "
        f"azul {pd.Timestamp(fight['B_profile_date']):%d-%m-%Y}. "
        "No son necesariamente las estadísticas actuales."
    )
    if st.session_state.red_odds is None or st.session_state.blue_odds is None:
        st.warning(
            "Falta al menos una cuota. El modelo fue entrenado con cuotas; "
            "los valores vacíos se imputarán con valores típicos del entrenamiento. "
            "Esto no equivale al modelo sin cuotas."
        )
    
    # Generar predicciones
    input_row = fight.loc[feature_columns].to_frame().T
    winner_probabilities = winner_model.predict_proba(input_row)[0]
    winner_classes = list(winner_model.steps[-1][1].classes_)
    red_class_index = winner_classes.index(1)
    red_probability = float(winner_probabilities[red_class_index])
    blue_probability = 1.0 - red_probability
    predicted_side = "Red" if red_probability >= 0.5 else "Blue"
    predicted_fighter = red_name if predicted_side == "Red" else blue_name
    
    # Sección de ganador
    st.markdown("### ¿A quién eligió el modelo?")
    
    winner_col1, winner_col2, winner_col3 = st.columns([1, 2, 1])
    with winner_col1:
        st.metric(
            f"{red_name}",
            f"{red_probability:.1%}",
            delta=None,
            delta_color="normal"
        )
    with winner_col2:
        st.progress(
            red_probability,
            text=f"Probabilidad estimada para {red_name} (rojo)"
        )
    with winner_col3:
        st.metric(
            f"{blue_name}",
            f"{blue_probability:.1%}",
            delta=None,
            delta_color="normal"
        )
    
    st.success(f"**Predicción del modelo:** gana **{predicted_fighter}** ({'roja' if predicted_side == 'Red' else 'azul'})")
    
    # Sección de método de victoria
    st.markdown("---")
    st.markdown("### ¿Cómo podría ganar?")
    
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
    
    # Estadísticas comparativas
    st.markdown("---")
    st.markdown("### Comparación de estadísticas previas")
    
    with st.expander("Ver estadísticas detalladas de ambos peleadores"):
        red_stats = fighter_stats(fight, "R")
        blue_stats = fighter_stats(fight, "B")
        comparison = pd.concat([red_stats, blue_stats], axis=1)
        comparison.columns = [f"{red_name}", f"{blue_name}"]
        st.dataframe(comparison, width="stretch")
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
    
    # Botón para reiniciar
    st.markdown("---")
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        if st.button("Volver al inicio", use_container_width=True):
            st.session_state.step = 1
            st.session_state.mode = None
            st.session_state.red_fighter = None
            st.session_state.blue_fighter = None
            st.session_state.weight_class = None
            st.session_state.title_bout = False
            st.session_state.location = None
            st.session_state.red_odds = None
            st.session_state.blue_odds = None
            st.session_state.historical_fight = None
            st.session_state.analysis_shown = False
            st.rerun()
    
    st.divider()
    st.caption(
        "Proyecto educativo. Los resultados son históricos y experimentales; "
        "las probabilidades no están calibradas, no garantizan resultados y no son consejo de apuestas."
    )
