# ============================================================
# dashboard.py — Dashboard Streamlit para monitoreo de oro
# Gold Price Monitor — Phase 4
#
# Ejecución:
#   streamlit run dashboard.py
#   streamlit run dashboard.py -- --mode mock
# ============================================================

import sys
import os
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Cargar variables desde config.env automáticamente
# ─────────────────────────────────────────────────────────────
def _load_env_file():
    """Lee config.env del directorio del proyecto y carga las variables."""
    env_path = Path(__file__).parent / "config.env"
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value

_load_env_file()   # ← يُشغَّل عند بدء الداشبورد

def _load_streamlit_secrets():
    """
    يقرأ API keys من Streamlit Secrets (للـ Cloud deployment).
    له الأولوية على config.env.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ["NEWS_API_KEY", "ALPHAVANTAGE_KEY"]:
                val = st.secrets.get(key, "")
                if val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass

_load_streamlit_secrets()

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Streamlit — Debe ser lo primero en importarse
# ─────────────────────────────────────────────────────────────
try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)

# ── MetaTrader Bridge (opcional) ────────────────────────────────
try:
    from metatrader_bridge import MetaTraderBridge, MTBridgeConfig, get_atr_from_df
    _MT_AVAILABLE = True
except ImportError:
    _MT_AVAILABLE = False

# ── News Sentiment Filter (opcional) ────────────────────────────
try:
    from news_filter import GoldNewsSentimentFilter, NewsConfig, SentimentResult
    _NEWS_AVAILABLE = True
except ImportError:
    _NEWS_AVAILABLE = False

# ── Performance Tracker (opcional) ──────────────────────────────
try:
    from performance_tracker import PerformanceTracker, PerformanceStats
    from trading_filters import TradingFiltersManager
    _PERF_AVAILABLE = True
except ImportError:
    _PERF_AVAILABLE = False

# ── Support Resistance & Economic Calendar (opcionales) ──────────
try:
    from support_resistance import SupportResistanceCalculator
    from economic_calendar import EconomicCalendar
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────
def _setup_page():
    st.set_page_config(
        page_title="Gold Price Monitor 🥇",
        page_icon="🥇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #ffd700;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .signal-buy    { color: #00ff88; font-weight: bold; font-size: 1.4em; }
        .signal-sell   { color: #ff4444; font-weight: bold; font-size: 1.4em; }
        .signal-neutral{ color: #aaaaaa; font-weight: bold; font-size: 1.4em; }
        .regime-bear   { color: #ff6b6b; }
        .regime-bull   { color: #00cc66; }
        .regime-side   { color: #aaaaaa; }
        div[data-testid="stMetric"] { background: #1a1a2e; border-radius: 8px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Cargar modelo (con caché de Streamlit)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model_and_data(mode: str = "mock", n_rows: int = 500):
    """Carga datos y entrena el modelo completo."""
    from config import AppConfig
    from data_fetcher import GoldDataFetcher
    from indicators import TechnicalIndicators
    from feature_engineer import FeatureEngineer, FEATURE_COLUMNS
    from lstm_model import LSTMConfig
    from ensemble import GoldEnsemble, EnsembleConfig
    from backtester import GoldBacktester, BacktestConfig

    cfg = AppConfig(mode=mode)
    cfg.mock.n_rows = n_rows

    df_raw  = GoldDataFetcher(cfg).get_data()
    df_ind  = TechnicalIndicators(df_raw, cfg).add_all().get_dataframe()
    fe      = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()
    df_feat = fe.get_full_data()

    ens_cfg = EnsembleConfig(
        rf_weight=0.40, lstm_weight=0.40, hmm_weight=0.20,
        lstm_config=LSTMConfig(hidden_size=48, seq_length=15,
                               epochs=60, patience=12, batch_size=16),
    )
    ensemble = GoldEnsemble(ens_cfg)
    ensemble.fit(df_feat, df_ind)
    results = ensemble.predict(df_feat, df_ind)

    # Backtesting
    test_prices = df_ind.loc[results.signals.index, "Close"]
    bt = GoldBacktester(BacktestConfig())
    bt_res = bt.run(test_prices, results.signals)
    trades_df = bt.get_trades_df()

    return {
        "df_ind":    df_ind,
        "df_feat":   df_feat,
        "results":   results,
        "bt_res":    bt_res,
        "trades_df": trades_df,
        "ensemble":  ensemble,
    }


# ─────────────────────────────────────────────────────────────
# Barra lateral
# ─────────────────────────────────────────────────────────────
def _sidebar(data: dict) -> dict:
    st.sidebar.markdown("## ⚙️ Configuración")
    st.sidebar.markdown("---")

    # ── Modo de datos — predeterminado: live ──────────────────────────
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "live"

    mode = st.sidebar.selectbox(
        "Modo de datos",
        options=["live", "mock"],
        index=0 if st.session_state["data_mode"] == "live" else 1,
        help="live = datos reales de yfinance | mock = datos sintéticos de prueba",
    )

    # Limpiar caché automáticamente al cambiar modo
    if mode != st.session_state["data_mode"]:
        st.session_state["data_mode"] = mode
        st.cache_resource.clear()
        st.rerun()

    n_rows = st.sidebar.slider("Número de filas (Mock)", 300, 1000, 500, 50) if mode == "mock" else 500
    show_features = st.sidebar.checkbox("Mostrar importancia de características", value=True)

    if st.sidebar.button("🔄 Reentrenar modelo", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    # ── Sección MetaTrader ─────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 MetaTrader Bridge")

    mt_enabled = False
    mt_cfg     = None

    if not _MT_AVAILABLE:
        st.sidebar.caption("⚠️ metatrader_bridge.py no encontrado")
    else:
        mt_enabled = st.sidebar.toggle("Activar envío de señales a MT", value=False)
        if mt_enabled:
            mt_files_path = st.sidebar.text_input(
                "Ruta MQL5\\Files",
                value=os.getenv(
                    "MT_FILES_PATH",
                    str(os.path.expanduser("~") +
                        "/AppData/Roaming/MetaQuotes/Terminal/Common/Files")
                ),
                help="Ruta donde se escribirá gold_signal.json"
            )
            mt_lot   = st.sidebar.number_input("Tamaño de lote", 0.01, 10.0, 0.01, 0.01)
            mt_conf  = st.sidebar.slider("Confianza mínima", 0.50, 0.95, 0.55, 0.05)
            mt_auto  = st.sidebar.checkbox("Ejecución automática (AutoTrade en EA)", value=False)
            mt_cfg   = MTBridgeConfig(
                mt_files_path=mt_files_path,
                lot_size=mt_lot,
                min_confidence=mt_conf,
            )
            if not os.path.isdir(mt_files_path):
                st.sidebar.warning("⚠️ Ruta no encontrada — asegúrese de ejecutar MT5 primero")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Calendario Económico")

    if _SR_AVAILABLE:
        try:
            from economic_calendar import EconomicCalendar
            ec = EconomicCalendar()
            # Obtener próximos eventos de alto impacto (7 días)
            all_upcoming = ec.get_upcoming_events(days_ahead=7)
            upcoming_high = [e for e in all_upcoming if e.impact == "HIGH"][:3]

            if upcoming_high:
                for evt in upcoming_high:
                    now      = datetime.now()
                    delta    = evt.date - now
                    days_d   = delta.days
                    hours_d  = delta.seconds // 3600
                    if days_d >= 1:
                        time_left = f"{days_d}d {hours_d}h"
                    else:
                        time_left = f"{hours_d}h {(delta.seconds % 3600) // 60}m"
                    icon = "🔴" if delta.days == 0 else "🟡"
                    st.sidebar.markdown(f"{icon} **{evt.name}**")
                    st.sidebar.caption(f"⏱️ en {time_left}")

                # Verificar si hoy es día de alto riesgo
                is_high, msg = ec.is_high_impact_day()
                if is_high:
                    st.sidebar.warning("⚠️ **Día de alto riesgo** — Cuidado")
            else:
                st.sidebar.info("📡 Sin eventos HIGH próximos")
        except Exception as e:
            logging.debug(f"No se pudo cargar calendario económico: {e}")
    else:
        st.sidebar.caption("📡 economic_calendar.py no encontrado")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Sobre el modelo")
    st.sidebar.markdown("""
    - **RF** : Random Forest (40%)
    - **LSTM**: Red neuronal recurrente (40%)
    - **HMM** : Filtro de régimen de mercado (20%)
    """)

    # ────── Último tiempo de entrenamiento y botón de reentrenamiento ──────
    if "last_training_time" not in st.session_state:
        st.session_state["last_training_time"] = "Nunca"

    st.sidebar.caption(f"⏱️ Último entrenamiento: {st.session_state.get('last_training_time', 'Nunca')}")

    if st.sidebar.button("🔄 Reentrenar ahora", use_container_width=True, key="sidebar_retrain"):
        st.session_state["last_training_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.cache_resource.clear()
        st.success("✅ Caché limpiado — El modelo se reentrenará en la próxima carga.")
        st.rerun()

    # ── Indicadores de rendimiento (si están disponibles) ──────────
    if _PERF_AVAILABLE:
        try:
            signals_raw, signals_list = _load_performance_data("signals_history.json")
            if signals_list and len(signals_list) > 0:
                closed_signals = [s for s in signals_list if s.outcome != "PENDING"]
                if len(closed_signals) > 0:
                    winners = [s for s in closed_signals if s.pnl_usd > 0]
                    win_rate = len(winners) / len(closed_signals)
                    total_pnl = sum(s.pnl_usd for s in closed_signals)

                    st.sidebar.markdown("---")
                    st.sidebar.markdown("### 📈 Indicadores de Desempeño")
                    col_wr, col_pnl = st.sidebar.columns(2)
                    with col_wr:
                        st.metric("Win Rate", f"{win_rate:.1%}", label_visibility="collapsed")
                    with col_pnl:
                        color = "normal" if total_pnl >= 0 else "inverse"
                        st.metric("PnL", f"${total_pnl:+,.0f}", delta_color=color, label_visibility="collapsed")
        except Exception as e:
            logging.debug(f"No se pudieron cargar indicadores de rendimiento: {e}")

    return {
        "mode": mode, "n_rows": n_rows,
        "show_features": show_features,
        "mt_enabled": mt_enabled,
        "mt_cfg":     mt_cfg,
    }


# ─────────────────────────────────────────────────────────────
# Funciones auxiliares — niveles y alertas
# ─────────────────────────────────────────────────────────────
def _calculate_support_resistance(df_ind: pd.DataFrame) -> dict:
    """Calcula niveles de soporte y resistencia si SR está disponible, sino usa Pivot."""
    result = {"s3": None, "s2": None, "s1": None, "r1": None, "r2": None, "r3": None, "pp": None}

    if len(df_ind) < 20:
        return result

    # Si SupportResistanceCalculator está disponible, usarlo
    if _SR_AVAILABLE:
        try:
            calc = SupportResistanceCalculator(df_ind)
            pivots = calc.get_pivot_points()          # método correcto
            result = {
                "pp": pivots.get("P"),
                "r1": pivots.get("R1"), "r2": pivots.get("R2"), "r3": pivots.get("R3"),
                "s1": pivots.get("S1"), "s2": pivots.get("S2"), "s3": pivots.get("S3"),
            }
        except Exception as e:
            logging.debug(f"No se pudo calcular SR: {e}")

    # Fallback: Pivot Point simple (últimas 20 velas)
    if result["pp"] is None:
        recent = df_ind.tail(20)
        high = recent["High"].max()
        low = recent["Low"].min()
        close = df_ind["Close"].iloc[-1]
        pp = (high + low + close) / 3.0

        result["pp"] = pp
        result["r1"] = (2 * pp) - low
        result["s1"] = (2 * pp) - high
        result["r2"] = pp + (high - low)
        result["s2"] = pp - (high - low)
        result["r3"] = high + 2 * (pp - low)
        result["s3"] = low - 2 * (high - pp)

    return result


def _get_warning_level(df_ind: pd.DataFrame, results) -> str:
    """
    Determina el nivel de alerta combinando calendario económico + técnicos.
      DANGER  = evento HIGH en < 2h   ó  RSI extremo + ATR elevado
      CAUTION = evento HIGH hoy/mañana ó  RSI moderadamente extremo
      CLEAR   = condiciones normales
    """
    try:
        # ── 1. Consultar calendario económico ─────────────────────────
        cal_level = "CLEAR"
        if _SR_AVAILABLE:
            try:
                from economic_calendar import EconomicCalendar
                ec = EconomicCalendar()
                cal_level = ec.get_warning_level()
                if cal_level == "DANGER":
                    return "DANGER"          # prioridad máxima
            except Exception:
                cal_level = "CLEAR"

        # ── 2. Indicadores técnicos ────────────────────────────────────
        rsi = df_ind["RSI"].iloc[-1] if "RSI" in df_ind.columns else 50.0
        atr_ok = "ATR" in df_ind.columns
        atr_val  = df_ind["ATR"].iloc[-1]      if atr_ok else 0.0
        atr_mean = df_ind["ATR"].mean()         if atr_ok else 0.0

        # RSI muy extremo + volatilidad alta → DANGER
        if (rsi > 77 or rsi < 23) and atr_ok and atr_val > atr_mean:
            return "DANGER"

        # Calendario CAUTION  ó  RSI moderadamente extremo
        if cal_level == "CAUTION" or rsi > 70 or rsi < 30:
            return "CAUTION"

        return "CLEAR"
    except Exception:
        return "CLEAR"


def _get_dxy_bias(df_ind: pd.DataFrame) -> tuple:
    """
    Estima el sesgo del DXY a partir de la tendencia del precio del oro
    (correlación inversa: oro sube → DXY baja, y viceversa).
    Retorna (texto, color_hex).
    """
    try:
        close = df_ind["Close"]
        if len(close) < 10:
            return "Neutral", "#aaaaaa"
        pct = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]
        if pct > 0.005:
            return "Bajista 📉", "#88ff88"    # oro sube → DXY baja → alcista para oro
        elif pct < -0.005:
            return "Alcista 📈", "#ff8888"    # oro baja → DXY sube → bajista para oro
        return "Neutral", "#aaaaaa"
    except Exception:
        return "Neutral", "#aaaaaa"


def _get_next_event_str() -> str:
    """Retorna texto del próximo evento económico HIGH impact, o '—'."""
    if not _SR_AVAILABLE:
        return "—"
    try:
        from economic_calendar import EconomicCalendar
        ec = EconomicCalendar()
        evt = ec.get_next_high_impact()
        if evt is None:
            return "Sin eventos HIGH"
        now = datetime.now()
        delta = evt.date - now
        days  = delta.days
        hours = delta.seconds // 3600
        mins  = (delta.seconds % 3600) // 60
        if days > 1:
            time_str = f"{days}d"
        elif days == 1:
            time_str = f"1d {hours}h"
        elif hours > 0:
            time_str = f"{hours}h {mins}m"
        else:
            time_str = f"{mins}m"
        return f"{evt.name} en {time_str}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────────
# Tarjetas de señal principal — LANZAMIENTO NUEVO
# ─────────────────────────────────────────────────────────────
def _render_signal_cards(data: dict, news_score: float = 0.0):
    results  = data["results"]
    df_ind   = data["df_ind"]
    bt_res   = data["bt_res"]

    # Última señal
    last_sig   = int(results.signals.iloc[-1])
    last_conf  = float(results.confidence.iloc[-1])
    last_regime = int(results.regimes.iloc[-1])
    last_price  = float(df_ind["Close"].iloc[-1])
    prev_price  = float(df_ind["Close"].iloc[-2])
    price_chg   = (last_price - prev_price) / prev_price

    sig_map     = {1: "BUY 🟢", -1: "SELL 🔴", 0: "NEUTRAL ⚪"}
    regime_map  = {0: "Bajista 🔴", 1: "Lateral ⚪", 2: "Alcista 🟢"}
    sig_emoji   = {1: "🟢", -1: "🔴", 0: "⚪"}

    sig_text = sig_map.get(last_sig, "NEUTRAL ⚪")

    # Calcular SL/TP
    atr_val = df_ind["ATR"].iloc[-1] if "ATR" in df_ind.columns else 20.0
    sl_price = last_price - (atr_val * 1.5)
    tp_price = last_price + (atr_val * 2.5)
    rr_ratio = abs(tp_price - last_price) / abs(last_price - sl_price) if (last_price - sl_price) != 0 else 0

    st.markdown("## 🥇 Gold Price Monitor — Panel de Control Integral")
    st.markdown("---")

    # ────── FILA 1: Señal Principal (ancha) ──────
    st.markdown("### Señal Principal")
    col_sig = st.columns(1)[0]
    with col_sig:
        sig_color = "#00ff88" if last_sig == 1 else "#ff4444" if last_sig == -1 else "#aaaaaa"
        st.markdown(
            f"""<div style="background:#1a1a2e;border:2px solid {sig_color};border-radius:12px;
                padding:20px;text-align:center;">
                <div style="font-size:2em;font-weight:bold;color:{sig_color};">{sig_text}</div>
                <div style="font-size:1.1em;color:#e0c97f;margin-top:8px;">Precio: ${last_price:,.2f}</div>
                <div style="font-size:0.95em;color:#aaaaaa;margin-top:4px;">
                Confianza: <span style="color:#ffd700;font-weight:bold;">{last_conf:.0%}</span> |
                Régimen: <span style="color:#ffd700;font-weight:bold;">{regime_map[last_regime]}</span>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    # ────── FILA 2: Detalles de Posición (4 columnas) ──────
    st.markdown("### Detalles de la Entrada")
    col_e, col_sl, col_tp, col_rr = st.columns(4)

    with col_e:
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #ffd700;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:0.85em;color:#aaaaaa;">🎯 Entrada</div>
        <div style="font-size:1.3em;font-weight:bold;color:#e0c97f;">${last_price:,.2f}</div>
        </div>""", unsafe_allow_html=True)

    with col_sl:
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #ff4444;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:0.85em;color:#aaaaaa;">🛡️ Stop Loss</div>
        <div style="font-size:1.1em;font-weight:bold;color:#ff8888;">${sl_price:,.2f}</div>
        <div style="font-size:0.75em;color:#ff8888;">({abs(sl_price - last_price) / last_price:.2%})</div>
        </div>""", unsafe_allow_html=True)

    with col_tp:
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #00cc66;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:0.85em;color:#aaaaaa;">🏆 Take Profit</div>
        <div style="font-size:1.1em;font-weight:bold;color:#88ff88;">${tp_price:,.2f}</div>
        <div style="font-size:0.75em;color:#88ff88;">({abs(tp_price - last_price) / last_price:.2%})</div>
        </div>""", unsafe_allow_html=True)

    with col_rr:
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #9966ff;border-radius:8px;padding:12px;text-align:center;">
        <div style="font-size:0.85em;color:#aaaaaa;">⚖️ Ratio R/R</div>
        <div style="font-size:1.3em;font-weight:bold;color:#bb88ff;">1 : {rr_ratio:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # ────── FILA 3: Condiciones del Mercado (iconos pequeños) ──────
    st.markdown("### Contexto del Mercado")
    col_t, col_sent, col_rsi, col_dxy, col_event = st.columns(5)

    with col_t:
        # Sesión actual — OVERLAP tiene prioridad sobre LONDON y NY
        from datetime import timezone
        utc_now  = datetime.now(timezone.utc)
        hour_utc = utc_now.hour
        if 13 <= hour_utc < 17:
            session_name = "OVERLAP 🔥"
            sess_color   = "#ffd700"
            mult         = "1.3x"
        elif 8 <= hour_utc < 17:
            session_name = "LONDON"
            sess_color   = "#60a5fa"
            mult         = "1.1x"
        elif 17 <= hour_utc < 22:
            session_name = "NEW YORK"
            sess_color   = "#34d399"
            mult         = "1.1x"
        else:
            session_name = "ASIAN 😴"
            sess_color   = "#aaaaaa"
            mult         = "0.7x"

        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {sess_color};border-radius:6px;padding:8px;font-size:0.85em;">
        🕐 {session_name}<br/><span style="color:{sess_color};font-weight:bold;">{mult}</span>
        </div>""", unsafe_allow_html=True)

    with col_sent:
        # Sentiment dinámico (del parámetro news_score)
        if news_score >= 0.25:
            sentiment  = "Alcista 🟢"
            sent_color = "#00cc66"
        elif news_score <= -0.25:
            sentiment  = "Bajista 🔴"
            sent_color = "#ff4444"
        else:
            sentiment  = "Neutral ⚪"
            sent_color = "#aaaaaa"
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {sent_color};border-radius:6px;padding:8px;font-size:0.85em;">
        📰 Noticias<br/><span style="color:{sent_color};">{sentiment} ({news_score:+.2f})</span>
        </div>""", unsafe_allow_html=True)

    with col_rsi:
        rsi_val = df_ind["RSI"].iloc[-1] if "RSI" in df_ind.columns else 50.0
        if rsi_val > 70:
            rsi_label = "Sobrecomprado"
            rsi_color = "#ff8888"
        elif rsi_val < 30:
            rsi_label = "Sobrevendido"
            rsi_color = "#88ff88"
        else:
            rsi_label = "Neutral"
            rsi_color = "#aaaaaa"
        # Mostrar también ATR si disponible
        atr_str = ""
        if "ATR" in df_ind.columns:
            atr_val = df_ind["ATR"].iloc[-1]
            atr_str = f"<br/><span style='color:#aaaaaa;font-size:0.8em;'>ATR: {atr_val:.2f}</span>"
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {rsi_color};border-radius:6px;padding:8px;font-size:0.85em;">
        📊 RSI {rsi_val:.1f}<br/><span style="color:{rsi_color};">{rsi_label}</span>{atr_str}
        </div>""", unsafe_allow_html=True)

    with col_dxy:
        # DXY bias estimado desde tendencia del oro (correlación inversa)
        dxy_bias, dxy_color = _get_dxy_bias(df_ind)
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {dxy_color};border-radius:6px;padding:8px;font-size:0.85em;">
        💵 DXY (est.)<br/><span style="color:{dxy_color};">{dxy_bias}</span>
        </div>""", unsafe_allow_html=True)

    with col_event:
        next_evt = _get_next_event_str()
        evt_color = "#ff4444" if "h" in next_evt and any(
            x in next_evt for x in ["0h", "1h", "2h"]) else "#ffaa00"
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {evt_color};border-radius:6px;padding:8px;font-size:0.85em;">
        ⚠️ Próx. evento<br/><span style="color:{evt_color};font-size:0.9em;">{next_evt}</span>
        </div>""", unsafe_allow_html=True)

    # ────── FILA 4: Nivel de Alerta ──────
    st.markdown("### Estado del Clima de Trading")
    warning_level = _get_warning_level(df_ind, results)

    if warning_level == "DANGER":
        st.error("🚨 **PELIGRO** — Clima de trading adverso. Considere reducir exposición o no operar.")
    elif warning_level == "CAUTION":
        st.warning("⚠️ **PRECAUCIÓN** — Condiciones moderadas. Use stops ajustados.")
    else:
        st.success("✅ **CLIMA FAVORABLE** — Condiciones óptimas para operar.")


# ─────────────────────────────────────────────────────────────
# Datos del gráfico según período — granularidad adaptativa
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def _fetch_period_data(period_key: str, mode: str) -> pd.DataFrame:
    """
    Descarga datos con el intervalo adecuado para cada período:
      24H → 5m  | 3D → 15m  | 1S → 30m
      2S  → 1h  | 1M → 1h   | 3M → 1d  | MAX → 1d
    En modo mock genera velas sintéticas con la misma granularidad.
    """
    CFG = {
        "24H": ("1d",   "5m",  288),
        "3D":  ("5d",   "15m", 288),
        "1S":  ("7d",   "30m", 336),
        "2S":  ("14d",  "1h",  336),
        "1M":  ("1mo",  "1h",  720),
        "3M":  ("3mo",  "1d",  90),
        "MAX": ("1y",   "1d",  365),
    }
    yf_period, yf_interval, n_mock = CFG.get(period_key, ("1mo", "1h", 720))

    # ── Modo live: yfinance ────────────────────────────────────
    if mode == "live":
        try:
            import yfinance as yf
            df = yf.download("GC=F", period=yf_period, interval=yf_interval,
                             progress=False, auto_adjust=True)
            if df is not None and len(df) > 5:
                df.index = pd.to_datetime(df.index)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.rename(columns={"Open":"Open","High":"High",
                                        "Low":"Low","Close":"Close","Volume":"Volume"})
                return df[["Open","High","Low","Close","Volume"]].dropna()
        except Exception:
            pass

    # ── Fallback mock: paseo aleatorio con granularidad correcta ──
    import numpy as np
    freq_map = {"5m":"5min","15m":"15min","30m":"30min","1h":"h","1d":"D"}
    freq  = freq_map.get(yf_interval, "h")
    base  = 3300.0
    vol   = {"5m":0.0008,"15m":0.0012,"30m":0.0015,"1h":0.002,"1d":0.008}.get(yf_interval, 0.002)
    rng   = np.random.default_rng(42)
    idx   = pd.date_range(end=pd.Timestamp.now(), periods=n_mock, freq=freq)
    rets  = rng.normal(0.0001, vol, n_mock)
    close = base * np.exp(np.cumsum(rets))
    hi    = close * (1 + rng.uniform(0, vol * 1.5, n_mock))
    lo    = close * (1 - rng.uniform(0, vol * 1.5, n_mock))
    op    = np.roll(close, 1); op[0] = close[0]
    return pd.DataFrame({"Open":op,"High":hi,"Low":lo,"Close":close,
                         "Volume":rng.integers(100,5000,n_mock).astype(float)}, index=idx)


# ─────────────────────────────────────────────────────────────
# Gráfico de precios + señales
# ─────────────────────────────────────────────────────────────
def _render_price_chart(data: dict):
    import streamlit.components.v1 as components

    st.markdown("### 📉 Gráfico de precios y señales")

    # ── Selector de intervalo ─────────────────────────────────
    INTERVALS = {
        "1m": ("1 minuto",  "1"),
        "5m": ("5 minutos", "5"),
        "15m":("15 min",    "15"),
        "1h": ("1 hora",    "60"),
        "4h": ("4 horas",   "240"),
        "D":  ("Diario",    "D"),
        "W":  ("Semanal",   "W"),
    }

    if "tv_interval" not in st.session_state:
        st.session_state["tv_interval"] = "1h"

    btn_cols = st.columns(len(INTERVALS))
    for i, (key, (label, _)) in enumerate(INTERVALS.items()):
        with btn_cols[i]:
            is_active = st.session_state["tv_interval"] == key
            if st.button(label, key=f"tv_btn_{key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["tv_interval"] = key
                st.rerun()

    tv_interval = INTERVALS[st.session_state["tv_interval"]][1]

    # ── Widget de TradingView ─────────────────────────────────
    tv_html = f"""
    <div id="tv_chart_container" style="height:620px;">
      <div class="tradingview-widget-container" style="height:100%;width:100%;">
        <div id="tradingview_gold" style="height:100%;width:100%;"></div>
        <script type="text/javascript"
          src="https://s3.tradingview.com/tv.js"></script>
        <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "OANDA:XAUUSD",
          "interval": "{tv_interval}",
          "timezone": "Asia/Riyadh",
          "theme": "dark",
          "style": "1",
          "locale": "ar",
          "toolbar_bg": "#0e1117",
          "enable_publishing": false,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "save_image": true,
          "container_id": "tradingview_gold",
          "studies": [
            "MASimple@tv-basicstudies",
            "RSI@tv-basicstudies",
            "MACD@tv-basicstudies"
          ],
          "show_popup_button": true,
          "popup_width": "1000",
          "popup_height": "650"
        }});
        </script>
      </div>
    </div>
    """
    components.html(tv_html, height=640, scrolling=False)

    # ────── SECCIÓN: Niveles Clave ──────
    st.markdown("### 📍 Niveles Clave XAU/USD")

    sr_levels  = _calculate_support_resistance(data["df_ind"])
    last_price = float(data["df_ind"]["Close"].iloc[-1])

    # ── Construir HTML completo y renderizar con components.html ──
    # (evita que st.markdown() stripee display:flex y otros CSS)
    rows_html = ""

    for i, lk in enumerate(["r3", "r2", "r1"], 1):
        if sr_levels.get(lk) is not None:
            p   = sr_levels[lk]
            pct = (p - last_price) / last_price * 100
            rows_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        border-bottom:1px solid #333;padding:9px 6px;">
              <span style="color:#ff7070;font-weight:600;min-width:130px;">🔴 Resistencia {i}</span>
              <span style="color:#e0c97f;font-weight:700;">${p:,.2f}</span>
              <span style="color:#ff9090;">▲ {pct:+.2f}%</span>
            </div>"""

    rows_html += f"""
    <div style="padding:12px 6px;text-align:center;font-size:1.05em;font-weight:bold;
                color:#ffd700;border-top:2px solid #ffd700;border-bottom:2px solid #ffd700;
                margin:6px 0;letter-spacing:1px;">
      ── PRECIO ACTUAL: ${last_price:,.2f} ──
    </div>"""

    for i, lk in enumerate(["s1", "s2", "s3"], 1):
        if sr_levels.get(lk) is not None:
            p   = sr_levels[lk]
            pct = (p - last_price) / last_price * 100
            rows_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        border-bottom:1px solid #333;padding:9px 6px;">
              <span style="color:#70ff90;font-weight:600;min-width:130px;">🟢 Soporte {i}</span>
              <span style="color:#e0c97f;font-weight:700;">${p:,.2f}</span>
              <span style="color:#90ff90;">▼ {pct:+.2f}%</span>
            </div>"""

    if sr_levels.get("pp") is not None:
        rows_html += f"""
        <div style="padding:8px 6px;text-align:center;font-size:0.88em;color:#888;">
          Pivot Point: ${sr_levels['pp']:,.2f}
        </div>"""

    full_html = f"""
    <div style="background:#0e1117;border:2px solid #ffd700;border-radius:10px;
                padding:14px 18px;font-family:sans-serif;font-size:14px;">
      <div style="text-align:center;margin-bottom:12px;color:#aaa;
                  font-size:0.9em;font-weight:bold;letter-spacing:1px;">
        ANÁLISIS DE SOPORTE &amp; RESISTENCIA
      </div>
      {rows_html}
    </div>"""

    components.html(full_html, height=320, scrolling=False)



# ─────────────────────────────────────────────────────────────
# Curva de capital
# ─────────────────────────────────────────────────────────────
def _render_equity_curve(data: dict):
    bt_res = data["bt_res"]
    df_ind = data["df_ind"]

    st.markdown("### 💹 Curva de capital vs Buy & Hold")

    equity = bt_res.equity_curve
    if equity.empty:
        st.info("No hay operaciones en el período de prueba.")
        return

    test_prices = df_ind.loc[equity.index, "Close"] if equity.index[0] in df_ind.index else None
    if test_prices is not None:
        bnh = bt_res.initial_capital * (test_prices / test_prices.iloc[0])
    else:
        bnh = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Ensemble Strategy", fill="tozeroy",
        line=dict(color="#ffd700", width=2),
        fillcolor="rgba(255,215,0,0.1)",
    ))
    if bnh is not None:
        fig.add_trace(go.Scatter(
            x=bnh.index, y=bnh.values,
            name="Buy & Hold", line=dict(color="#4488ff", width=1.5, dash="dash"),
        ))

    fig.update_layout(
        height=320, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        yaxis_title="Valor de cartera ($)",
        xaxis_title="Fecha",
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Importancia de características
# ─────────────────────────────────────────────────────────────
def _render_feature_importance(data: dict):
    st.markdown("### 🔬 Importancia de características (Random Forest)")

    rf_model = data["ensemble"]._rf
    if not rf_model.is_trained():
        st.warning("El modelo aún no fue entrenado.")
        return

    importance = rf_model.metrics.feature_importance if rf_model.metrics else {}
    if not importance:
        try:
            from feature_engineer import FEATURE_COLUMNS
            feat_cols = rf_model._feature_names
            imp_vals  = rf_model.model.feature_importances_
            importance = dict(zip(feat_cols, imp_vals.round(4)))
        except Exception:
            st.warning("No hay datos de importancia disponibles.")
            return

    # Top 15 características
    top_n  = 15
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feat_names = [x[0] for x in sorted_imp]
    feat_vals  = [x[1] for x in sorted_imp]

    # Colorear por tipo
    color_map = {
        "RSI": "#a78bfa", "MACD": "#60a5fa", "EMA": "#fbbf24",
        "BB": "#34d399", "ATR": "#f87171", "OBV": "#fb923c",
        "VWAP": "#38bdf8", "Confluence": "#f472b6",
    }
    bar_colors = []
    for f in feat_names:
        matched = "#ffd700"
        for key, col in color_map.items():
            if key.lower() in f.lower():
                matched = col
                break
        bar_colors.append(matched)

    fig = go.Figure(go.Bar(
        x=feat_vals, y=feat_names, orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in feat_vals],
        textposition="outside",
    ))
    fig.update_layout(
        height=420, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis_title="Importancia relativa",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Métricas de rendimiento
# ─────────────────────────────────────────────────────────────
def _render_performance_metrics(data: dict):
    st.markdown("### 📊 Métricas de rendimiento")
    bt_res = data["bt_res"]

    metrics = {
        "Rendimiento total":         f"{bt_res.total_return_pct:+.2%}",
        "Rendimiento anual":        f"{bt_res.annual_return_pct:+.2%}",
        "Alpha vs Buy&Hold": f"{bt_res.alpha:+.2%}",
        "Sharpe Ratio":         f"{bt_res.sharpe_ratio:.3f}",
        "Max Drawdown":         f"{bt_res.max_drawdown_pct:.2%}",
        "Calmar Ratio":         f"{bt_res.calmar_ratio:.3f}",
        "Win Rate":             f"{bt_res.win_rate:.1%}",
        "Profit Factor":        f"{bt_res.profit_factor:.2f}",
        "Total operaciones":       str(bt_res.total_trades),
    }

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (k, v) in enumerate(metrics.items()):
        with cols[i % 3]:
            # تحديد اللون
            is_positive = (
                "+" in v or
                (any(c.isdigit() for c in v) and
                 float(v.replace("%","").replace("+","").replace(",","")) > 0
                 if v not in ("0.000", "0.00%", "0") else False)
            )
            delta_color = "normal" if is_positive else "inverse"
            st.metric(k, v)


# ─────────────────────────────────────────────────────────────
# Registro de operaciones
# ─────────────────────────────────────────────────────────────
def _render_trades_table(data: dict):
    trades = data["trades_df"]
    st.markdown("### 📋 Registro de operaciones recientes")

    if trades.empty:
        st.info("No hay operaciones cerradas aún.")
        return

    # عرض آخر 20 صفقة
    display = trades.tail(20).copy()
    if "return_pct" in display.columns:
        display["return_pct"] = display["return_pct"].map("{:+.2%}".format)
    if "profit_loss" in display.columns:
        display["profit_loss"] = display["profit_loss"].map("${:+,.2f}".format)

    st.dataframe(display, use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────────
# Pestaña de Noticias y Sentiment
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner="📰 Analizando noticias del mercado...")
def _fetch_news_cached(newsapi_key: str, av_key: str) -> dict:
    """Obtiene y analiza noticias — con caché de 30 minutos."""
    if not _NEWS_AVAILABLE:
        return {}
    cfg    = NewsConfig(newsapi_key=newsapi_key, alphavantage_key=av_key)
    nf     = GoldNewsSentimentFilter(cfg)
    result = nf.analyze()
    return result.to_dict() | {"_result_obj": result}


def _render_news_tab(newsapi_key: str = "", av_key: str = ""):
    """Renderiza la pestaña de análisis de noticias y sentiment."""
    st.markdown("### 📰 Análisis de Noticias — Sentiment del Mercado")

    if not _NEWS_AVAILABLE:
        st.error("❌ news_filter.py no encontrado en el proyecto.")
        return

    col_refresh, col_info = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Actualizar noticias", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_info:
        st.caption("Actualización automática cada 30 min · presiona 🔄 para forzar actualización")

    # ── Diagnóstico de fuentes activas ──────────────────────────────────
    with st.expander("🔌 Estado de fuentes de noticias", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            if newsapi_key:
                st.success("✅ NewsAPI — **Activa**")
            else:
                st.warning("⚠️ NewsAPI — Sin clave")
        with c2:
            if av_key:
                st.success("✅ Alpha Vantage — **Activa**")
            else:
                st.warning("⚠️ Alpha Vantage — Sin clave")
        with c3:
            st.info("📡 RSS Gratuito — **Siempre activo**")
        if not newsapi_key and not av_key:
            st.info(
                "💡 Agrega tus claves en **config.env** para más noticias:\n"
                "`NEWS_API_KEY=tu_clave`  `ALPHAVANTAGE_KEY=tu_clave`"
            )

    # Obtener datos (con caché)
    try:
        cached = _fetch_news_cached(newsapi_key, av_key)
    except Exception as e:
        st.error(f"Error al cargar noticias: {e}")
        return

    if not cached:
        st.warning("No se pudieron obtener noticias. Intenta más tarde.")
        return

    result: SentimentResult = cached.get("_result_obj")
    if result is None:
        st.warning("Datos de noticias no disponibles.")
        return

    # ── Timestamp de última obtención ────────────────────────
    next_update = result.timestamp + timedelta(minutes=30)
    remaining   = max(0, int((next_update - datetime.now()).total_seconds() / 60))
    fetch_str   = result.timestamp.strftime("%H:%M:%S")
    if remaining > 0:
        st.info(f"⏱️ Última obtención: **{fetch_str}** · próxima actualización en **{remaining} min**")
    else:
        st.warning(f"⏱️ Última obtención: **{fetch_str}** · ⚡ datos listos para refrescar — presiona 🔄")

    # ── Tarjetas de métricas ──────────────────────────────────
    st.markdown("---")

    # Conteo de artículos por fuente
    all_articles = result.all_articles if result else []
    from collections import Counter
    source_counts = Counter(
        ("AlphaVantage" if "alpha" in a.source.lower() or "vantage" in a.source.lower()
         else "NewsAPI" if any(x in a.source.lower() for x in ["newsapi", "crypto briefing", "times", "globe", "financial", "khabar"])
         else "RSS")
        for a in all_articles
    )
    av_n  = source_counts.get("AlphaVantage", 0)
    na_n  = source_counts.get("NewsAPI", 0)
    rss_n = source_counts.get("RSS", 0)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        label = "Alpha Vantage ✅" if av_key else "Alpha Vantage ⚠️"
        st.metric(label, f"{av_n} artículos")
    with sc2:
        label = "NewsAPI ✅" if newsapi_key else "NewsAPI ⚠️"
        st.metric(label, f"{na_n} artículos")
    with sc3:
        st.metric("RSS 📡", f"{rss_n} artículos")

    col1, col2, col3, col4, col5 = st.columns(5)

    score = result.aggregate_score
    score_color = "normal" if score > 0 else "inverse"

    with col1:
        st.metric("📊 Score Sentiment",
                  f"{score:+.2f}",
                  delta=result.sentiment_label,
                  delta_color=score_color)
    with col2:
        st.metric("📰 Total Artículos", str(result.total_articles))
    with col3:
        st.metric("🟢 Alcistas", str(result.bullish_count))
    with col4:
        st.metric("🔴 Bajistas", str(result.bearish_count))
    with col5:
        trump_label = f"🎯 {result.trump_news_count}"
        st.metric("Trump News", trump_label,
                  delta="⚡ Alto impacto" if result.trump_news_count >= 3 else "")

    # ── Alertas especiales ────────────────────────────────────
    if result.should_freeze_trading():
        st.error(
            "🚨 **EVENTO EXTREMO** — El sistema recomienda NO operar ahora. "
            f"Score: {score:+.2f} | Noticias Trump: {result.trump_news_count}"
        )
    elif result.high_volatility:
        st.warning(
            "⚡ **Alta volatilidad** — Reducir tamaño de posiciones. "
            "El mercado está muy sensible a noticias."
        )
    elif abs(score) >= 0.25:
        direction = "alcista 🟢" if score > 0 else "bajista 🔴"
        st.info(f"📡 Sentiment {direction} — El mercado favorece al oro {'al alza' if score > 0 else 'a la baja'}.")

    # ── Gauge visual del score ────────────────────────────────
    st.markdown("#### 🎯 Barómetro de Sentiment")
    fig_gauge = _build_sentiment_gauge(score)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Titulares con links — todos los artículos ────────────
    all_arts = sorted(result.all_articles,
                      key=lambda x: abs(x.weighted_score), reverse=True)
    total_arts = len(all_arts)

    st.markdown(f"#### 📋 Titulares ({total_arts} noticias)")

    # Control: cuántos mostrar
    if "news_show_count" not in st.session_state:
        st.session_state["news_show_count"] = 10

    show_n = st.session_state["news_show_count"]
    arts_to_show = all_arts[:show_n]

    if not arts_to_show:
        st.info("No hay titulares disponibles en este momento.")

    for art in arts_to_show:
            score_val = art.weighted_score
            if score_val >= 0.2:
                border = "#00cc66"
                icon   = "🟢"
            elif score_val <= -0.2:
                border = "#ff4444"
                icon   = "🔴"
            else:
                border = "#ffaa00"
                icon   = "⚪"

            trump_badge  = " &nbsp;🎯 <b>Trump</b>" if art.is_trump else ""
            keywords_str = " &nbsp;·&nbsp; 🏷️ " + ", ".join(art.keywords_hit[:3]) if art.keywords_hit else ""
            date_str     = art.published.strftime("%Y-%m-%d %H:%M")
            score_color  = "#00cc66" if score_val >= 0 else "#ff4444"

            # Título: link si hay URL, texto plano si no
            if art.url:
                title_html = (f'<a href="{art.url}" target="_blank" '
                              f'style="color:#e0c97f;text-decoration:none;font-size:14px;font-weight:600;">'
                              f'{art.title[:100]}{"..." if len(art.title) > 100 else ""}</a>')
            else:
                title_html = (f'<span style="color:#e0c97f;font-size:14px;font-weight:600;">'
                              f'{art.title[:100]}{"..." if len(art.title) > 100 else ""}</span>')

            st.markdown(
                f"""<div style="border-left:4px solid {border};padding:10px 14px;
                    margin-bottom:10px;background:#1a1a2e;border-radius:6px;line-height:1.6;">
                    {icon} {title_html}
                    <div style="color:#999;font-size:12px;margin-top:5px;">
                    📰 {art.source} &nbsp;·&nbsp; 🕐 {date_str} &nbsp;·&nbsp;
                    Score: <b style="color:{score_color}">{score_val:+.2f}</b>
                    {trump_badge}{keywords_str}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
    # ── Botones ver más / ver menos ──────────────────────────
    if total_arts > 0:
        col_more, col_less, col_all = st.columns([1, 1, 1])
        with col_more:
            if show_n < total_arts:
                if st.button(f"⬇️ Ver más ({min(show_n+10, total_arts)}/{total_arts})",
                             use_container_width=True):
                    st.session_state["news_show_count"] = min(show_n + 10, total_arts)
                    st.rerun()
        with col_less:
            if show_n > 10:
                if st.button("⬆️ Ver menos", use_container_width=True):
                    st.session_state["news_show_count"] = 10
                    st.rerun()
        with col_all:
            if show_n < total_arts:
                if st.button(f"📋 Ver todas ({total_arts})", use_container_width=True):
                    st.session_state["news_show_count"] = total_arts
                    st.rerun()

    # ── Distribución de sentiment ─────────────────────────────
    if result.total_articles > 0:
        st.markdown("#### 📊 Distribución de Artículos")
        neutral_count = result.total_articles - result.bullish_count - result.bearish_count
        col_a, col_b = st.columns(2)
        with col_a:
            import plotly.graph_objects as go
            fig_pie = go.Figure(go.Pie(
                labels=["Alcistas 🟢", "Bajistas 🔴", "Neutros ⚪"],
                values=[result.bullish_count, result.bearish_count,
                        max(0, neutral_count)],
                hole=0.45,
                marker_colors=["#00cc66", "#ff4444", "#aaaaaa"],
            ))
            fig_pie.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0e1117",
                margin=dict(t=20, b=20, l=20, r=20),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_b:
            st.markdown("##### Fuentes utilizadas")
            for src in result.sources_used:
                st.markdown(f"- {src}")
            st.markdown("---")
            st.markdown(f"**Última actualización:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**Noticias Trump:** {result.trump_news_count}")
            st.markdown(f"**Alta volatilidad:** {'Sí ⚡' if result.high_volatility else 'No ✅'}")

    # ── Configuración de API keys — solo mostrar si faltan claves ────
    if not newsapi_key or not av_key:
        missing = []
        if not newsapi_key:  missing.append("**NewsAPI** — [newsapi.org](https://newsapi.org) — 100 req/día gratis")
        if not av_key:       missing.append("**Alpha Vantage** — [alphavantage.co](https://www.alphavantage.co) — 500 req/día gratis")

        with st.expander("⚙️ Mejorar calidad — agregar API Keys"):
            st.info("Actualmente usas solo RSS gratuito. Agrega estas claves en **config.env** para más noticias:")
            for m in missing:
                st.markdown(f"- 🔑 {m}")
            st.code("NEWS_API_KEY=tu_clave_aqui\nALPHAVANTAGE_KEY=tu_clave_aqui", language="bash")


def _build_sentiment_gauge(score: float):
    """Construye un gráfico de gauge para el score de sentiment."""
    import plotly.graph_objects as go

    # Redondear a 2 decimales para evitar flotantes largos en el gauge
    score = round(float(score), 2)

    # Color según score
    if score >= 0.25:
        bar_color = "#00cc66"
    elif score <= -0.25:
        bar_color = "#ff4444"
    else:
        bar_color = "#ffaa00"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Sentiment del Oro", "font": {"color": "white", "size": 16}},
        delta={"reference": 0,
               "valueformat": "+.2f",
               "increasing": {"color": "#00cc66"},
               "decreasing": {"color": "#ff4444"}},
        number={"font": {"color": "white", "size": 48}, "valueformat": "+.2f"},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1,
                     "tickcolor": "white", "tickfont": {"color": "white"}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "#1a1a2e",
            "borderwidth": 2,
            "bordercolor": "#ffd700",
            "steps": [
                {"range": [-1.0, -0.25], "color": "#3d0000"},
                {"range": [-0.25, 0.25], "color": "#1a1a2e"},
                {"range": [0.25, 1.0],   "color": "#003d1a"},
            ],
            "threshold": {
                "line": {"color": "#ffd700", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=280, template="plotly_dark",
        paper_bgcolor="#0e1117",
        font={"color": "white"},
        margin=dict(t=50, b=20, l=30, r=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Enviar señal a MetaTrader y mostrar estado
# ─────────────────────────────────────────────────────────────
def _send_and_render_mt_signal(data: dict, mt_cfg):
    """Envía la última señal a MT a través del puente y muestra el estado."""
    if not _MT_AVAILABLE or mt_cfg is None:
        return

    results = data["results"]
    df_ind  = data["df_ind"]

    last_sig    = int(results.signals.iloc[-1])
    last_conf   = float(results.confidence.iloc[-1])
    last_regime = int(results.regimes.iloc[-1])
    last_price  = float(df_ind["Close"].iloc[-1])
    atr_val     = get_atr_from_df(df_ind)

    # Crear el puente y enviar la señal
    bridge = MetaTraderBridge(mt_cfg)
    sig_obj = bridge.send_signal(
        signal=last_sig,
        confidence=last_conf,
        regime=last_regime,
        price=last_price,
        atr=atr_val,
    )

    # ── Mostrar tarjeta de estado ────────────────────────────────────
    st.markdown("### 🔗 Estado de MetaTrader Bridge")
    col1, col2, col3, col4 = st.columns(4)

    file_path  = bridge.get_signal_file_path()
    file_exists = os.path.isfile(file_path)

    with col1:
        if file_exists:
            st.success("✅ Archivo de señal escrito")
            st.caption(file_path.split("\\")[-1])
        else:
            st.error("❌ Archivo no escrito")

    with col2:
        if sig_obj:
            action_color = {"BUY": "🟢", "SELL": "🔴", "NEUTRAL": "⚪"}.get(sig_obj.action, "⚪")
            st.metric("Señal enviada", f"{action_color} {sig_obj.action}")
        else:
            st.metric("Señal enviada", "⏭ Ignorada")

    with col3:
        st.metric("SL", f"${bridge._last_signal.sl_price:,.2f}" if bridge._last_signal else "—")

    with col4:
        st.metric("TP", f"${bridge._last_signal.tp_price:,.2f}" if bridge._last_signal else "—")

    if sig_obj is None:
        st.info(f"💡 Señal no enviada — confianza {last_conf:.0%} inferior a mínimo {mt_cfg.min_confidence:.0%} o misma señal anterior.")
    else:
        st.success(f"📤 Señal enviada `{sig_obj.action}` @ ${last_price:,.2f} | "
                   f"SL: ${sig_obj.sl_price:,.2f} | TP: ${sig_obj.tp_price:,.2f} | "
                   f"Confianza: {last_conf:.0%}")

    with st.expander("📄 Ver contenido de gold_signal.json"):
        if file_exists:
            try:
                import json
                with open(file_path, "r", encoding="utf-8") as f:
                    st.json(json.load(f))
            except Exception as e:
                st.error(f"Error al leer archivo: {e}")
        else:
            st.warning("El archivo aún no existe.")


# ─────────────────────────────────────────────────────────────
# Barra de señales recientes
# ─────────────────────────────────────────────────────────────
def _render_recent_signals(data: dict):
    st.markdown("### 📡 Últimas 10 señales")
    results = data["results"]
    df_ind  = data["df_ind"]

    n = 10
    recent = results.signals.iloc[-n:]
    recent_conf = results.confidence.iloc[-n:]
    recent_reg  = results.regimes.iloc[-n:]

    rows = []
    for idx in recent.index:
        sig    = int(recent.loc[idx])
        conf   = float(recent_conf.loc[idx]) if idx in recent_conf.index else 0
        regime = int(recent_reg.loc[idx]) if idx in recent_reg.index else 1
        price  = float(df_ind.loc[idx, "Close"]) if idx in df_ind.index else 0
        rows.append({
            "Fecha":     str(idx)[:10],
            "Precio":       f"${price:,.2f}",
            "Señal":     {1: "Compra 📈", -1: "Venta 📉", 0: "Neutro ⏸"}.get(sig, "—"),
            "Confianza":       f"{conf:.0%}",
            "Régimen":      {0: "Bajista 🔴", 1: "Lateral ⚪", 2: "Alcista 🟢"}.get(regime, "—"),
        })

    df_sig = pd.DataFrame(rows).iloc[::-1]
    st.dataframe(df_sig, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# Pestaña de rendimiento de señales
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="📈 Cargando datos de rendimiento...")
def _load_performance_data(tracker_path: str) -> tuple:
    """Carga datos de rendimiento desde el archivo JSON de historial."""
    # Intenta cargar el archivo JSON de señales
    try:
        from pathlib import Path
        path_obj = Path(tracker_path)
        if not path_obj.exists():
            return None, None

        with open(path_obj, "r", encoding="utf-8") as f:
            signals_data = json.load(f)

        # Convertir a lista de SignalRecord si está disponible
        if _PERF_AVAILABLE:
            from performance_tracker import SignalRecord
            signals_list = [SignalRecord.from_dict(s) for s in signals_data]
        else:
            signals_list = None

        return signals_data, signals_list
    except Exception as e:
        logging.warning(f"No se pudo cargar {tracker_path}: {e}")
        return None, None


def _render_performance_tab(tracker_path: str = "signals_history.json"):
    """
    Renderiza la pestaña completa de rendimiento de señales.
    Incluye: tarjetas de métricas, gráfico PnL, distribución de resultados,
    tabla de últimas señales, Kelly Criterion y reentrenamiento.
    """

    if not _PERF_AVAILABLE:
        st.warning(
            "⚠️ **performance_tracker.py no encontrado**\n\n"
            "Para habilitar esta pestaña, asegúrate de que el módulo de rastreo esté disponible."
        )
        return

    # Cargar datos de rendimiento
    signals_raw, signals_list = _load_performance_data(tracker_path)

    if signals_raw is None or len(signals_raw) == 0:
        st.info(
            "📊 **Las señales se registran automáticamente**\n\n"
            "Cuando el sistema genere señales de trading y cierre posiciones, "
            "los resultados aparecerán aquí con estadísticas detalladas."
        )
        return

    # ─────────────────────────────────────────────────────────────
    # 1. Calcular estadísticas principales
    # ─────────────────────────────────────────────────────────────
    closed_signals = [s for s in signals_list if s.outcome != "PENDING"]
    executed_signals = [s for s in signals_list if s.action != "NEUTRAL"]

    if len(closed_signals) > 0:
        winners = [s for s in closed_signals if s.pnl_usd > 0]
        losers = [s for s in closed_signals if s.pnl_usd < 0]

        win_rate = len(winners) / len(closed_signals) if closed_signals else 0.0
        total_pnl = sum(s.pnl_usd for s in closed_signals)

        # Profit Factor = ganancia bruta / pérdida bruta
        gross_profit = sum(s.pnl_usd for s in winners) if winners else 0.0
        gross_loss = abs(sum(s.pnl_usd for s in losers)) if losers else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (1.0 if gross_profit > 0 else 0.0)

        # Max Drawdown en pips
        cumulative_pnl = [0]
        for s in closed_signals:
            cumulative_pnl.append(cumulative_pnl[-1] + s.pnl_pips)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - np.array(cumulative_pnl)
        max_drawdown_pips = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Tarjetas de métricas principales
        st.markdown("### 📊 Estadísticas Principales")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            color_wr = "normal" if win_rate >= 0.5 else "inverse"
            st.metric("📊 Win Rate", f"{win_rate:.1%}", delta_color=color_wr)

        with col2:
            color_pnl = "normal" if total_pnl >= 0 else "inverse"
            st.metric("💰 PnL Total", f"${total_pnl:+,.2f}", delta_color=color_pnl)

        with col3:
            color_pf = "normal" if profit_factor > 1.0 else "inverse"
            st.metric("📈 Profit Factor", f"{profit_factor:.2f}", delta_color=color_pf)

        with col4:
            st.metric("🎯 Operaciones", str(len(closed_signals)))

        with col5:
            st.metric("📉 Max Drawdown", f"{max_drawdown_pips:.1f} pips")

        # ─────────────────────────────────────────────────────────────
        # 2. Gráfico PnL Acumulativo
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 💹 Curva PnL Acumulativo")

        cumulative_pnl_list = []
        dates = []
        cumulative = 0.0
        for s in closed_signals:
            cumulative += s.pnl_usd
            cumulative_pnl_list.append(cumulative)
            dates.append(s.exit_time if s.exit_time else s.timestamp)

        if len(cumulative_pnl_list) > 0:
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=dates,
                y=cumulative_pnl_list,
                mode="lines+markers",
                name="PnL Acumulativo",
                line=dict(color="#ffd700", width=2),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor="rgba(255, 215, 0, 0.1)",
            ))

            # Colorear área según positivo/negativo
            fig_pnl.update_layout(
                height=320,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="white"),
                xaxis_title="Fecha de Salida",
                yaxis_title="PnL Acumulativo ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            st.info("No hay suficientes datos para el gráfico de PnL.")

        # ─────────────────────────────────────────────────────────────
        # 3. Distribución de Resultados (Pie Chart)
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 🎯 Distribución de Resultados")

        tp_count = len([s for s in closed_signals if s.outcome == "TP_HIT"])
        sl_count = len([s for s in closed_signals if s.outcome == "SL_HIT"])
        manual_count = len([s for s in closed_signals if s.outcome == "MANUAL_CLOSE"])
        pending_count = len([s for s in signals_list if s.outcome == "PENDING"])

        fig_dist = go.Figure(go.Pie(
            labels=["TP Hit 🟢", "SL Hit 🔴", "Manual Close 📌", "Pending ⏳"],
            values=[tp_count, sl_count, manual_count, pending_count],
            hole=0.4,
            marker_colors=["#00cc66", "#ff4444", "#ffaa00", "#999999"],
        ))
        fig_dist.update_layout(
            height=280,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # ─────────────────────────────────────────────────────────────
        # 4. Tabla de Últimas 10 Señales
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 📋 Últimas Señales Cerradas")

        recent_closed = sorted(closed_signals, key=lambda x: x.exit_time or x.timestamp, reverse=True)[:10]

        if recent_closed:
            table_data = []
            for sig in recent_closed:
                # Colorear fila según resultado
                row_color = "#00cc66" if sig.pnl_usd > 0 else "#ff4444"

                table_data.append({
                    "Fecha": (sig.exit_time or sig.timestamp).strftime("%Y-%m-%d"),
                    "Acción": sig.action,
                    "Entrada": f"${sig.entry_price:,.2f}",
                    "SL": f"${sig.sl_price:,.2f}",
                    "TP": f"${sig.tp_price:,.2f}",
                    "Resultado": sig.outcome.replace("_", " "),
                    "PnL (pips)": f"{sig.pnl_pips:+.1f}",
                    "PnL ($)": f"${sig.pnl_usd:+,.2f}",
                    "Confianza": f"{sig.confidence:.0%}",
                })

            df_display = pd.DataFrame(table_data)
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("No hay señales cerradas aún.")

        # ─────────────────────────────────────────────────────────────
        # 5. Kelly Criterion y Recomendación de Tamaño
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 💡 Recomendación de Tamaño de Posición")

        # Calcular Kelly Fraction
        if len(closed_signals) > 0 and profit_factor > 0:
            win_ratio = win_rate
            loss_ratio = 1.0 - win_rate

            # Kelly: f* = (bp - q) / b  donde b=reward/risk, p=win_rate, q=loss_rate
            if loss_ratio > 0:
                avg_win = np.mean([s.pnl_pips for s in winners]) if winners else 1.0
                avg_loss = abs(np.mean([s.pnl_pips for s in losers])) if losers else 1.0
                b = avg_win / avg_loss if avg_loss > 0 else 1.0

                kelly_frac = (b * win_ratio - loss_ratio) / b
                kelly_frac = max(0.0, min(kelly_frac, 0.25))  # Limitar entre 0% y 25%
            else:
                kelly_frac = 0.0
        else:
            kelly_frac = 0.0

        col_kelly, col_slider = st.columns([2, 1])

        with col_kelly:
            # Capital slider
            if "trading_balance" not in st.session_state:
                st.session_state["trading_balance"] = 10000

            balance = st.slider(
                "💰 Balance de trading",
                min_value=1000,
                max_value=100000,
                value=st.session_state.get("trading_balance", 10000),
                step=1000,
                format="$%d",
            )
            st.session_state["trading_balance"] = balance

            # Calcular lote recomendado
            risk_per_trade = balance * kelly_frac * 0.02  # 2% de riesgo por trade
            recommended_lot = risk_per_trade / 100.0  # Aproximado para oro
            recommended_lot = max(0.01, round(recommended_lot, 2))

            st.markdown(f"""
            **Kelly Fraction:** `{kelly_frac:.2%}`  |  **Lote Recomendado:** `{recommended_lot:.2f}`

            *(para balance ${balance:,})*
            """)

        # ─────────────────────────────────────────────────────────────
        # 6. Reentrenamiento del Modelo
        # ─────────────────────────────────────────────────────────────
        st.markdown("### 🔄 Reentrenamiento del Modelo")

        with st.expander("Reentrenar con últimos datos"):
            st.markdown(
                "El modelo se reentrenará con todos los datos disponibles, "
                "incluyendo las nuevas señales y resultados registrados."
            )

            if "last_training_time" not in st.session_state:
                st.session_state["last_training_time"] = "Nunca"

            st.caption(f"Última actualización: {st.session_state['last_training_time']}")

            col_btn, col_info = st.columns([1, 3])
            with col_btn:
                if st.button("🔄 Reentrenar Modelo Ahora", use_container_width=True):
                    st.cache_resource.clear()
                    st.session_state["last_training_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("✅ Caché limpiado. El modelo se reentrenará en la próxima carga.")
                    st.rerun()

            with col_info:
                st.caption(
                    "⏱️ El reentrenamiento puede tomar algunos segundos "
                    "dependiendo del volumen de datos."
                )

    else:
        st.info(
            "📊 Aún no hay operaciones cerradas.\n\n"
            "Los resultados aparecerán aquí cuando se completen las primeras posiciones."
        )


# ─────────────────────────────────────────────────────────────
# قائمة تدقيق التداول اليومي — Daily Trading Checklist
# ─────────────────────────────────────────────────────────────
def _render_trading_checklist(data: dict, news_score: float = 0.0):
    """
    يعرض قائمة شاملة للتحقق من شروط الدخول اليومي.
    كل شرط يُعطي ✅ أو ❌ مع سبب واضح.
    في النهاية: توصية نهائية (ENTRADA PERMITIDA / ESPERAR).
    """
    df_ind  = data["df_ind"]
    results = data["results"]

    last_sig  = int(results.signals.iloc[-1])
    last_conf = float(results.confidence.iloc[-1])
    rsi_val   = df_ind["RSI"].iloc[-1] if "RSI" in df_ind.columns else 50.0
    atr_val   = df_ind["ATR"].iloc[-1] if "ATR" in df_ind.columns else 0.0
    atr_mean  = df_ind["ATR"].mean()   if "ATR" in df_ind.columns else 0.0

    # ── Sesión activa ──────────────────────────────────────────
    from datetime import timezone
    hour_utc = datetime.now(timezone.utc).hour
    is_active_session = (8 <= hour_utc < 22)        # London + NY + Overlap
    is_overlap        = (13 <= hour_utc < 17)

    # ── Calendario económico ───────────────────────────────────
    warning_level = _get_warning_level(df_ind, results)
    cal_ok = warning_level != "DANGER"

    # ── Alineación MTF (si hay EMA20 en df_ind) ───────────────
    mtf_aligned = False
    mtf_note    = "Sin datos MTF"
    if "EMA20" in df_ind.columns and "EMA50" in df_ind.columns:
        close   = float(df_ind["Close"].iloc[-1])
        ema20   = float(df_ind["EMA20"].iloc[-1])
        ema50   = float(df_ind["EMA50"].iloc[-1])
        if last_sig == 1 and close > ema20 > ema50:
            mtf_aligned = True
            mtf_note    = "Precio > EMA20 > EMA50 — tendencia alcista"
        elif last_sig == -1 and close < ema20 < ema50:
            mtf_aligned = True
            mtf_note    = "Precio < EMA20 < EMA50 — tendencia bajista"
        elif last_sig == 0:
            mtf_aligned = True
            mtf_note    = "Señal NEUTRAL — sin entrada requerida"
        else:
            mtf_note = f"Precio {close:.0f} | EMA20 {ema20:.0f} | EMA50 {ema50:.0f}"
    else:
        mtf_aligned = True    # Sin datos MTF → no bloquear
        mtf_note    = "EMAs no disponibles (no bloquea)"

    # ── RSI no extremo ─────────────────────────────────────────
    rsi_ok   = 25 <= rsi_val <= 75
    rsi_note = f"RSI = {rsi_val:.1f}"

    # ── Confianza suficiente ───────────────────────────────────
    conf_ok   = last_conf >= 0.55
    conf_note = f"Confianza = {last_conf:.0%}"

    # ── Sentiment no adverso ───────────────────────────────────
    # Adverso solo si señal BUY pero sentiment muy bajista, o viceversa
    sent_conflict = (last_sig == 1 and news_score < -0.4) or \
                    (last_sig == -1 and news_score > 0.4)
    sent_ok   = not sent_conflict
    sent_note = f"Score noticias = {news_score:+.2f}"

    # ── Volatilidad razonable ──────────────────────────────────
    vol_ok = True
    vol_note = "Volatilidad normal"
    if atr_mean > 0:
        atr_ratio = atr_val / atr_mean
        if atr_ratio > 2.5:
            vol_ok   = False
            vol_note = f"ATR {atr_ratio:.1f}x sobre la media — volatilidad extrema"
        elif atr_ratio > 1.5:
            vol_note = f"ATR {atr_ratio:.1f}x sobre la media — volatilidad elevada"
        else:
            vol_note = f"ATR {atr_ratio:.1f}x de la media — normal"

    # ── Compilar checks ────────────────────────────────────────
    checks = [
        ("Sesión activa (London/NY/Overlap)",   is_active_session,
         f"Hora UTC: {hour_utc:02d}:xx" + (" — ¡OVERLAP activo! 🔥" if is_overlap else "")),
        ("Sin evento HIGH impact inminente",     cal_ok,
         f"Nivel de riesgo: {warning_level}"),
        ("Alineación tendencia (EMAs)",          mtf_aligned,   mtf_note),
        ("RSI en zona válida (25–75)",           rsi_ok,        rsi_note),
        ("Confianza del modelo ≥ 55%",           conf_ok,       conf_note),
        ("Sentiment no contradice señal",        sent_ok,       sent_note),
        ("Volatilidad no extrema (ATR < 2.5x)",  vol_ok,        vol_note),
    ]

    passed = sum(1 for _, ok, _ in checks if ok)
    total  = len(checks)
    pct    = passed / total

    # ── Renderizar ─────────────────────────────────────────────
    st.markdown("### 📋 Checklist de Trading Diario")

    # Tarjeta de veredicto final
    if pct >= 6/7:
        verdict_bg    = "#003d1a"
        verdict_border= "#00cc66"
        verdict_icon  = "✅"
        verdict_text  = "ENTRADA PERMITIDA"
        verdict_color = "#00ff88"
        verdict_sub   = f"Todas las condiciones favorables ({passed}/{total})"
    elif pct >= 4/7:
        verdict_bg    = "#1a1a00"
        verdict_border= "#ffaa00"
        verdict_icon  = "⚠️"
        verdict_text  = "PROCEDER CON PRECAUCIÓN"
        verdict_color = "#ffdd00"
        verdict_sub   = f"{passed}/{total} condiciones cumplidas — reduce el tamaño de lote"
    else:
        verdict_bg    = "#3d0000"
        verdict_border= "#ff4444"
        verdict_icon  = "🚫"
        verdict_text  = "ESPERAR — NO ENTRAR"
        verdict_color = "#ff4444"
        verdict_sub   = f"Solo {passed}/{total} condiciones cumplidas"

    st.markdown(f"""
    <div style="background:{verdict_bg};border:2px solid {verdict_border};border-radius:12px;
                padding:16px 24px;margin-bottom:16px;display:flex;align-items:center;gap:16px;">
      <div style="font-size:2.5em;">{verdict_icon}</div>
      <div>
        <div style="font-size:1.5em;font-weight:bold;color:{verdict_color};">{verdict_text}</div>
        <div style="color:#cccccc;font-size:0.9em;margin-top:4px;">{verdict_sub}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Lista de checks
    check_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
    for label, ok, note in checks:
        icon     = "✅" if ok else "❌"
        bg       = "#0a2a0a" if ok else "#2a0a0a"
        bd_color = "#00aa44" if ok else "#aa2222"
        check_html += f"""
        <div style="background:{bg};border:1px solid {bd_color};border-radius:8px;
                    padding:10px 14px;font-size:0.88em;">
          <span style="font-size:1.1em;">{icon}</span>
          <span style="color:#e0e0e0;font-weight:600;"> {label}</span><br/>
          <span style="color:#999999;font-size:0.85em;padding-left:1.6em;">{note}</span>
        </div>"""
    check_html += "</div>"
    st.markdown(check_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Aplicación principal
# ─────────────────────────────────────────────────────────────
def main():
    if not _STREAMLIT_AVAILABLE:
        print("❌ Streamlit no instalado. Ejecute: pip install streamlit plotly")
        return

    _setup_page()

    # ── Barra lateral ────────────────────────────────────────
    sidebar_opts = _sidebar({})

    # ── Auto-refresh (selector en sidebar) ───────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔄 Auto-refresh")
    refresh_options = {"Desactivado": 0, "1 min": 60, "5 min": 300, "10 min": 600, "15 min": 900}
    refresh_label   = st.sidebar.selectbox(
        "Intervalo de actualización",
        options=list(refresh_options.keys()),
        index=2,
        help="Recarga automática del dashboard"
    )
    refresh_secs = refresh_options[refresh_label]

    # ── Cargar datos ────────────────────────────────────────
    with st.spinner("⏳ Cargando datos y entrenando modelo..."):
        try:
            data = load_model_and_data(
                mode=sidebar_opts["mode"],
                n_rows=sidebar_opts["n_rows"],
            )
        except Exception as e:
            st.error(f"❌ Error al cargar datos: {e}")
            st.exception(e)
            return

    # ── Obtener score de noticias (silencioso, usa caché) ─────
    news_score = 0.0
    try:
        if _NEWS_AVAILABLE:
            nk = os.getenv("NEWS_API_KEY", "")
            ak = os.getenv("ALPHAVANTAGE_KEY", "")
            cached_news = _fetch_news_cached(nk, ak)
            if cached_news:
                result_obj = cached_news.get("_result_obj")
                if result_obj:
                    news_score = float(result_obj.aggregate_score)
    except Exception:
        pass

    # ── Tarjetas principales ──────────────────────────────────
    _render_signal_cards(data, news_score=news_score)

    # ── Checklist de Trading Diario ──────────────────────────
    _render_trading_checklist(data, news_score=news_score)

    st.markdown("---")

    # ── MetaTrader: Enviar señal + mostrar estado ─────────────
    if sidebar_opts.get("mt_enabled") and sidebar_opts.get("mt_cfg"):
        _send_and_render_mt_signal(data, sidebar_opts["mt_cfg"])

    # ── Pestañas ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Gráfico",
        "🔬 Características",
        "📰 Noticias",
        "📈 Rendimiento",
    ])

    with tab1:
        _render_price_chart(data)
        _render_recent_signals(data)

    with tab2:
        if sidebar_opts["show_features"]:
            _render_feature_importance(data)

    with tab3:
        _render_news_tab(
            newsapi_key=os.getenv("NEWS_API_KEY", ""),
            av_key=os.getenv("ALPHAVANTAGE_KEY", ""),
        )

    with tab4:
        _render_performance_tab("signals_history.json")

    # ── Footer: timestamp + auto-refresh JS ──────────────────
    st.markdown("---")
    col_l, col_r = st.columns([3, 1])
    with col_l:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"🕒 Última actualización: {now_str}")
    with col_r:
        if refresh_secs > 0:
            st.caption(f"⏱️ Auto-refresh: {refresh_label}")

    # Inyectar auto-refresh via JavaScript
    if refresh_secs > 0:
        import streamlit.components.v1 as components
        components.html(
            f"""<script>
            setTimeout(function() {{
                window.parent.location.reload();
            }}, {refresh_secs * 1000});
            </script>""",
            height=0,
        )


if __name__ == "__main__":
    main()
