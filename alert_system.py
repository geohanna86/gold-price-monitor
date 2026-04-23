# ============================================================
# alert_system.py — Sistema de alertas (Telegram + Console)
# Gold Price Monitor — Phase 4
#
# Comportamiento automático:
#   - Si TELEGRAM_TOKEN está definido → envía a Telegram
#   - Si no → imprime en Console (Mock Mode)
#
# Configuración:
#   export TELEGRAM_TOKEN="your_bot_token"
#   export TELEGRAM_CHAT_ID="your_chat_id"
#
# Obtener Chat ID:
#   1. Busca @userinfobot en Telegram
#   2. Envía /start y responderá con Chat ID
# ============================================================

import os
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger("AlertSystem")


# ─────────────────────────────────────────────────────────────
# Configuración del sistema de alertas
# ─────────────────────────────────────────────────────────────
@dataclass
class AlertConfig:
    # Datos de Telegram (opcional — se lee de variables de entorno)
    telegram_token:   str  = field(default_factory=lambda: os.getenv("TELEGRAM_TOKEN", ""))
    telegram_chat_id: str  = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # Umbrales de alerta
    min_confidence:   float = 0.50   # No enviar alerta si confianza < 50%
    regime_change_alert: bool = True  # Alerta al cambiar régimen
    signal_change_alert: bool = True  # Alerta al cambiar señal
    daily_summary:    bool = True     # Resumen diario

    # Período de enfriamiento (segundos) — para evitar spam
    cooldown_seconds: int = 3600      # Una hora entre alertas similares


# ─────────────────────────────────────────────────────────────
# Mensaje de alerta
# ─────────────────────────────────────────────────────────────
@dataclass
class AlertMessage:
    title:     str
    body:      str
    alert_type: str   # "signal", "regime", "summary", "error"
    timestamp: datetime = field(default_factory=datetime.now)
    signal:    Optional[int]   = None   # {-1, 0, 1}
    confidence: Optional[float] = None
    regime:    Optional[int]   = None   # {0, 1, 2}
    price:     Optional[float] = None

    def to_telegram_html(self) -> str:
        """Formato de mensaje HTML para Telegram."""
        emoji_map = {
            "signal": {"1": "🟢", "-1": "🔴", "0": "⚪"},
            "regime": {0: "🔴 Bajista", 1: "⚪ Lateral", 2: "🟢 Alcista"},
        }
        lines = [
            f"<b>{self.title}</b>",
            f"<i>{self.timestamp.strftime('%Y-%m-%d %H:%M')}</i>",
            "─────────────────",
            self.body,
        ]
        if self.price:
            lines.append(f"💰 Precio: <b>${self.price:,.2f}</b>")
        if self.signal is not None:
            sig_emoji = emoji_map["signal"].get(str(self.signal), "⚪")
            sig_text  = {1: "Compra", -1: "Venta", 0: "Neutro"}.get(self.signal, "—")
            lines.append(f"📡 Señal: <b>{sig_emoji} {sig_text}</b>")
        if self.confidence is not None:
            lines.append(f"🎯 Confianza: <b>{self.confidence:.0%}</b>")
        if self.regime is not None:
            reg_text = emoji_map["regime"].get(self.regime, "—")
            lines.append(f"🌐 Régimen: <b>{reg_text}</b>")
        lines.append("\n<i>Gold Price Monitor 🥇</i>")
        return "\n".join(lines)

    def to_console_str(self) -> str:
        """Texto simple para terminal."""
        return (
            f"\n{'='*50}\n"
            f"[{self.alert_type.upper()}] {self.title}\n"
            f"Hora: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{self.body}\n"
            + (f"Precio: ${self.price:,.2f}\n" if self.price else "")
            + (f"Señal: {self.signal} | Confianza: {self.confidence:.0%}\n"
               if self.signal is not None and self.confidence else "")
            + (f"Régimen: {self.regime}\n" if self.regime is not None else "")
            + f"{'='*50}"
        )


# ─────────────────────────────────────────────────────────────
# Enviador de Telegram
# ─────────────────────────────────────────────────────────────
class TelegramSender:
    """Envía mensajes a Telegram usando requests."""

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self._available = bool(token and chat_id)

        if self._available:
            try:
                import requests
                self._requests = requests
            except ImportError:
                logger.warning("La librería requests no está instalada — se usará solo Console")
                self._available = False
                self._requests  = None
        else:
            self._requests = None

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self._available:
            return False
        try:
            url  = self.BASE_URL.format(token=self.token)
            data = {
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": parse_mode,
            }
            resp = self._requests.post(url, data=data, timeout=10)
            if resp.status_code == 200:
                logger.info("✅ Mensaje de Telegram enviado exitosamente")
                return True
            else:
                logger.error(f"Error en Telegram API: {resp.status_code} — {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Fallo al enviar Telegram: {e}")
            return False

    @property
    def is_available(self) -> bool:
        return self._available


# ─────────────────────────────────────────────────────────────
# Sistema completo de alertas
# ─────────────────────────────────────────────────────────────
class GoldAlertSystem:
    """
    Monitorea señales y regímenes, envía alertas al cambiar.

    Uso:
        alert = GoldAlertSystem(AlertConfig())
        alert.process_new_data(current_signal, current_regime,
                               current_price, confidence)
    """

    REGIME_NAMES = {0: "Bajista 🔴", 1: "Lateral ⚪", 2: "Alcista 🟢"}
    SIGNAL_NAMES = {1: "Compra 📈", -1: "Venta 📉", 0: "Neutro ⏸"}

    def __init__(self, config: AlertConfig = None):
        self.cfg      = config or AlertConfig()
        self._sender  = TelegramSender(
            self.cfg.telegram_token, self.cfg.telegram_chat_id
        )
        self._prev_signal:  Optional[int] = None
        self._prev_regime:  Optional[int] = None
        self._last_sent:    dict = {}   # alert_type → timestamp del último envío
        self.alert_history: List[AlertMessage] = []

    # ── Procesamiento principal ─────────────────────────────────────
    def process_new_data(
        self,
        signal:     int,
        regime:     int,
        price:      float,
        confidence: float,
        date:       Optional[datetime] = None,
    ):
        """
        Procesa datos nuevos y envía alertas necesarias.
        Se llama en cada ciclo de actualización.
        """
        now = date or datetime.now()

        alerts_to_send: List[AlertMessage] = []

        # ── Alerta de cambio de señal ─────────────────────────────
        if (self.cfg.signal_change_alert
                and self._prev_signal != signal
                and confidence >= self.cfg.min_confidence):
            alert = self._build_signal_alert(signal, confidence, price, regime, now)
            alerts_to_send.append(alert)

        # ── Alerta de cambio de régimen ──────────────────────────────
        if (self.cfg.regime_change_alert
                and self._prev_regime is not None
                and self._prev_regime != regime):
            alert = self._build_regime_alert(regime, self._prev_regime, price, now)
            alerts_to_send.append(alert)

        # ── Enviar alertas ──────────────────────────────────
        for alert in alerts_to_send:
            if self._should_send(alert.alert_type):
                self._dispatch(alert)
                self._last_sent[alert.alert_type] = now

        # ── Actualizar estado previo ────────────────────────────
        self._prev_signal = signal
        self._prev_regime = regime

    # ── Resumen diario ─────────────────────────────────────────────
    def send_daily_summary(
        self,
        price:       float,
        daily_return: float,
        signal:      int,
        regime:      int,
        confidence:  float,
        total_return: float,
        sharpe:      float,
    ):
        """Envía un resumen diario completo."""
        sign = "+" if daily_return >= 0 else ""
        body = (
            f"Cambio del día: {sign}{daily_return:.2%}\n"
            f"Retorno total (Backtest): {total_return:+.2%}\n"
            f"Sharpe Ratio: {sharpe:.2f}\n"
        )
        alert = AlertMessage(
            title="📊 Resumen diario — Gold Monitor",
            body=body,
            alert_type="summary",
            signal=signal,
            confidence=confidence,
            regime=regime,
            price=price,
        )
        self._dispatch(alert)

    # ── Construir alerta de señal ────────────────────────────────
    def _build_signal_alert(
        self, signal, confidence, price, regime, now
    ) -> AlertMessage:
        prev_text = self.SIGNAL_NAMES.get(self._prev_signal, "—") if self._prev_signal is not None else "—"
        new_text  = self.SIGNAL_NAMES.get(signal, "—")
        body = f"Señal cambió de {prev_text} → {new_text}"
        return AlertMessage(
            title="⚡ Nueva alerta de señal!",
            body=body,
            alert_type="signal",
            signal=signal,
            confidence=confidence,
            regime=regime,
            price=price,
            timestamp=now,
        )

    # ── Construir alerta de régimen ─────────────────────────────────
    def _build_regime_alert(
        self, new_regime, prev_regime, price, now
    ) -> AlertMessage:
        prev_text = self.REGIME_NAMES.get(prev_regime, "—")
        new_text  = self.REGIME_NAMES.get(new_regime, "—")
        body = f"Cambio de régimen de mercado:\n{prev_text}  →  {new_text}"
        if new_regime == 0:
            body += "\n⚠️ Advertencia: Entramos en régimen bajista — señales de compra suspendidas."
        elif new_regime == 2:
            body += "\n✅ Entramos en régimen alcista — tendencia positiva."
        return AlertMessage(
            title="🌐 Cambio de régimen de mercado!",
            body=body,
            alert_type="regime",
            regime=new_regime,
            price=price,
            timestamp=now,
        )

    # ── Filtro de enfriamiento ────────────────────────────────────────
    def _should_send(self, alert_type: str) -> bool:
        if alert_type not in self._last_sent:
            return True
        elapsed = (datetime.now() - self._last_sent[alert_type]).total_seconds()
        return elapsed >= self.cfg.cooldown_seconds

    # ── Envío real ────────────────────────────────────────────
    def _dispatch(self, alert: AlertMessage):
        self.alert_history.append(alert)
        if self._sender.is_available:
            self._sender.send(alert.to_telegram_html())
        else:
            # Console fallback
            print(alert.to_console_str())
        logger.info(f"Alert dispatched: [{alert.alert_type}] {alert.title}")

    def get_alert_history_df(self) -> pd.DataFrame:
        """Devuelve historial de alertas como DataFrame."""
        if not self.alert_history:
            return pd.DataFrame()
        rows = [
            {
                "timestamp":  a.timestamp,
                "type":       a.alert_type,
                "title":      a.title,
                "signal":     a.signal,
                "regime":     a.regime,
                "price":      a.price,
                "confidence": a.confidence,
            }
            for a in self.alert_history
        ]
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Interfaz de monitoreo periódico (loop)
# ─────────────────────────────────────────────────────────────
class GoldMonitorLoop:
    """
    Ejecuta un bucle de monitoreo periódico:
      Cada N segundos → obtiene datos recientes → ejecuta el modelo → envía alertas.
    """

    def __init__(
        self,
        alert_system: GoldAlertSystem,
        interval_seconds: int = 3600,  # cada hora
        mode: str = "mock",
    ):
        self.alert   = alert_system
        self.interval = interval_seconds
        self.mode     = mode
        self._ensemble = None
        self._running  = False

    def setup(self, n_rows: int = 500):
        """Entrena el modelo una sola vez antes de iniciar el bucle."""
        from config import AppConfig
        from data_fetcher import GoldDataFetcher
        from indicators import TechnicalIndicators
        from feature_engineer import FeatureEngineer
        from lstm_model import LSTMConfig
        from ensemble import GoldEnsemble, EnsembleConfig

        cfg = AppConfig(mode=self.mode)
        cfg.mock.n_rows = n_rows

        df_raw  = GoldDataFetcher(cfg).get_data()
        df_ind  = TechnicalIndicators(df_raw, cfg).add_all().get_dataframe()
        fe      = FeatureEngineer(df_ind)
        fe.build_features()
        df_feat = fe.get_full_data()

        ens_cfg = EnsembleConfig(
            lstm_config=LSTMConfig(hidden_size=32, epochs=30, patience=8)
        )
        self._ensemble = GoldEnsemble(ens_cfg)
        self._ensemble.fit(df_feat, df_ind)
        print("✅ Modelo listo — iniciando monitoreo...")

    def run_once(self) -> dict:
        """Ejecuta un ciclo y devuelve la última señal."""
        from config import AppConfig
        from data_fetcher import GoldDataFetcher
        from indicators import TechnicalIndicators
        from feature_engineer import FeatureEngineer

        cfg    = AppConfig(mode=self.mode)
        df_raw = GoldDataFetcher(cfg).get_data()
        df_ind = TechnicalIndicators(df_raw, cfg).add_all().get_dataframe()
        fe     = FeatureEngineer(df_ind)
        fe.build_features()
        df_feat = fe.get_full_data()

        results = self._ensemble.predict(df_feat, df_ind)

        last_signal  = int(results.signals.iloc[-1])
        last_conf    = float(results.confidence.iloc[-1])
        last_regime  = int(results.regimes.iloc[-1])
        last_price   = float(df_ind["Close"].iloc[-1])

        self.alert.process_new_data(
            signal=last_signal, regime=last_regime,
            price=last_price, confidence=last_conf,
        )
        return {
            "signal": last_signal, "confidence": last_conf,
            "regime": last_regime, "price": last_price,
        }

    def start(self, max_iterations: int = None):
        """Inicia el bucle — max_iterations=None significa continuo."""
        if self._ensemble is None:
            self.setup()
        self._running = True
        iteration     = 0
        print(f"🔄 Iniciando monitoreo — cada {self.interval} segundos")
        while self._running:
            try:
                result = self.run_once()
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"Precio: ${result['price']:,.2f} | "
                    f"Señal: {result['signal']:+d} | "
                    f"Confianza: {result['confidence']:.0%} | "
                    f"Régimen: {result['regime']}"
                )
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    break
                time.sleep(self.interval)
            except KeyboardInterrupt:
                print("\n⛔ Monitoreo detenido manualmente.")
                break
            except Exception as e:
                logger.error(f"Error en el ciclo: {e}")
                time.sleep(60)

    def stop(self):
        self._running = False


# ─────────────────────────────────────────────────────────────
# Mock Data para prueba independiente
# ─────────────────────────────────────────────────────────────
def _run_assertions():
    """Pruebas rápidas para el sistema de alertas."""
    print("  ← Ejecutando Assertions para AlertSystem ...")

    cfg = AlertConfig(
        telegram_token="",   # vacío → Console mode
        min_confidence=0.40,
        cooldown_seconds=0,  # sin enfriamiento en la prueba
    )
    alert = GoldAlertSystem(cfg)

    # 1) Alerta al cambiar señal
    alert.process_new_data(signal=1, regime=2, price=2100.0, confidence=0.65)
    assert len(alert.alert_history) == 1, "Debe enviarse alerta en primera señal"

    # 2) Sin alerta si la señal no cambia
    alert.process_new_data(signal=1, regime=2, price=2105.0, confidence=0.70)
    assert len(alert.alert_history) == 1, "Sin alerta si la señal no cambió"

    # 3) Alerta al cambiar señal
    alert.process_new_data(signal=0, regime=2, price=2095.0, confidence=0.55)
    assert len(alert.alert_history) == 2, "Debe enviarse alerta al cambiar señal"

    # 4) Alerta al cambiar régimen
    alert.process_new_data(signal=0, regime=0, price=2080.0, confidence=0.60)
    assert len(alert.alert_history) == 3, "Debe enviarse alerta al cambiar régimen"

    # 5) Formato de mensaje HTML
    msg = alert.alert_history[0]
    html = msg.to_telegram_html()
    assert "<b>" in html, "Mensaje HTML debe tener formato"

    # 6) Formato de Console
    console = msg.to_console_str()
    assert "=" in console, "Console debe tener separador"

    # 7) DataFrame del historial
    df = alert.get_alert_history_df()
    assert len(df) == 3, f"Debe haber 3 alertas, encontrado: {len(df)}"

    print("  ✅ Todos los Assertions de AlertSystem pasaron exitosamente!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_assertions()
