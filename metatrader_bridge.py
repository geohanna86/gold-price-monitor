# ============================================================
# metatrader_bridge.py — Puente de integración con MetaTrader 4/5
# Gold Price Monitor — Phase 4+
#
# Cómo funciona:
#   Python escribe una señal en un archivo JSON compartido
#   Expert Advisor en MT4/MT5 lee el archivo cada segundo y ejecuta
#
# Dos formas de integración:
#   1) FILE-BASED (MT4 + MT5): escribir signals.json en la carpeta MT
#   2) MT5 Python API: control directo de MT5 (solo Windows)
#
# Configuración:
#   pip install MetaTrader5   (opcional — solo MT5)
#   Copia GoldSignalEA.mq5 a MQL5/Experts/ en MT5
# ============================================================

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("MTBridge")


# ─────────────────────────────────────────────────────────────
# Configuración del puente
# ─────────────────────────────────────────────────────────────
@dataclass
class MTBridgeConfig:
    # Ruta de la carpeta MT4/MT5 en tu máquina
    # MT5 predeterminado: C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\<ID>\MQL5\Files
    mt_files_path: str = field(default_factory=lambda: os.getenv(
        "MT_FILES_PATH",
        str(Path.home() / "AppData/Roaming/MetaQuotes/Terminal/Common/Files")
    ))
    signal_filename: str = "gold_signal.json"   # nombre del archivo de señal

    # Configuración del riesgo
    lot_size:        float = 0.01   # tamaño del lote (0.01 = micro lote)
    risk_pct:        float = 1.0    # porcentaje de riesgo del capital %
    sl_atr_mult:     float = 1.5    # Stop Loss = 1.5 × ATR
    tp_atr_mult:     float = 3.0    # Take Profit = 3.0 × ATR (R:R = 1:2)
    magic_number:    int   = 20260413  # número mágico para identificar operaciones

    # Filtro de confianza — no enviar orden si confianza es menor que esto
    min_confidence:  float = 0.55

    # MT5 Python API (opcional)
    use_mt5_api:     bool  = False
    mt5_symbol:      str   = "XAUUSD"
    mt5_deviation:   int   = 20      # deslizamiento aceptable en pips


# ─────────────────────────────────────────────────────────────
# Estructura de señal
# ─────────────────────────────────────────────────────────────
@dataclass
class TradingSignal:
    timestamp:   str    # formato ISO
    symbol:      str    # XAUUSD
    action:      str    # "BUY" | "SELL" | "CLOSE" | "NEUTRAL"
    signal_raw:  int    # +1 | -1 | 0
    confidence:  float  # 0.0 → 1.0
    regime:      int    # 0=Bajista | 1=Lateral | 2=Alcista
    price:       float  # precio de cierre actual
    atr:         float  # valor de ATR
    sl_price:    float  # precio de Stop Loss sugerido
    tp_price:    float  # precio de Take Profit sugerido
    lot_size:    float  # tamaño del lote
    magic:       int    # número mágico para el EA
    source:      str = "GoldMonitor_v2"
    status:      str = "NEW"   # NEW | SENT | EXECUTED | EXPIRED

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def neutral(cls, price: float, magic: int) -> "TradingSignal":
        return cls(
            timestamp=datetime.now().isoformat(),
            symbol="XAUUSD", action="NEUTRAL", signal_raw=0,
            confidence=0.0, regime=1, price=price, atr=0.0,
            sl_price=0.0, tp_price=0.0, lot_size=0.0, magic=magic,
        )


# ─────────────────────────────────────────────────────────────
# Puente principal
# ─────────────────────────────────────────────────────────────
class MetaTraderBridge:
    """
    Conecta los resultados del modelo con MetaTrader 4/5.

    Uso:
        bridge = MetaTraderBridge(MTBridgeConfig())
        bridge.send_signal(signal=1, confidence=0.7,
                           regime=2, price=3100.0, atr=15.0)
    """

    ACTION_MAP = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}
    REGIME_MAP = {0: "Bajista 🔴", 1: "Lateral ⚪", 2: "Alcista 🟢"}

    def __init__(self, config: MTBridgeConfig = None):
        self.cfg          = config or MTBridgeConfig()
        self._mt5_ready   = False
        self._last_signal: Optional[TradingSignal] = None
        self.signal_log: list = []

        # Intenta inicializar MT5 API
        if self.cfg.use_mt5_api:
            self._init_mt5()

        # Asegurar que existe la carpeta de archivos
        Path(self.cfg.mt_files_path).mkdir(parents=True, exist_ok=True)

    # ── Inicializar MT5 API ─────────────────────────────────────────
    def _init_mt5(self):
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            if mt5.initialize():
                self._mt5_ready = True
                info = mt5.terminal_info()
                logger.info(f"✅ MT5 conectado | Build: {info.build}")
            else:
                logger.warning(f"⚠️ MT5 falló en inicialización: {mt5.last_error()}")
        except ImportError:
            logger.info("MetaTrader5 no instalado — se usará solo File Mode")
            self._mt5_ready = False

    # ── Construir señal ──────────────────────────────────────────
    def _build_signal(
        self,
        signal:     int,
        confidence: float,
        regime:     int,
        price:      float,
        atr:        float,
    ) -> TradingSignal:
        action = self.ACTION_MAP.get(signal, "NEUTRAL")

        # Calcular SL y TP
        if signal == 1:    # BUY
            sl = round(price - self.cfg.sl_atr_mult * atr, 2)
            tp = round(price + self.cfg.tp_atr_mult * atr, 2)
        elif signal == -1: # SELL
            sl = round(price + self.cfg.sl_atr_mult * atr, 2)
            tp = round(price - self.cfg.tp_atr_mult * atr, 2)
        else:
            sl = tp = 0.0

        return TradingSignal(
            timestamp  = datetime.now().isoformat(),
            symbol     = self.cfg.mt5_symbol,
            action     = action,
            signal_raw = signal,
            confidence = round(confidence, 4),
            regime     = regime,
            price      = round(price, 2),
            atr        = round(atr, 4),
            sl_price   = sl,
            tp_price   = tp,
            lot_size   = self.cfg.lot_size,
            magic      = self.cfg.magic_number,
        )

    # ── Envío principal ───────────────────────────────────────
    def send_signal(
        self,
        signal:     int,
        confidence: float,
        regime:     int,
        price:      float,
        atr:        float = 0.0,
        force:      bool  = False,
    ) -> Optional[TradingSignal]:
        """
        Envía una señal a MT4/MT5.

        force=True: envía incluso si confianza es baja.
        Devuelve TradingSignal o None si fue ignorada.
        """
        # Filtro de confianza
        if not force and confidence < self.cfg.min_confidence and signal != 0:
            logger.info(f"⏭ Señal ignorada — confianza baja: {confidence:.0%}")
            return None

        # No repetir la misma señal
        if (self._last_signal and
                self._last_signal.signal_raw == signal and
                signal != 0):
            logger.info(f"⏭ Misma señal anterior ({signal}) — no enviar")
            return None

        sig_obj = self._build_signal(signal, confidence, regime, price, atr)

        # ── 1) Escribir archivo (MT4 + MT5) ──────────────────────
        self._write_signal_file(sig_obj)

        # ── 2) MT5 Python API directo (opcional) ───────────────
        if self._mt5_ready and signal != 0:
            self._send_mt5_order(sig_obj)

        self._last_signal = sig_obj
        self.signal_log.append(sig_obj)
        logger.info(
            f"📤 Señal enviada | {sig_obj.action} @ ${price:,.2f} | "
            f"SL: ${sig_obj.sl_price:,.2f} | TP: ${sig_obj.tp_price:,.2f} | "
            f"Confianza: {confidence:.0%} | {self.REGIME_MAP.get(regime,'—')}"
        )
        return sig_obj

    # ── Escribir archivo de señal ─────────────────────────────────────
    def _write_signal_file(self, sig: TradingSignal):
        path = Path(self.cfg.mt_files_path) / self.cfg.signal_filename
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sig.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Archivo de señal escrito: {path}")
        except Exception as e:
            logger.error(f"❌ Fallo al escribir archivo: {e}")

    # ── Orden MT5 directo ───────────────────────────────────────
    def _send_mt5_order(self, sig: TradingSignal):
        try:
            mt5 = self._mt5
            order_type = (mt5.ORDER_TYPE_BUY
                         if sig.action == "BUY"
                         else mt5.ORDER_TYPE_SELL)

            request = {
                "action":    mt5.TRADE_ACTION_DEAL,
                "symbol":    sig.symbol,
                "volume":    sig.lot_size,
                "type":      order_type,
                "price":     mt5.symbol_info_tick(sig.symbol).ask
                             if sig.action == "BUY"
                             else mt5.symbol_info_tick(sig.symbol).bid,
                "sl":        sig.sl_price,
                "tp":        sig.tp_price,
                "deviation": self.cfg.mt5_deviation,
                "magic":     sig.magic,
                "comment":   f"GoldMonitor_{sig.confidence:.0%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                sig.status = "EXECUTED"
                logger.info(f"✅ Orden MT5 ejecutada | Ticket: {result.order}")
            else:
                logger.error(f"❌ MT5 rechazó orden: {result.comment}")
        except Exception as e:
            logger.error(f"❌ Error en MT5 API: {e}")

    # ── Cerrar posiciones ─────────────────────────────────────────
    def close_all_positions(self):
        """Cierra todas las posiciones abiertas por número mágico."""
        if not self._mt5_ready:
            self._write_signal_file(TradingSignal.neutral(0.0, self.cfg.magic_number))
            return
        try:
            mt5 = self._mt5
            positions = mt5.positions_get(symbol=self.cfg.mt5_symbol)
            for pos in (positions or []):
                if pos.magic == self.cfg.magic_number:
                    close_type = (mt5.ORDER_TYPE_SELL
                                  if pos.type == mt5.ORDER_TYPE_BUY
                                  else mt5.ORDER_TYPE_BUY)
                    request = {
                        "action":   mt5.TRADE_ACTION_DEAL,
                        "symbol":   pos.symbol,
                        "volume":   pos.volume,
                        "type":     close_type,
                        "position": pos.ticket,
                        "price":    mt5.symbol_info_tick(pos.symbol).bid,
                        "magic":    self.cfg.magic_number,
                        "comment":  "GoldMonitor_Close",
                    }
                    mt5.order_send(request)
        except Exception as e:
            logger.error(f"Error al cerrar: {e}")

    def get_signal_log_df(self) -> pd.DataFrame:
        if not self.signal_log:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.signal_log])

    def get_signal_file_path(self) -> str:
        return str(Path(self.cfg.mt_files_path) / self.cfg.signal_filename)

    def disconnect(self):
        if self._mt5_ready:
            self._mt5.shutdown()
            self._mt5_ready = False


# ─────────────────────────────────────────────────────────────
# Integración con Dashboard — función auxiliar
# ─────────────────────────────────────────────────────────────
def get_atr_from_df(df_ind: pd.DataFrame) -> float:
    """Extrae el último valor de ATR del DataFrame de indicadores."""
    for col in ["ATR", "ATR_14"]:
        if col in df_ind.columns:
            val = df_ind[col].dropna().iloc[-1]
            return float(val) if pd.notna(val) else 0.0
    return 0.0


# ─────────────────────────────────────────────────────────────
# Pruebas unitarias / Assertions
# ─────────────────────────────────────────────────────────────
def _run_assertions():
    print("  ← Ejecutando Assertions para MetaTrader Bridge ...")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = MTBridgeConfig(
            mt_files_path=tmpdir,
            min_confidence=0.50,
            lot_size=0.01,
        )
        bridge = MetaTraderBridge(cfg)

        # 1) Señal de compra
        sig = bridge.send_signal(1, 0.70, 2, 3100.0, atr=15.0)
        assert sig is not None, "Debe enviarse la señal"
        assert sig.action == "BUY"
        assert sig.sl_price < 3100.0, "SL debe ser menor que el precio al comprar"
        assert sig.tp_price > 3100.0, "TP debe ser mayor que el precio al comprar"

        # 2) Archivo escrito
        signal_path = bridge.get_signal_file_path()
        assert os.path.exists(signal_path), "Archivo de señal debe ser escrito"
        with open(signal_path) as f:
            data = json.load(f)
        assert data["action"] == "BUY"
        assert data["confidence"] == 0.7

        # 3) Misma señal no se repite
        sig2 = bridge.send_signal(1, 0.75, 2, 3102.0, atr=15.0)
        assert sig2 is None, "No debe repetirse la misma señal"

        # 4) Señal de baja confianza ignorada
        sig3 = bridge.send_signal(-1, 0.30, 0, 3090.0, atr=15.0)
        assert sig3 is None, "Confianza < 50% debe ser ignorada"

        # 5) Registro de señales
        df_log = bridge.get_signal_log_df()
        assert len(df_log) == 1, "Debe haber una señal en el registro"

    print("  ✅ Todos los Assertions de MetaTrader Bridge pasaron exitosamente!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_assertions()
