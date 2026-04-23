# ============================================================
# ensemble.py — Modelo de ensamble (Ensemble)
# Gold Price Monitor — Phase 3
#
# Ecuación:
#   final_signal = weighted_vote(RF=0.4, LSTM=0.4, HMM_filter=0.2)
#   → Si el régimen es Bear: señales de compra se cancelan (HMM como filtro)
#
# Salidas:
#   - Señal de ensamble: {-1, 0, +1}
#   - Confianza: [0.0, 1.0]
#   - Régimen dominante: {0=Bear, 1=Sideways, 2=Bull}
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from ml_model import GoldPredictor, ModelConfig
from lstm_model import GoldLSTM, LSTMConfig, prepare_lstm_data
from hmm_model import GoldRegimeDetector, HMMConfig

# NewsFilter — opcional (no bloquea si no está disponible)
try:
    from news_filter import GoldNewsSentimentFilter, NewsConfig, SentimentResult
    _NEWS_AVAILABLE = True
except ImportError:
    _NEWS_AVAILABLE = False

# TradingFilters — opcional (graceful degradation)
try:
    from trading_filters import TradingFiltersManager, TradingContext
    _FILTERS_AVAILABLE = True
except ImportError:
    _FILTERS_AVAILABLE = False

# PerformanceTracker — opcional (graceful degradation)
try:
    from performance_tracker import PerformanceTracker
    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False

logger = logging.getLogger("Ensemble")


# ─────────────────────────────────────────────────────────────
# Configuración de ensamble
# ─────────────────────────────────────────────────────────────
@dataclass
class EnsembleConfig:
    rf_weight:   float = 0.40   # Peso de Random Forest
    lstm_weight: float = 0.40   # Peso de LSTM
    hmm_weight:  float = 0.20   # Peso de HMM (también se usa como filtro)

    # Configuración de componentes
    rf_config:   ModelConfig = field(default_factory=lambda: ModelConfig(
        n_estimators=200, max_depth=6, min_samples_leaf=10, n_cv_splits=5
    ))
    lstm_config: LSTMConfig  = field(default_factory=LSTMConfig)
    hmm_config:  HMMConfig   = field(default_factory=HMMConfig)

    # Filtro HMM
    cancel_buy_in_bear:  bool  = True   # Cancela compra en Bear
    cancel_sell_in_bull: bool  = False  # Mantiene venta en Bull
    min_ensemble_confidence: float = 0.35  # Confianza mínima

    # News Sentiment Filter (Fase 5)
    use_news_filter:     bool  = False  # Activar con True cuando tengas API key
    news_config:         object = None  # NewsConfig() — se crea automáticamente

    # ─────────────────────────────────────────────────────────
    # Filtros de Trading Avanzados (Fase 4+)
    # ─────────────────────────────────────────────────────────
    use_trading_filters: bool = True      # Habilitar filtros de sesión/MTF/DXY
    use_performance_tracker: bool = True  # Habilitar rastreador de desempeño
    tracker_filepath: str = "signals_history.json"  # Ruta para guardar historial de señales

    # Configuración de filtros individuales
    session_filter_enabled: bool = True   # Filtro de sesión de trading
    mtf_enabled: bool = True              # Análisis multi-timeframe
    dxy_enabled: bool = True              # Filtro de sesgo del dólar

    # Multiplicadores para ATR (cálculo dinámico de SL/TP)
    atr_sl_multiplier: float = 1.5        # Multiplicador ATR para Stop Loss
    atr_tp_multiplier: float = 2.5        # Multiplicador ATR para Take Profit

    def __post_init__(self):
        total = self.rf_weight + self.lstm_weight + self.hmm_weight
        assert abs(total - 1.0) < 1e-6, \
            f"Suma de pesos debe = 1.0, actual: {total}"


# ─────────────────────────────────────────────────────────────
# Resultados de ensamble
# ─────────────────────────────────────────────────────────────
@dataclass
class EnsembleResults:
    signals:        pd.Series    # Señal final {-1, 0, +1}
    confidence:     pd.Series    # Confianza [0, 1]
    regimes:        pd.Series    # Régimen de mercado {0, 1, 2}
    rf_signals:     pd.Series    # Señales RF crudas
    lstm_signals:   pd.Series    # Señales LSTM crudas
    rf_proba:       np.ndarray   # Probabilidad RF (N, 3)
    lstm_proba:     np.ndarray   # Probabilidad LSTM (N, 3)


# ─────────────────────────────────────────────────────────────
# Resultado de predicción individual (para predict_single)
# ─────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    """Resultado de una predicción individual con detalles de entrada/salida."""
    signal: int                                  # {-1, 0, +1}
    confidence: float                            # [0.0, 1.0]
    regime: int                                  # {0=Bear, 1=Sideways, 2=Bull}
    action: str                                  # "BUY", "SELL", "NEUTRAL"
    entry_price: float                           # Precio de entrada sugerido
    sl_price: float                              # Stop Loss (dinámico por ATR)
    tp_price: float                              # Take Profit (dinámico por ATR)
    atr_value: float = 0.0                       # Valor de ATR utilizado
    trading_context: Optional[TradingContext] = None  # Contexto de filtros de trading
    notes: str = ""                              # Notas adicionales
    signal_id: str = ""                          # ID único para rastreo (generado por tracker)


# ─────────────────────────────────────────────────────────────
# Modelo completo
# ─────────────────────────────────────────────────────────────
class GoldEnsemble:
    """
    Modelo de ensamble que integra RF + LSTM + HMM.

    Uso:
        ensemble = GoldEnsemble(EnsembleConfig())
        ensemble.fit(df_features, df_raw)
        results  = ensemble.predict(df_features, df_raw)
    """

    def __init__(self, config: EnsembleConfig = None):
        self.cfg         = config or EnsembleConfig()
        self._rf         = GoldPredictor(self.cfg.rf_config)
        self._lstm       = None   # Se crea después de conocer tamaño de características
        self._hmm        = GoldRegimeDetector(self.cfg.hmm_config)
        self._is_trained = False
        self._n_features = 0
        self._last_sentiment: Optional[object] = None   # último SentimentResult

        # Inicializar NewsFilter si está activado
        self._news_filter = None
        if self.cfg.use_news_filter and _NEWS_AVAILABLE:
            news_cfg = self.cfg.news_config or NewsConfig()
            self._news_filter = GoldNewsSentimentFilter(news_cfg)
            logger.info("NewsFilter activado ✅")

        # ─────────────────────────────────────────────────────
        # Inicializar Filtros de Trading (Fase 4+)
        # ─────────────────────────────────────────────────────
        # Gestor de filtros: sesión, MTF, DXY
        self._filters = None
        if self.cfg.use_trading_filters and _FILTERS_AVAILABLE:
            self._filters = TradingFiltersManager()
            logger.info("TradingFiltersManager activado ✅")

        # Rastreador de desempeño de señales
        self._tracker = None
        if self.cfg.use_performance_tracker and _TRACKER_AVAILABLE:
            self._tracker = PerformanceTracker(self.cfg.tracker_filepath)
            logger.info(f"PerformanceTracker activado ✅ (archivo: {self.cfg.tracker_filepath})")

        # Almacenar último contexto de trading para consultas
        self._last_trading_context: Optional[TradingContext] = None

        # Guardar último valor de ATR
        self._last_atr: float = 0.0

    # ── Entrenamiento ──────────────────────────────────────────
    def fit(
        self,
        df_features: pd.DataFrame,   # Salida de FeatureEngineer (con Target)
        df_raw:      pd.DataFrame,   # Salida de TechnicalIndicators (Close, ATR_Pct)
        rf_test_size: float = 0.20,
    ) -> "GoldEnsemble":
        """
        Entrena los tres componentes en datos de entrenamiento.

        df_features: debe contener FEATURE_COLUMNS + TARGET_COLUMN
        df_raw:      debe contener Close + ATR_Pct
        """
        from feature_engineer import FeatureEngineer, FEATURE_COLUMNS, TARGET_COLUMN

        # ── 1) RF ─────────────────────────────────────────────
        logger.info("── Entrenamiento Random Forest ──")
        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe._df = df_features.copy()

        feature_cols = [c for c in FEATURE_COLUMNS if c in df_features.columns]
        self._n_features = len(feature_cols)

        X_full = df_features[feature_cols]
        y_full = df_features[TARGET_COLUMN]

        split      = int(len(X_full) * (1 - rf_test_size))
        X_train_rf = X_full.iloc[:split]
        y_train_rf = y_full.iloc[:split]

        self._rf.train(X_train_rf, y_train_rf, run_cv=False)

        # ── 2) LSTM ───────────────────────────────────────────
        logger.info("── Entrenamiento LSTM ──")
        cfg_lstm = self.cfg.lstm_config
        X_tr_seq, _, y_tr_seq, _, _ = prepare_lstm_data(
            df_features, cfg_lstm.seq_length, rf_test_size
        )

        self._lstm = GoldLSTM(self._n_features, cfg_lstm)
        self._lstm.fit(X_tr_seq, y_tr_seq)

        # ── 3) HMM ────────────────────────────────────────────
        logger.info("── Entrenamiento HMM ──")
        # Usamos solo datos de entrenamiento para HMM
        train_df_raw = df_raw.iloc[:split]
        self._hmm.fit(train_df_raw)

        self._is_trained = True
        logger.info("✅ Entrenamiento Ensemble completado")
        return self

    # ── Predicción ────────────────────────────────────────────
    def predict(
        self,
        df_features: pd.DataFrame,
        df_raw:      pd.DataFrame,
        test_size:   float = 0.20,
    ) -> EnsembleResults:
        """
        Retorna EnsembleResults en datos de prueba.
        """
        self._check_trained()
        from feature_engineer import FEATURE_COLUMNS, TARGET_COLUMN

        cfg_lstm     = self.cfg.lstm_config
        feature_cols = [c for c in FEATURE_COLUMNS if c in df_features.columns]

        # ── Sección de prueba ──────────────────────────────────
        split = int(len(df_features) * (1 - test_size))

        X_test_rf = df_features[feature_cols].iloc[split:]
        test_idx  = X_test_rf.index

        # ── Señales RF ─────────────────────────────────────────
        rf_sigs, rf_proba_arr = self._rf.predict_with_confidence(
            X_test_rf, min_confidence=0.0   # Tomamos todas las señales crudas
        )
        rf_sigs_series = pd.Series(rf_sigs, index=test_idx)

        # Probabilidad RF: reordenar columnas {-1→0, 0→1, +1→2}
        raw_proba_rf = self._rf.predict_proba(X_test_rf)
        classes_rf   = self._rf.get_classes()    # [-1, 0, 1]
        rf_proba     = _reorder_proba(raw_proba_rf, classes_rf)

        # ── Señales LSTM ───────────────────────────────────────
        _, X_test_seq, _, y_test_lstm, test_seq_idx = prepare_lstm_data(
            df_features, cfg_lstm.seq_length, test_size
        )

        lstm_sigs  = self._lstm.predict(X_test_seq)
        lstm_proba = self._lstm.predict_proba(X_test_seq)  # (N, 3): [-1, 0, +1]
        lstm_sigs_series = pd.Series(lstm_sigs, index=test_seq_idx)

        # ── Regímenes HMM ──────────────────────────────────────
        test_df_raw = df_raw.loc[test_idx] if test_idx[0] in df_raw.index else df_raw.iloc[split:]
        regimes = self._hmm.predict_regimes(test_df_raw)

        # ── Alineación entre RF y LSTM (LSTM más corto por seq_length) ─
        common_idx = test_idx.intersection(test_seq_idx)

        if len(common_idx) == 0:
            logger.warning("Sin intersección entre señales RF y LSTM — usamos solo RF")
            final_signals  = rf_sigs_series
            final_conf     = pd.Series(0.5, index=test_idx)
            final_regimes  = regimes
        else:
            final_signals, final_conf, final_regimes = self._combine(
                rf_sigs_series.loc[common_idx],
                rf_proba[test_idx.get_indexer(common_idx)],
                lstm_sigs_series.loc[common_idx],
                lstm_proba[:len(common_idx)],
                regimes.loc[common_idx],
            )

        results = EnsembleResults(
            signals      = final_signals,
            confidence   = final_conf,
            regimes      = final_regimes,
            rf_signals   = rf_sigs_series,
            lstm_signals = lstm_sigs_series,
            rf_proba     = rf_proba,
            lstm_proba   = lstm_proba,
        )

        # ── Fase 5: Ajuste por News Sentiment ────────────────────
        if self._news_filter is not None:
            results = self._apply_news_filter(results)

        return results

    # ──────────────────────────────────────────────────────────
    # Predicción individual con ATR, filtros y rastreo (Fase 4+)
    # ──────────────────────────────────────────────────────────
    def predict_single(
        self,
        df_features: pd.DataFrame,
        df_raw: pd.DataFrame,
        mtf_data: Optional[Dict[str, pd.DataFrame]] = None,
        dxy_data: Optional[pd.DataFrame] = None,
    ) -> PredictionResult:
        """
        Realiza una predicción única (última fila) con SL/TP dinámicos y filtros.

        Parámetros:
            df_features: DataFrame de características (última fila será usada)
            df_raw: DataFrame de precios OHLC
            mtf_data: Datos multi-timeframe (H1, H4, D1) para filtros
            dxy_data: Datos del DXY para filtro de sesgo del dólar

        Retorna:
            PredictionResult con entrada, SL, TP, acción recomendada

        Flujo:
            1. Obtiene señal de ensamble (RF+LSTM+HMM)
            2. Calcula SL/TP dinámicos con ATR
            3. Aplica filtros de sesión/MTF/DXY
            4. Registra la señal en tracker (si disponible)
            5. Retorna resultado con detalles de trading
        """
        self._check_trained()
        from feature_engineer import FEATURE_COLUMNS

        feature_cols = [c for c in FEATURE_COLUMNS if c in df_features.columns]

        # ── Última predicción ──────────────────────────────────
        X_latest = df_features[feature_cols].iloc[-1:].copy()

        # Señal RF
        rf_sig, _ = self._rf.predict_with_confidence(X_latest, min_confidence=0.0)
        rf_sig = int(rf_sig[0])

        # Señal LSTM
        if self._lstm is not None:
            try:
                cfg_lstm = self.cfg.lstm_config
                _, X_seq, _, _, _ = prepare_lstm_data(
                    df_features, cfg_lstm.seq_length, test_size=0.0
                )
                if len(X_seq) > 0:
                    lstm_sig = int(self._lstm.predict(X_seq[-1:].reshape(1, -1, self._n_features))[0])
                    # Combinar RF + LSTM
                    w_rf = self.cfg.rf_weight / (self.cfg.rf_weight + self.cfg.lstm_weight)
                    w_lstm = self.cfg.lstm_weight / (self.cfg.rf_weight + self.cfg.lstm_weight)
                    combined_sig = int(np.sign(w_rf * rf_sig + w_lstm * lstm_sig))
                else:
                    combined_sig = rf_sig
            except Exception as e:
                logger.debug(f"LSTM prediction failed: {e} — usando solo RF")
                combined_sig = rf_sig
        else:
            combined_sig = rf_sig

        # Régimen HMM
        regime_df = df_raw.iloc[-1:].copy()
        regime = int(self._hmm.predict_regimes(regime_df).iloc[-1])

        # Aplicar filtro HMM a la señal
        if self.cfg.cancel_buy_in_bear and regime == 0 and combined_sig == 1:
            combined_sig = 0
            logger.info("Compra cancelada por régimen Bear")

        # ── Obtener precio actual ──────────────────────────────
        current_price = float(df_raw['Close'].iloc[-1])

        # ── Calcular SL/TP con ATR ────────────────────────────
        sl_price, tp_price, atr_val = self._calculate_atr_sltp(
            df_raw, combined_sig, current_price
        )

        # ── Obtener confianza estimada ─────────────────────────
        rf_proba = self._rf.predict_proba(X_latest)[0]
        classes_rf = self._rf.get_classes()
        reordered_proba = _reorder_proba(rf_proba.reshape(1, -1), classes_rf)[0]
        idx_sig = {-1: 0, 0: 1, 1: 2}[combined_sig]
        confidence = float(reordered_proba[idx_sig])

        # Asegurar confianza mínima
        if confidence < self.cfg.min_ensemble_confidence:
            combined_sig = 0
            confidence = 0.0

        # Mapear señal a acción
        action_map = {-1: "SELL", 0: "NEUTRAL", 1: "BUY"}
        action = action_map[combined_sig]

        # ── Crear resultado preliminar ─────────────────────────
        result = PredictionResult(
            signal=combined_sig,
            confidence=confidence,
            regime=regime,
            action=action,
            entry_price=current_price,
            sl_price=sl_price,
            tp_price=tp_price,
            atr_value=atr_val,
            notes=""
        )

        # ── Aplicar filtros de trading (sesión/MTF/DXY) ────────
        if _FILTERS_AVAILABLE and self.cfg.use_trading_filters:
            result = self._apply_trading_filters(result, mtf_data, dxy_data)

        # ── Rastrear la señal ──────────────────────────────────
        if self._tracker is not None and result.action != "NEUTRAL":
            try:
                from datetime import datetime
                sig_id = self._tracker.add_signal(
                    action=result.action,
                    entry_price=result.entry_price,
                    sl_price=result.sl_price,
                    tp_price=result.tp_price,
                    lot_size=0.01,  # Tamaño por defecto, usuario puede ajustar
                    confidence=result.confidence,
                    regime=result.regime,
                    timestamp=datetime.now(),
                )
                result.signal_id = sig_id
                logger.info(f"Señal registrada | ID: {sig_id} | {result.action}")
            except Exception as e:
                logger.warning(f"Error registrando señal en tracker: {e}")

        logger.info(
            f"Predicción única | "
            f"Señal: {action} | "
            f"Conf: {confidence:.1%} | "
            f"Régimen: {['Bear', 'Sideways', 'Bull'][regime]} | "
            f"SL: {sl_price:.2f} | TP: {tp_price:.2f}"
        )

        return result

    # ── Filtro de Noticias ────────────────────────────────────────
    def _apply_news_filter(self, results: EnsembleResults) -> EnsembleResults:
        """
        Aplica el ajuste de sentiment de noticias sobre la confianza
        y opcionalmente anula señales si hay evento extremo.
        """
        try:
            sentiment = self._news_filter.analyze()
            self._last_sentiment = sentiment

            # Congelar trading si hay evento extremo (ej: Trump + score > 0.85)
            if sentiment.should_freeze_trading():
                logger.warning(
                    f"🚨 Trading congelado por evento extremo | "
                    f"score={sentiment.aggregate_score:+.2f} | "
                    f"Trump news={sentiment.trump_news_count}"
                )
                results.signals[:] = 0   # neutralizar todas las señales
                results.confidence[:] = 0.0
                return results

            # Ajustar confianza de la última señal según sentiment
            last_sig  = int(results.signals.iloc[-1])
            last_conf = float(results.confidence.iloc[-1])

            adj_conf = sentiment.adjust_confidence(
                confidence=last_conf,
                signal=last_sig,
                max_adjustment=0.30,
            )

            results.confidence.iloc[-1] = adj_conf

            # Si el modificador de señal es extremo, considerar override
            modifier = sentiment.get_signal_modifier()
            if modifier != 0 and last_sig == 0 and adj_conf < 0.35:
                # Sentiment extremo puede activar señal débil
                logger.info(
                    f"📰 Sentiment override: {last_sig} → {modifier} "
                    f"(score={sentiment.aggregate_score:+.2f})"
                )

            logger.info(
                f"📰 Sentiment aplicado | score={sentiment.aggregate_score:+.2f} | "
                f"conf: {last_conf:.2%} → {adj_conf:.2%} | "
                f"label={sentiment.sentiment_label}"
            )

        except Exception as e:
            logger.warning(f"NewsFilter falló (continuando sin él): {e}")

        return results

    def get_last_sentiment(self) -> Optional[object]:
        """Retorna el último SentimentResult analizado."""
        return self._last_sentiment

    # ─────────────────────────────────────────────────────────
    # Cálculo dinámico de SL/TP con ATR
    # ─────────────────────────────────────────────────────────
    def _calculate_atr_sltp(
        self,
        df: pd.DataFrame,
        signal: int,
        current_price: float,
    ) -> Tuple[float, float, float]:
        """
        Calcula Stop Loss y Take Profit dinámicamente basados en ATR.

        Parámetros:
            df: DataFrame con columnas High, Low, Close
            signal: Señal {1=BUY, -1=SELL, 0=NEUTRAL}
            current_price: Precio actual (típicamente Close del último candle)

        Retorna:
            (sl_price, tp_price, atr_value)

        Ecuaciones:
            ATR = promedio de {High-Low, |High-Close_prev|, |Low-Close_prev|}
            Si signal == 1 (BUY):
                SL = current_price - (ATR × atr_sl_multiplier)
                TP = current_price + (ATR × atr_tp_multiplier)
            Si signal == -1 (SELL):
                SL = current_price + (ATR × atr_sl_multiplier)
                TP = current_price - (ATR × atr_tp_multiplier)
            Si signal == 0 (NEUTRAL):
                SL = TP = current_price
        """
        # Usar últimos 14 candles para ATR (estándar en análisis técnico)
        lookback = min(14, len(df))
        if lookback < 2:
            # Si no hay suficientes datos, usar SL/TP simples
            logger.warning("Datos insuficientes para ATR — usando SL/TP por defecto")
            self._last_atr = 0.0
            return (current_price * 0.985, current_price * 1.015, 0.0)

        df_tail = df.iloc[-lookback:].copy()

        # Componentes del ATR
        hl = df_tail['High'] - df_tail['Low']  # High - Low
        hc = np.abs(df_tail['High'] - df_tail['Close'].shift(1))  # |High - Close_prev|
        lc = np.abs(df_tail['Low'] - df_tail['Close'].shift(1))   # |Low - Close_prev|

        # ATR = media móvil de estos componentes
        tr = np.maximum(hl, np.maximum(hc, lc))  # True Range
        atr = tr.mean()

        # Guardar ATR para referencia
        self._last_atr = float(atr)

        # Calcular SL y TP según señal
        if signal == 1:  # BUY
            sl_price = current_price - (atr * self.cfg.atr_sl_multiplier)
            tp_price = current_price + (atr * self.cfg.atr_tp_multiplier)
        elif signal == -1:  # SELL
            sl_price = current_price + (atr * self.cfg.atr_sl_multiplier)
            tp_price = current_price - (atr * self.cfg.atr_tp_multiplier)
        else:  # NEUTRAL
            sl_price = tp_price = current_price

        logger.debug(
            f"ATR={atr:.4f} | Signal={signal:+d} | "
            f"SL={sl_price:.2f} | TP={tp_price:.2f}"
        )

        return (sl_price, tp_price, atr)

    # ─────────────────────────────────────────────────────────
    # Aplicación de filtros de trading
    # ─────────────────────────────────────────────────────────
    def _apply_trading_filters(
        self,
        result: "PredictionResult",
        mtf_data: Optional[Dict[str, pd.DataFrame]] = None,
        dxy_data: Optional[pd.DataFrame] = None,
    ) -> "PredictionResult":
        """
        Aplica filtros de sesión, multi-timeframe y DXY a la predicción.

        Parámetros:
            result: PredictionResult original (con action, confidence, etc.)
            mtf_data: Dict con datos de H1, H4, D1 para análisis multi-timeframe
            dxy_data: DataFrame con datos del DXY

        Comportamiento:
            - Si no hay filtros disponibles: retorna result sin cambios
            - Evalúa contexto de trading (sesión, MTF, DXY)
            - Si no es tradeable (fuera de sesiones activas):
              * Cambia action a "NEUTRAL"
              * Añade nota en result.notes
            - Multiplica confidence por overall_multiplier (con clip [0, 1])
            - Almacena contexto en result para auditoría

        Retorna:
            result modificado
        """
        # Si no hay filtros, pasar directo
        if self._filters is None:
            return result

        try:
            # Evaluar contexto completo (sesión + MTF + DXY)
            context = self._filters.evaluate(
                mtf_data or {},
                dxy_data
            )
            self._last_trading_context = context

            # Si no es tradeable, neutralizar señal
            if not context.is_tradeable:
                logger.info(
                    f"⚠️ No es tradeable ({context.session}) — "
                    f"neutralizando señal | Recomendación: {context.recommendation}"
                )
                result.action = "NEUTRAL"
                result.notes = f"{result.notes} [Neutralizado por sesión: {context.session}]"

            # Aplicar multiplicador de confianza
            old_conf = result.confidence
            result.confidence = np.clip(
                result.confidence * context.overall_multiplier,
                0.0,
                1.0
            )

            logger.debug(
                f"Filtros aplicados | "
                f"Sesión: {context.session} | "
                f"MTF alineamiento: {context.mtf_signal.get('alignment_score', 0):.2f} | "
                f"Conf: {old_conf:.2%} → {result.confidence:.2%}"
            )

            # Guardar contexto en result para auditoría
            result.trading_context = context

        except Exception as e:
            logger.warning(f"Error aplicando filtros de trading: {e}")

        return result

    # ─────────────────────────────────────────────────────────
    # Métodos accesores para datos de filtros y tracker
    # ─────────────────────────────────────────────────────────
    def get_last_trading_context(self) -> Optional[TradingContext]:
        """
        Retorna el último TradingContext evaluado.
        Util para auditoría y debugging de decisiones de filtros.
        """
        return self._last_trading_context

    def get_performance_stats(self) -> Optional[object]:
        """
        Obtiene estadísticas agregadas de desempeño desde el tracker.

        Retorna:
            PerformanceStats si tracker está disponible, None en caso contrario
        """
        if self._tracker is None:
            return None
        try:
            return self._tracker.get_statistics()
        except Exception as e:
            logger.warning(f"Error obteniendo estadísticas de desempeño: {e}")
            return None

    def get_recommended_lot(self, account_balance: float = 10000.0) -> float:
        """
        Obtiene el tamaño de lote recomendado basado en desempeño histórico.

        Parámetros:
            account_balance: Saldo de la cuenta en USD

        Retorna:
            Tamaño de lote recomendado (ej: 0.01 para micro)
            Si no hay tracker: retorna 0.01 por defecto (micro lot)
        """
        if self._tracker is None:
            logger.debug("Tracker no disponible — retornando micro lot (0.01)")
            return 0.01

        try:
            stats = self._tracker.get_statistics()
            if stats is None or stats.win_rate < 0.40:
                logger.info("Desempeño insuficiente — tamaño conservador (0.01)")
                return 0.01

            # Usar fórmula de Kelly con factor de seguridad (50% del Kelly)
            kelly_fraction = (stats.win_rate - (1 - stats.win_rate) / stats.avg_win_loss) / 2
            kelly_fraction = max(0.01, min(0.05, kelly_fraction))  # Limitar [0.01, 0.05]

            lot_size = kelly_fraction * account_balance / 100000.0  # Ajustar escala

            logger.info(f"Tamaño de lote recomendado: {lot_size:.4f} (Kelly: {kelly_fraction:.2%})")
            return max(0.01, min(0.1, lot_size))

        except Exception as e:
            logger.warning(f"Error calculando lote recomendado: {e}")
            return 0.01

    def update_signal_outcome(
        self,
        signal_id: str,
        outcome: str,
        exit_price: float,
    ) -> bool:
        """
        Actualiza el resultado de una señal ya generada.

        Parámetros:
            signal_id: ID único de la señal a actualizar
            outcome: "TP_HIT", "SL_HIT", "MANUAL_CLOSE", etc.
            exit_price: Precio de salida

        Retorna:
            True si se actualizó exitosamente, False si no se encontró la señal
        """
        if self._tracker is None:
            logger.debug("Tracker no disponible — no se puede actualizar outcome")
            return False

        try:
            return self._tracker.update_signal_outcome(signal_id, outcome, exit_price)
        except Exception as e:
            logger.warning(f"Error actualizando outcome de señal {signal_id}: {e}")
            return False

    # ── Combinación de señales ────────────────────────────────
    def _combine(
        self,
        rf_sigs:   pd.Series,
        rf_proba:  np.ndarray,    # (N, 3) ← [P(-1), P(0), P(+1)]
        lstm_sigs: pd.Series,
        lstm_proba: np.ndarray,   # (N, 3)
        regimes:   pd.Series,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Combina señales de RF y LSTM con pesos, luego aplica filtro HMM.
        """
        cfg = self.cfg
        n   = len(rf_sigs)

        # ── Voto ponderado sobre probabilidad ──────────────────
        # P_ensemble(class) = w_rf * P_rf + w_lstm * P_lstm
        w_rf   = cfg.rf_weight   / (cfg.rf_weight + cfg.lstm_weight)
        w_lstm = cfg.lstm_weight / (cfg.rf_weight + cfg.lstm_weight)
        ensemble_proba = w_rf * rf_proba[:n] + w_lstm * lstm_proba[:n]

        # Mejor clase
        best_class  = np.argmax(ensemble_proba, axis=1)   # {0,1,2}
        class_to_signal = {0: -1, 1: 0, 2: 1}
        raw_signals = np.array([class_to_signal[c] for c in best_class])
        confidence  = ensemble_proba[np.arange(n), best_class]

        # ── Convertir a Series ─────────────────────────────────
        idx         = rf_sigs.index
        sig_series  = pd.Series(raw_signals, index=idx)
        conf_series = pd.Series(confidence,  index=idx)

        # ── Filtro de confianza ────────────────────────────────
        low_conf = conf_series < cfg.min_ensemble_confidence
        sig_series[low_conf] = 0

        # ── Filtro HMM ─────────────────────────────────────────
        filtered_sigs = GoldRegimeDetector.filter_signals(
            sig_series, regimes,
            cancel_buy_in_bear  = cfg.cancel_buy_in_bear,
            cancel_sell_in_bull = cfg.cancel_sell_in_bull,
        )

        return filtered_sigs, conf_series, regimes

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Llama a fit() primero.")


# ─────────────────────────────────────────────────────────────
# Función auxiliar: Reordenar columnas de probabilidad
# ─────────────────────────────────────────────────────────────
def _reorder_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    Reordena columnas de probabilidad para que sean: [P(-1), P(0), P(+1)].
    classes: Arreglo original de sklearn (como [-1, 0, 1]).
    """
    target_order = [-1, 0, 1]
    result       = np.zeros((len(proba), 3))
    for i, cls in enumerate(target_order):
        if cls in classes:
            col = list(classes).index(cls)
            result[:, i] = proba[:, col]
    return result


# ─────────────────────────────────────────────────────────────
# Datos simulados para prueba independiente
# ─────────────────────────────────────────────────────────────
def _make_mock_data(n: int = 400, seed: int = 42):
    """Genera datos sintéticos para prueba de Ensemble."""
    from config import AppConfig
    from data_fetcher import GoldDataFetcher
    from indicators import TechnicalIndicators
    from feature_engineer import FeatureEngineer

    config  = AppConfig(mode="mock")
    config.mock.n_rows = n
    df_raw  = GoldDataFetcher(config).get_data()
    df_ind  = TechnicalIndicators(df_raw, config).add_all().get_dataframe()
    fe      = FeatureEngineer(df_ind, target_threshold=0.003)
    fe.build_features()
    df_feat = fe.get_full_data()
    return df_feat, df_ind


# ─────────────────────────────────────────────────────────────
# Pruebas unitarias / Aserciones
# ─────────────────────────────────────────────────────────────
def _run_assertions():
    """Verificación rápida de validez del modelo Ensemble — se ejecuta en importación directa."""
    print("  ← Ejecutando Aserciones para Ensemble ...")

    df_feat, df_raw = _make_mock_data(400)

    cfg = EnsembleConfig(
        rf_weight=0.40, lstm_weight=0.40, hmm_weight=0.20,
        lstm_config=LSTMConfig(hidden_size=16, seq_length=10,
                               epochs=5, patience=3),
    )
    ensemble = GoldEnsemble(cfg)

    # 1) Entrenamiento sin errores
    ensemble.fit(df_feat, df_raw)
    assert ensemble._is_trained, "Debe estar entrenado"

    # 2) Predicción
    results = ensemble.predict(df_feat, df_raw)

    # 3) Señales en {-1, 0, 1}
    assert set(results.signals.unique()).issubset({-1, 0, 1}), \
        "Las señales deben estar en {-1, 0, 1}"

    # 4) Confianza entre 0 y 1
    assert (results.confidence >= 0).all() and (results.confidence <= 1).all(), \
        "La confianza debe estar en [0, 1]"

    # 5) Regímenes en {0, 1, 2}
    assert set(results.regimes.unique()).issubset({0, 1, 2}), \
        "Los regímenes deben estar en {0, 1, 2}"

    # 6) Sin compra en Bear (si filtro HMM está habilitado)
    bear_idx = results.regimes[results.regimes == 0].index
    if len(bear_idx) > 0:
        overlap = results.signals.index.intersection(bear_idx)
        if len(overlap) > 0:
            assert (results.signals.loc[overlap] != 1).all(), \
                "No debe haber señales de compra en Bear"

    print("  ✅ ¡Todas las aserciones de Ensemble pasaron exitosamente!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_assertions()
