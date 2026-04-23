# ============================================================
# ml_model.py — Modelo Random Forest para predicción de señales de oro
# Gold Price Monitor — Phase 2
#
# Librería: scikit-learn (pip install scikit-learn)
# Documentación: https://scikit-learn.org
#
# Lógica:
#  - Clasificación ternaria: +1 (compra) / -1 (venta) / 0 (neutro)
#  - TimeSeriesSplit: evaluación realista sin fuga temporal
#  - Feature Importance: ranking automático de importancia de indicadores
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("GoldPredictor")


# ── Configuración del modelo ──────────────────────────────────────────
@dataclass
class ModelConfig:
    # Random Forest
    n_estimators:     int   = 200     # Número de árboles (más = más preciso pero más lento)
    max_depth:        int   = 6       # Profundidad del árbol (previene Overfitting)
    min_samples_leaf: int   = 10      # Mínimo de muestras en cada hoja
    max_features:     str   = "sqrt"  # Número de características en cada split
    class_weight:     str   = "balanced"  # Manejo del desequilibrio de clases
    random_state:     int   = 42
    n_jobs:           int   = -1      # Usar todos los núcleos del procesador

    # Cross-Validation
    n_cv_splits:      int   = 5       # Número de splits en TimeSeriesSplit
    test_size_pct:    float = 0.20    # Porcentaje de datos de prueba


@dataclass
class ModelMetrics:
    """Resultados de evaluación del modelo."""
    accuracy:        float = 0.0
    f1_weighted:     float = 0.0
    precision:       float = 0.0
    recall:          float = 0.0
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std:  float = 0.0
    train_samples:   int   = 0
    test_samples:    int   = 0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    classification_report_str: str = ""


class GoldPredictor:
    """
    Modelo Random Forest para predicción de señales de trading de oro.

    Uso:
        predictor = GoldPredictor()
        predictor.train(X_train, y_train)
        signals   = predictor.predict(X_test)
        metrics   = predictor.evaluate(X_test, y_test)
    """

    def __init__(self, config: ModelConfig = None):
        self.config  = config or ModelConfig()
        self.model   = self._build_model()
        self.scaler  = StandardScaler()
        self.metrics: Optional[ModelMetrics] = None
        self._is_trained = False
        self._feature_names: List[str] = []

    # ─────────────────────────────────────────────────────────
    # Construcción del modelo
    # ─────────────────────────────────────────────────────────
    def _build_model(self) -> RandomForestClassifier:
        cfg = self.config
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            max_features=cfg.max_features,
            class_weight=cfg.class_weight,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )

    # ─────────────────────────────────────────────────────────
    # Entrenamiento
    # ─────────────────────────────────────────────────────────
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        run_cv: bool = True,
    ) -> "GoldPredictor":
        """
        Entrena el modelo en datos de entrenamiento.

        Parámetros:
            X_train:  Características de entrenamiento (sin datos futuros!)
            y_train:  Objetivo {-1, 0, +1}
            run_cv:   Ejecutar Cross-Validation para evaluación realista
        """
        self._feature_names = X_train.columns.tolist()

        logger.info(
            f"Iniciando entrenamiento | "
            f"Muestras: {len(X_train)} | "
            f"Características: {len(self._feature_names)} | "
            f"Distribución de clases: {dict(y_train.value_counts().sort_index())}"
        )

        # Escalado de características (ayuda a Random Forest aunque no es obligatorio)
        X_scaled = self.scaler.fit_transform(X_train)

        # ── Cross-Validation temporal ────────────────────────────
        cv_scores = []
        if run_cv and len(X_train) >= 50:
            tscv = TimeSeriesSplit(n_splits=self.config.n_cv_splits)
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_fold_train = X_scaled[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val   = X_scaled[val_idx]
                y_fold_val   = y_train.iloc[val_idx]

                fold_model = self._build_model()
                fold_model.fit(X_fold_train, y_fold_train)
                y_pred_fold = fold_model.predict(X_fold_val)
                score = accuracy_score(y_fold_val, y_pred_fold)
                cv_scores.append(score)
                logger.debug(f"  Fold {fold+1}: Accuracy = {score:.4f}")

            logger.info(
                f"Resultados CV | "
                f"Media: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
            )

        # ── Entrenamiento final en todos los datos de entrenamiento ──────────
        self.model.fit(X_scaled, y_train)
        self._is_trained = True

        # Almacenar resultados CV
        self._cv_scores = cv_scores

        logger.info("✅ Entrenamiento completado")
        return self

    # ─────────────────────────────────────────────────────────
    # Predicción
    # ─────────────────────────────────────────────────────────
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las señales predichas: {-1, 0, +1}."""
        self._check_trained()
        X_scaled = self.scaler.transform(X[self._feature_names])
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Devuelve la probabilidad de cada clase para cada muestra.
        Las columnas están en orden de self.model.classes_
        """
        self._check_trained()
        X_scaled = self.scaler.transform(X[self._feature_names])
        return self.model.predict_proba(X_scaled)

    def predict_with_confidence(
        self, X: pd.DataFrame, min_confidence: float = 0.45
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Devuelve señales con nivel de confianza.
        Las señales con confianza < min_confidence se convierten a 0 (neutro).

        Devuelve: (signals, confidence_scores)
        """
        self._check_trained()
        probas   = self.predict_proba(X)
        max_prob = probas.max(axis=1)
        raw_pred = self.predict(X)

        # Señales con confianza baja → neutro
        confident_pred = np.where(max_prob >= min_confidence, raw_pred, 0)
        return confident_pred, max_prob

    # ─────────────────────────────────────────────────────────
    # Evaluación
    # ─────────────────────────────────────────────────────────
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> ModelMetrics:
        """
        Evalúa el modelo en datos de prueba y devuelve métricas completas.
        """
        self._check_trained()

        y_pred = self.predict(X_test)
        labels = sorted(y_test.unique().tolist())

        accuracy   = accuracy_score(y_test, y_pred)
        f1         = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall     = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        # Importancia de características
        importance = dict(zip(
            self._feature_names,
            self.model.feature_importances_.round(4)
        ))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        report_str = classification_report(
            y_test, y_pred,
            target_names=["Venta (-1)", "Neutro (0)", "Compra (+1)"]
            if -1 in labels else ["Neutro (0)", "Compra (+1)"],
            zero_division=0,
        )

        self.metrics = ModelMetrics(
            accuracy=round(accuracy, 4),
            f1_weighted=round(f1, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            cv_accuracy_mean=round(np.mean(self._cv_scores), 4) if self._cv_scores else 0.0,
            cv_accuracy_std=round(np.std(self._cv_scores), 4)  if self._cv_scores else 0.0,
            train_samples=int(self.model.n_features_in_),
            test_samples=len(X_test),
            feature_importance=importance,
            classification_report_str=report_str,
        )

        logger.info(
            f"✅ Evaluación del modelo | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f} | "
            f"CV: {self.metrics.cv_accuracy_mean:.4f} ± {self.metrics.cv_accuracy_std:.4f}"
        )

        return self.metrics

    # ─────────────────────────────────────────────────────────
    # Presentación
    # ─────────────────────────────────────────────────────────
    def print_report(self):
        """Imprime un informe completo con los resultados del modelo."""
        if self.metrics is None:
            logger.warning("Llama evaluate() primero.")
            return

        m = self.metrics
        GREEN, RED, YELLOW, BLUE, GOLD, RESET, BOLD = (
            "\033[92m", "\033[91m", "\033[93m", "\033[94m",
            "\033[33m", "\033[0m", "\033[1m"
        )
        SEP = "=" * 55

        print(f"\n{GOLD}{BOLD}{SEP}")
        print(f"  🤖 Informe del modelo Random Forest — Oro XAU/USD")
        print(f"{SEP}{RESET}")

        print(f"\n{BLUE}{BOLD}  📊 Métricas de desempeño:{RESET}")
        color = GREEN if m.accuracy > 0.55 else YELLOW
        print(f"  {BOLD}Accuracy      :{RESET} {color}{m.accuracy:.2%}{RESET}")
        print(f"  {BOLD}F1 Weighted   :{RESET} {m.f1_weighted:.4f}")
        print(f"  {BOLD}Precision     :{RESET} {m.precision:.4f}")
        print(f"  {BOLD}Recall        :{RESET} {m.recall:.4f}")
        print(f"  {BOLD}CV Accuracy   :{RESET} {m.cv_accuracy_mean:.2%} "
              f"± {m.cv_accuracy_std:.2%}")

        print(f"\n{BLUE}{BOLD}  🏆 Top 10 indicadores (Feature Importance):{RESET}")
        for i, (feat, imp) in enumerate(list(m.feature_importance.items())[:10], 1):
            bar_len = int(imp * 50)
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            color   = GOLD if i <= 3 else RESET
            print(f"  {i:2}. {color}{feat:<20}{RESET} {bar} {imp:.4f}")

        print(f"\n{BLUE}{BOLD}  📋 Informe detallado de clasificación:{RESET}")
        print(m.classification_report_str)

    def get_top_features(self, n: int = 5) -> Dict[str, float]:
        """Devuelve las n características más importantes."""
        if self.metrics is None:
            return {}
        return dict(list(self.metrics.feature_importance.items())[:n])

    # ─────────────────────────────────────────────────────────
    # Funciones de apoyo
    # ─────────────────────────────────────────────────────────
    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Llama train() primero antes de predecir o evaluar.")

    def is_trained(self) -> bool:
        return self._is_trained

    def get_classes(self) -> np.ndarray:
        """Devuelve las clases que aprendió el modelo."""
        self._check_trained()
        return self.model.classes_
