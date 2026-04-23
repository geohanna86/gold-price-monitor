# ============================================================
# lstm_model.py — Red LSTM con numpy puro (sin marco de trabajo externo)
# Gold Price Monitor — Phase 3
#
# Arquitectura:
#   Input(seq=15, features=N) → LSTM(hidden=48) → Dense(3) → Softmax
#
# Algoritmo:
#   - Forward Pass: ecuaciones LSTM cuatro (f, i, g, o)
#   - Backward Pass: BPTT (Backpropagation Through Time)
#   - Optimizer: Adam (lr=0.001, β1=0.9, β2=0.999)
#   - Loss: Categorical Cross-Entropy
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from feature_engineer import FEATURE_COLUMNS, TARGET_COLUMN

logger = logging.getLogger("LSTMModel")


# ─────────────────────────────────────────────────────────────
# Configuración del modelo
# ─────────────────────────────────────────────────────────────
@dataclass
class LSTMConfig:
    hidden_size:   int   = 48      # Tamaño de la memoria interna
    seq_length:    int   = 15      # Longitud de la ventana temporal (días)
    output_size:   int   = 3       # Número de clases: {-1, 0, +1}
    epochs:        int   = 80      # Ciclos de entrenamiento
    batch_size:    int   = 16      # Tamaño del lote
    learning_rate: float = 0.001   # Tasa de aprendizaje
    dropout:       float = 0.15    # Tasa de Dropout (reducción de Overfitting)
    grad_clip:     float = 5.0     # Gradient Clipping (evita explosión)
    patience:      int   = 15      # Early Stopping (detén si no mejora)
    random_state:  int   = 42


# ─────────────────────────────────────────────────────────────
# Funciones de activación
# ─────────────────────────────────────────────────────────────
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1 / (1 + np.exp(-np.clip(x, -500, 500))),
                    np.exp(np.clip(x, -500, 0)) /
                    (1 + np.exp(np.clip(x, -500, 0))))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(x, -500, 500))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)


def _cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    return -np.log(probs[np.arange(n), labels] + 1e-9).mean()


# ─────────────────────────────────────────────────────────────
# Celda LSTM — un paso temporal
# ─────────────────────────────────────────────────────────────
class LSTMCell:
    """
    Una celda LSTM — implementación de ecuaciones Hochreiter & Schmidhuber (1997).

    Ecuaciones:
        z  = concat(h_{t-1}, x_t)          ← Combinación del estado y entrada
        f  = σ(W_f @ z + b_f)              ← Puerta de olvido
        i  = σ(W_i @ z + b_i)              ← Puerta de entrada
        g  = tanh(W_g @ z + b_g)           ← Contenido nuevo
        o  = σ(W_o @ z + b_o)              ← Puerta de salida
        C_t = f ⊙ C_{t-1} + i ⊙ g         ← Actualización de memoria
        h_t = o ⊙ tanh(C_t)               ← Salida
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng   = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        z_dim = input_size + hidden_size   # Tamaño del vector combinado

        # Pesos de las cuatro puertas: (hidden_size × z_dim)
        self.W_f = rng.randn(hidden_size, z_dim) * scale
        self.W_i = rng.randn(hidden_size, z_dim) * scale
        self.W_g = rng.randn(hidden_size, z_dim) * scale
        self.W_o = rng.randn(hidden_size, z_dim) * scale

        # Sesgos de las puertas — f comienza con 1 (técnica Forget Gate Bias Trick)
        self.b_f = np.ones(hidden_size)
        self.b_i = np.zeros(hidden_size)
        self.b_g = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

        self.hidden_size = hidden_size
        self.input_size  = input_size
        self._cache: List = []

    def forward_step(
        self,
        x: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Un paso temporal hacia adelante.
        Retorna: (h_t, c_t)
        """
        z  = np.concatenate([h_prev, x])
        f  = _sigmoid(self.W_f @ z + self.b_f)
        i  = _sigmoid(self.W_i @ z + self.b_i)
        g  = _tanh(   self.W_g @ z + self.b_g)
        o  = _sigmoid(self.W_o @ z + self.b_o)
        c_t = f * c_prev + i * g
        h_t = o * _tanh(c_t)

        # Guardar para Backward
        self._cache.append((z, f, i, g, o, c_prev, c_t, h_prev))
        return h_t, c_t

    def forward_sequence(
        self,
        X_seq: np.ndarray,
        h0: np.ndarray,
        c0: np.ndarray,
        training: bool = False,
        dropout: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pasa una secuencia completa a través de la celda.
        X_seq: (seq_len, input_size)
        Retorna h_T final y c_T final.
        """
        self._cache.clear()
        h, c = h0.copy(), c0.copy()
        for t in range(X_seq.shape[0]):
            h, c = self.forward_step(X_seq[t], h, c)
            if training and dropout > 0:
                mask = (np.random.rand(*h.shape) > dropout) / (1 - dropout)
                h    = h * mask
        return h, c

    def backward_sequence(
        self,
        dh_T: np.ndarray,
        dc_T: np.ndarray,
        clip: float = 5.0,
    ) -> Tuple[dict, np.ndarray]:
        """
        BPTT — Calcula gradientes a través del tiempo desde el paso final.
        Retorna: (gradients_dict, dx_0) donde dx_0 es el gradiente de la primera entrada.
        """
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_g = np.zeros_like(self.W_g)
        dW_o = np.zeros_like(self.W_o)
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_g = np.zeros_like(self.b_g)
        db_o = np.zeros_like(self.b_o)

        dh_next = dh_T.copy()
        dc_next = dc_T.copy()

        for z, f, i, g, o, c_prev, c_t, h_prev in reversed(self._cache):
            tanh_ct = _tanh(c_t)

            # Gradientes a través de h_t = o ⊙ tanh(c_t)
            do    = dh_next * tanh_ct
            dc    = dh_next * o * (1 - tanh_ct ** 2) + dc_next
            df    = dc * c_prev
            di    = dc * g
            dg    = dc * i
            dc_next = dc * f

            # Gradientes a través de funciones de activación
            do_raw = do * o * (1 - o)
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            dg_raw = dg * (1 - g ** 2)

            dW_o += np.outer(do_raw, z);  db_o += do_raw
            dW_f += np.outer(df_raw, z);  db_f += df_raw
            dW_i += np.outer(di_raw, z);  db_i += di_raw
            dW_g += np.outer(dg_raw, z);  db_g += dg_raw

            dz = (self.W_f.T @ df_raw + self.W_i.T @ di_raw
                + self.W_g.T @ dg_raw + self.W_o.T @ do_raw)
            dh_next = dz[:self.hidden_size]

        grads = {
            "W_f": np.clip(dW_f, -clip, clip),
            "W_i": np.clip(dW_i, -clip, clip),
            "W_g": np.clip(dW_g, -clip, clip),
            "W_o": np.clip(dW_o, -clip, clip),
            "b_f": np.clip(db_f, -clip, clip),
            "b_i": np.clip(db_i, -clip, clip),
            "b_g": np.clip(db_g, -clip, clip),
            "b_o": np.clip(db_o, -clip, clip),
        }
        return grads, dh_next


# ─────────────────────────────────────────────────────────────
# Adam Optimizer
# ─────────────────────────────────────────────────────────────
class AdamOptimizer:
    """Adam optimizer — actualiza los pesos inteligentemente basado en el momentum."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._m: dict = {}
        self._v: dict = {}
        self._t: int  = 0

    def update(self, params: dict, grads: dict):
        self._t += 1
        for key in grads:
            if key not in self._m:
                self._m[key] = np.zeros_like(grads[key])
                self._v[key] = np.zeros_like(grads[key])
            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grads[key]
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────
# Modelo completo
# ─────────────────────────────────────────────────────────────
class GoldLSTM:
    """
    Modelo LSTM completo para predicción de señales de oro.

    Uso:
        model = GoldLSTM(LSTMConfig())
        model.fit(X_train_seq, y_train)
        preds = model.predict(X_test_seq)
    """

    CLASS_MAP  = {0: -1, 1: 0, 2: 1}   # Índice de celda → valor de señal
    LABEL_MAP  = {-1: 0,  0: 1, 1: 2}  # Valor de señal → índice de celda

    def __init__(self, input_size: int, config: LSTMConfig = None):
        self.cfg         = config or LSTMConfig()
        self.input_size  = input_size
        self._is_trained = False

        np.random.seed(self.cfg.random_state)
        H = self.cfg.hidden_size

        # Celda LSTM
        self.cell = LSTMCell(input_size, H, seed=self.cfg.random_state)

        # Capa de salida: (output_size × hidden_size)
        scale       = np.sqrt(2.0 / H)
        self.W_out  = np.random.randn(self.cfg.output_size, H) * scale
        self.b_out  = np.zeros(self.cfg.output_size)

        # Adam para cada conjunto de pesos
        self._opt_lstm = AdamOptimizer(lr=self.cfg.learning_rate)
        self._opt_out  = AdamOptimizer(lr=self.cfg.learning_rate)

        self.train_losses: List[float] = []
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std:  Optional[np.ndarray] = None

    # ── Preparación de secuencias ────────────────────────────
    @staticmethod
    def build_sequences(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforma datos aplanados en secuencias temporales.
        X_seq[i] = X[i : i+seq_len] ← entrada para el paso i+seq_len
        y_seq[i] = y[i+seq_len]
        """
        n        = len(X) - seq_len
        X_seqs   = np.array([X[i: i + seq_len] for i in range(n)])
        y_seqs   = y[seq_len:]
        return X_seqs, y_seqs

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._feature_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            self._feature_std  = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
        return (X - self._feature_mean) / self._feature_std

    # ── Forward Pass completo ───────────────────────────────
    def _forward(
        self, X_seq: np.ndarray, training: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        X_seq: (seq_len, input_size)
        Retorna: (probs, h_T)
        """
        H  = self.cfg.hidden_size
        h0 = np.zeros(H)
        c0 = np.zeros(H)
        h_T, _ = self.cell.forward_sequence(
            X_seq, h0, c0,
            training=training,
            dropout=self.cfg.dropout if training else 0.0,
        )
        logits = self.W_out @ h_T + self.b_out
        probs  = _softmax(logits)
        return probs, h_T

    # ── Backward Pass completo ──────────────────────────────
    def _backward(
        self,
        probs:  np.ndarray,
        h_T:    np.ndarray,
        label:  int,
    ) -> float:
        """
        Calcula los gradientes y actualiza los pesos.
        Retorna la pérdida.
        """
        # ── Gradiente de Cross-Entropy + Softmax ──
        y_onehot       = np.zeros(self.cfg.output_size)
        y_onehot[label] = 1
        d_logits       = probs - y_onehot      # (output_size,)

        # ── Gradientes de la capa de salida ──
        dW_out = np.outer(d_logits, h_T)
        db_out = d_logits.copy()
        dh_T   = self.W_out.T @ d_logits       # (hidden_size,)

        # ── BPTT a través de LSTM ──
        grads_lstm, _ = self.cell.backward_sequence(
            dh_T, np.zeros(self.cfg.hidden_size),
            clip=self.cfg.grad_clip,
        )

        # ── Actualizar pesos con Adam ──
        lstm_params = {
            "W_f": self.cell.W_f, "W_i": self.cell.W_i,
            "W_g": self.cell.W_g, "W_o": self.cell.W_o,
            "b_f": self.cell.b_f, "b_i": self.cell.b_i,
            "b_g": self.cell.b_g, "b_o": self.cell.b_o,
        }
        self._opt_lstm.update(lstm_params, grads_lstm)

        out_params = {"W_out": self.W_out, "b_out": self.b_out}
        out_grads  = {
            "W_out": np.clip(dW_out, -self.cfg.grad_clip, self.cfg.grad_clip),
            "b_out": np.clip(db_out, -self.cfg.grad_clip, self.cfg.grad_clip),
        }
        self._opt_out.update(out_params, out_grads)

        return -np.log(probs[label] + 1e-9)

    # ── Entrenamiento ──────────────────────────────────────────
    def fit(
        self,
        X_seq:  np.ndarray,   # (n_samples, seq_len, n_features)
        y_raw:  np.ndarray,   # valores de {-1, 0, +1}
    ) -> "GoldLSTM":
        """Entrena el modelo con Early Stopping."""
        y = np.array([self.LABEL_MAP[int(v)] for v in y_raw])  # {0,1,2}

        X_norm        = self._normalize(X_seq, fit=True)
        n             = len(X_norm)
        best_loss     = float("inf")
        patience_cnt  = 0
        cfg           = self.cfg

        logger.info(
            f"Iniciando entrenamiento LSTM | Muestras: {n} | "
            f"Seq={cfg.seq_length} | Hidden={cfg.hidden_size} | "
            f"Epochs={cfg.epochs}"
        )

        for epoch in range(cfg.epochs):
            # Mezcla aleatoria de lotes
            idx   = np.random.permutation(n)
            epoch_loss = 0.0

            for start in range(0, n, cfg.batch_size):
                batch_idx = idx[start: start + cfg.batch_size]
                batch_loss = 0.0
                for j in batch_idx:
                    probs, h_T = self._forward(X_norm[j], training=True)
                    loss       = self._backward(probs, h_T, y[j])
                    batch_loss += loss
                epoch_loss += batch_loss

            avg_loss = epoch_loss / n
            self.train_losses.append(avg_loss)

            # Early Stopping
            if avg_loss < best_loss - 1e-4:
                best_loss    = avg_loss
                patience_cnt = 0
                self._save_best()
            else:
                patience_cnt += 1

            if patience_cnt >= cfg.patience:
                logger.info(f"  Early Stopping en Epoch {epoch+1} | Pérdida: {avg_loss:.4f}")
                self._load_best()
                break

            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{cfg.epochs} | Pérdida: {avg_loss:.4f}")

        self._is_trained = True
        logger.info(f"✅ Entrenamiento LSTM completado | Mejor Pérdida: {best_loss:.4f}")
        return self

    # ── Guardar mejores pesos ──────────────────────────────────
    def _save_best(self):
        self._best = {
            "W_f": self.cell.W_f.copy(), "W_i": self.cell.W_i.copy(),
            "W_g": self.cell.W_g.copy(), "W_o": self.cell.W_o.copy(),
            "b_f": self.cell.b_f.copy(), "b_i": self.cell.b_i.copy(),
            "b_g": self.cell.b_g.copy(), "b_o": self.cell.b_o.copy(),
            "W_out": self.W_out.copy(),  "b_out": self.b_out.copy(),
        }

    def _load_best(self):
        if not hasattr(self, "_best"):
            return
        self.cell.W_f = self._best["W_f"]; self.cell.W_i = self._best["W_i"]
        self.cell.W_g = self._best["W_g"]; self.cell.W_o = self._best["W_o"]
        self.cell.b_f = self._best["b_f"]; self.cell.b_i = self._best["b_i"]
        self.cell.b_g = self._best["b_g"]; self.cell.b_o = self._best["b_o"]
        self.W_out    = self._best["W_out"]; self.b_out = self._best["b_out"]

    # ── Predicción ─────────────────────────────────────────────
    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        """Retorna señales {-1, 0, +1} para cada muestra."""
        self._check_trained()
        X_norm   = self._normalize(X_seq)
        results  = []
        for i in range(len(X_norm)):
            probs, _ = self._forward(X_norm[i], training=False)
            label    = int(np.argmax(probs))
            results.append(self.CLASS_MAP[label])
        return np.array(results)

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Retorna la probabilidad de cada clase. Columnas: [P(-1), P(0), P(+1)]."""
        self._check_trained()
        X_norm = self._normalize(X_seq)
        out    = np.zeros((len(X_norm), 3))
        for i in range(len(X_norm)):
            probs, _ = self._forward(X_norm[i], training=False)
            out[i]   = probs
        return out

    def _check_trained(self):
        if not self._is_trained:
            raise RuntimeError("Llama a fit() primero.")

    def is_trained(self) -> bool:
        return self._is_trained


# ─────────────────────────────────────────────────────────────
# Asistente: Preparar datos del Feature Engineer
# ─────────────────────────────────────────────────────────────
def prepare_lstm_data(
    full_df:    pd.DataFrame,
    seq_length: int = 15,
    test_size:  float = 0.20,
) -> Tuple:
    """
    Transforma DataFrame de características en secuencias LSTM.
    Retorna: (X_train_seq, X_test_seq, y_train, y_test, test_index)
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in full_df.columns]
    X_arr        = full_df[feature_cols].values.astype(np.float32)
    y_arr        = full_df[TARGET_COLUMN].values

    # Construir secuencias
    X_seq, y_seq = GoldLSTM.build_sequences(X_arr, y_arr, seq_length)
    index_seq    = full_df.index[seq_length:]

    # División temporal
    split      = int(len(X_seq) * (1 - test_size))
    X_train    = X_seq[:split]
    X_test     = X_seq[split:]
    y_train    = y_seq[:split]
    y_test     = y_seq[split:]
    test_index = index_seq[split:]

    return X_train, X_test, y_train, y_test, test_index
