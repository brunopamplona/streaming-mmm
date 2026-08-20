"""
validation.py
=============
Split treino/holdout temporal e métricas de validação do MMM.

Funções exportadas:
  - temporal_split()       → divide DataFrame em treino e holdout
  - compute_metrics()      → R², R² ajustado, MAPE, RMSE
  - holdout_week_by_week() → previsão semana a semana no período holdout
  - plot_holdout()         → gráfico fitted vs. actual no holdout
  - durbin_watson()        → estatística DW sobre resíduos
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ═══════════════════════════════════════════════════════════════
#  1.  SPLIT TEMPORAL
# ═══════════════════════════════════════════════════════════════

def temporal_split(
    df: pd.DataFrame,
    holdout_weeks: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o DataFrame em treino e holdout com base nas últimas semanas.

    ⚠ Importante: o split é temporal, não aleatório. As semanas de holdout
    são sempre as últimas `holdout_weeks` linhas — nunca misturadas com treino.
    Isso evita data leakage e simula como o modelo seria usado na prática.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame completo ordenado por semana (mais antigas primeiro).
    holdout_weeks : int
        Número de semanas a reservar para holdout. Padrão: 24 (~6 meses).

    Retorna
    -------
    (df_train, df_holdout) : Tuple[pd.DataFrame, pd.DataFrame]
    """
    if len(df) <= holdout_weeks:
        raise ValueError(
            f"DataFrame tem {len(df)} linhas, mas holdout_weeks={holdout_weeks}. "
            "Reduza holdout_weeks."
        )

    df_train   = df.iloc[:-holdout_weeks].copy()
    df_holdout = df.iloc[-holdout_weeks:].copy()

    return df_train, df_holdout


# ═══════════════════════════════════════════════════════════════
#  2.  MÉTRICAS DE VALIDAÇÃO
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: int = 0,
) -> dict:
    """
    Calcula métricas de qualidade de ajuste.

    Métricas calculadas:
      - R²            : coeficiente de determinação
      - R² ajustado   : penaliza pelo número de parâmetros
      - MAPE (%)      : erro percentual absoluto médio
      - RMSE          : raiz do erro quadrático médio
      - MAE           : erro absoluto médio

    Parâmetros
    ----------
    y_true : np.ndarray
        Valores reais observados.
    y_pred : np.ndarray
        Valores previstos pelo modelo.
    n_params : int
        Número de parâmetros estimados (para R² ajustado).

    Retorna
    -------
    dict com chaves: R2, R2_adj, MAPE_pct, RMSE, MAE
    """
    n      = len(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    r2_adj = (
        1 - (1 - r2) * (n - 1) / (n - n_params - 1)
        if (n - n_params - 1) > 0 else np.nan
    )

    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "R2":       round(float(r2),     4),
        "R2_adj":   round(float(r2_adj), 4),
        "MAPE_pct": round(mape, 4),
        "RMSE":     round(rmse, 2),
        "MAE":      round(mae, 2),
        "n_obs":    n,
    }


# ═══════════════════════════════════════════════════════════════
#  3.  PREVISÃO SEMANA A SEMANA NO HOLDOUT
# ═══════════════════════════════════════════════════════════════

def holdout_week_by_week(
    model,
    df_holdout: pd.DataFrame,
    target_col: str = "new_subscribers",
) -> pd.DataFrame:
    """
    Gera a previsão semana a semana no período de holdout
    e calcula o erro absoluto percentual por semana.

    Parâmetros
    ----------
    model : MMMModel
        Modelo já ajustado no período de treino.
    df_holdout : pd.DataFrame
        Dados do período de holdout.
    target_col : str
        Nome da coluna dependente.

    Retorna
    -------
    pd.DataFrame com colunas:
      week | actual | predicted | abs_error | abs_pct_error
    """
    y_true = df_holdout[target_col].values.astype(float)
    y_pred = model.predict(df_holdout)

    abs_err     = np.abs(y_true - y_pred)
    abs_pct_err = abs_err / np.maximum(np.abs(y_true), 1) * 100

    result = pd.DataFrame({
        "actual":        y_true,
        "predicted":     y_pred,
        "abs_error":     abs_err,
        "abs_pct_error": abs_pct_err,
    }, index=df_holdout.index)

    if "week" in df_holdout.columns:
        result.insert(0, "week", df_holdout["week"].values)

    return result


# ═══════════════════════════════════════════════════════════════
#  4.  ESTATÍSTICA DURBIN-WATSON
# ═══════════════════════════════════════════════════════════════

def durbin_watson(residuals: np.ndarray) -> float:
    """
    Calcula a estatística de Durbin-Watson para detecção de autocorrelação.

    DW = Σ(e_t - e_{t-1})² / Σ(e_t²)

    Interpretação:
      - DW ≈ 2.0 → sem autocorrelação
      - DW < 1.5 → autocorrelação positiva (comum em séries de marketing)
      - DW > 2.5 → autocorrelação negativa (raro)

    Por que importa no MMM?
    Autocorrelação residual significa que os erros-padrão OLS clássicos estão
    subestimados, inflando a significância dos coeficientes. É por isso que
    usamos HAC (Newey-West) no modelo final.
    """
    e     = np.asarray(residuals, dtype=float)
    diff  = np.diff(e)
    dw    = np.sum(diff ** 2) / np.sum(e ** 2)
    return float(dw)


# ═══════════════════════════════════════════════════════════════
#  5.  VISUALIZAÇÃO — FITTED vs. ACTUAL
# ═══════════════════════════════════════════════════════════════

def plot_holdout(
    holdout_df: pd.DataFrame,
    train_actual: Optional[np.ndarray] = None,
    train_fitted: Optional[np.ndarray] = None,
    title: str = "MMM — Validação Out-of-Sample (Holdout)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plota o ajuste in-sample e a previsão out-of-sample (holdout).

    Parâmetros
    ----------
    holdout_df : pd.DataFrame
        Resultado de holdout_week_by_week().
    train_actual : np.ndarray, opcional
        Valores reais do período de treino.
    train_fitted : np.ndarray, opcional
        Valores ajustados do período de treino.
    title : str
        Título do gráfico.
    save_path : str, opcional
        Caminho para salvar a figura (PNG). Se None, não salva.

    Retorna
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax_main, ax_err = axes

    n_holdout = len(holdout_df)

    # ── Período de treino ─────────────────────────────────────────
    if train_actual is not None and train_fitted is not None:
        x_train = np.arange(len(train_actual))
        ax_main.plot(x_train, train_actual, color="#1a1a2e", lw=1.2,
                     label="Real (treino)", alpha=0.8)
        ax_main.plot(x_train, train_fitted, color="#16213e", lw=1.0,
                     linestyle="--", label="Ajustado (treino)", alpha=0.7)
        offset = len(train_actual)
    else:
        offset = 0

    # ── Período de holdout ────────────────────────────────────────
    x_hold = np.arange(offset, offset + n_holdout)
    ax_main.axvspan(x_hold[0] - 0.5, x_hold[-1] + 0.5, alpha=0.07,
                    color="#e94560", label="Período holdout")
    ax_main.plot(x_hold, holdout_df["actual"].values, color="#e94560",
                 lw=1.8, label="Real (holdout)", marker="o", markersize=3)
    ax_main.plot(x_hold, holdout_df["predicted"].values, color="#0f3460",
                 lw=1.8, label="Previsto (holdout)", linestyle="--",
                 marker="s", markersize=3)

    # Métricas no título
    metrics = compute_metrics(
        holdout_df["actual"].values,
        holdout_df["predicted"].values,
    )
    ax_main.set_title(
        f"{title}\n"
        f"Holdout — R²: {metrics['R2']:.1%} | "
        f"MAPE: {metrics['MAPE_pct']:.2f}% | "
        f"RMSE: {metrics['RMSE']:,.0f} subs/semana",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax_main.set_ylabel("Novos Assinantes", fontsize=11)
    ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # ── Erro percentual semanal ───────────────────────────────────
    ax_err.bar(x_hold, holdout_df["abs_pct_error"].values,
               color="#e94560", alpha=0.7, width=0.8)
    ax_err.axhline(metrics["MAPE_pct"], color="#0f3460", lw=1.2,
                   linestyle="--", label=f"MAPE = {metrics['MAPE_pct']:.2f}%")
    ax_err.set_ylabel("Erro Abs. (%)", fontsize=10)
    ax_err.set_xlabel("Semana (#)", fontsize=10)
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"   Figura salva em: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════
#  6.  RESUMO COMPLETO DE VALIDAÇÃO
# ═══════════════════════════════════════════════════════════════

def validation_report(
    model,
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    target_col: str = "new_subscribers",
) -> dict:
    """
    Gera um dicionário com métricas de treino e holdout lado a lado.

    Parâmetros
    ----------
    model : MMMModel (já ajustado)
    df_train : pd.DataFrame
    df_holdout : pd.DataFrame
    target_col : str

    Retorna
    -------
    dict com chaves 'train' e 'holdout', cada uma contendo
    o resultado de compute_metrics().
    """
    n_params = len(model.feature_cols_)

    # Treino
    y_train      = df_train[target_col].values.astype(float)
    y_train_pred = model.result_.fittedvalues
    train_metrics = compute_metrics(y_train, y_train_pred, n_params)

    # Holdout
    y_hold      = df_holdout[target_col].values.astype(float)
    y_hold_pred = model.predict(df_holdout)
    hold_metrics = compute_metrics(y_hold, y_hold_pred, n_params)

    # DW nos resíduos de treino
    dw = durbin_watson(y_train - y_train_pred)

    return {
        "train":         train_metrics,
        "holdout":       hold_metrics,
        "durbin_watson": round(dw, 4),
        "n_train":       len(df_train),
        "n_holdout":     len(df_holdout),
    }
