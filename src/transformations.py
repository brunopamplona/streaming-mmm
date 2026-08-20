"""
transformations.py
==================
Funções de transformação de mídia para MMM:

  - Adstock geométrico (carryover / efeito residual)
  - Saturação de Hill (retornos decrescentes)
  - Pipeline completo por canal (adstock → Hill)
  - Grid search de parâmetros (gamma, kappa, lambda)

Cada função é documentada com a formulação matemática,
os parâmetros esperados e as referências bibliográficas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple


# ═══════════════════════════════════════════════════════════════
#  1.  ADSTOCK GEOMÉTRICO
# ═══════════════════════════════════════════════════════════════

def geometric_adstock(
    spend: np.ndarray,
    lambda_decay: float,
) -> np.ndarray:
    """
    Adstock geométrico — modela o carryover (efeito residual) da mídia.

    Fórmula:
        adstock_t = spend_t + λ · adstock_{t-1}

    onde λ ∈ [0, 1) é a taxa de decaimento:
      - λ próximo de 0 → efeito imediato, sem memória (ex.: Paid Search)
      - λ próximo de 1 → efeito prolongado, muita memória (ex.: TV, YouTube)

    Parâmetros
    ----------
    spend : np.ndarray
        Vetor de gastos brutos semanais, shape (T,).
    lambda_decay : float
        Taxa de decaimento ∈ [0, 1).

    Retorna
    -------
    np.ndarray
        Série de adstock, mesma shape de `spend`.

    Referências
    -----------
    Koyck (1954); Jin et al. (2017) — Bayesian Methods for Media Mix Modeling
    with Carryover and Shape Effects. Google Research.
    """
    if not 0.0 <= lambda_decay < 1.0:
        raise ValueError(f"lambda_decay deve estar em [0, 1). Recebido: {lambda_decay}")

    adstock = np.empty_like(spend, dtype=float)
    adstock[0] = float(spend[0])
    for t in range(1, len(spend)):
        adstock[t] = float(spend[t]) + lambda_decay * adstock[t - 1]
    return adstock


def adstock_halflife(lambda_decay: float) -> float:
    """
    Calcula a meia-vida do adstock em semanas.

    half_life = -ln(2) / ln(lambda_decay)

    Intuitivo: depois de `half_life` semanas sem investimento,
    metade do efeito acumulado se dissipou.
    """
    if lambda_decay <= 0:
        return 0.0
    return -np.log(2) / np.log(lambda_decay)


# ═══════════════════════════════════════════════════════════════
#  2.  SATURAÇÃO DE HILL
# ═══════════════════════════════════════════════════════════════

def hill_saturation(
    x: np.ndarray,
    gamma: float,
    kappa: float,
) -> np.ndarray:
    """
    Função de saturação de Hill — modela retornos decrescentes.

    Fórmula:
        S(x) = x^γ / (x^γ + κ^γ)

    Propriedades:
      - S(0) = 0          → zero gasto, zero efeito
      - S(κ) = 0.5        → κ é o ponto de meia-saturação
      - S(x) → 1 quando x → ∞   (assíntota superior)
      - γ > 1 → curva sigmoidal (efeito mínimo abaixo de certo threshold)
      - γ ≤ 1 → curva côncava desde a origem (retornos decrescentes imediatos)

    Parâmetros
    ----------
    x : np.ndarray
        Série de adstock (pós-transformação geométrica), shape (T,).
    gamma : float
        Parâmetro de forma (shape) > 0.
    kappa : float
        Ponto de meia-saturação > 0 (mesma unidade de `x`).

    Retorna
    -------
    np.ndarray
        Valores de saturação ∈ [0, 1), mesma shape de `x`.

    Referências
    -----------
    Hill (1910); Jin et al. (2017); Chan & Perry (2017) — Challenges and
    Opportunities in Media Mix Modeling. Google, Inc.
    """
    if gamma <= 0:
        raise ValueError(f"gamma deve ser > 0. Recebido: {gamma}")
    if kappa <= 0:
        raise ValueError(f"kappa deve ser > 0. Recebido: {kappa}")

    x_safe = np.maximum(x, 0.0)
    x_g    = np.power(x_safe, gamma)
    k_g    = np.power(kappa, gamma)
    return x_g / (x_g + k_g)


# ═══════════════════════════════════════════════════════════════
#  3.  PIPELINE COMPLETO POR CANAL
# ═══════════════════════════════════════════════════════════════

def transform_channel(
    spend: np.ndarray,
    lambda_decay: float,
    gamma: float,
    kappa: float,
) -> np.ndarray:
    """
    Pipeline de transformação: adstock geométrico → saturação de Hill.

    Aplica as duas transformações em sequência e retorna a série
    pronta para entrar na regressão OLS do MMM.

    Parâmetros
    ----------
    spend : np.ndarray
        Gastos brutos semanais.
    lambda_decay : float
        Taxa de decaimento do adstock ∈ [0, 1).
    gamma : float
        Parâmetro de forma da Hill > 0.
    kappa : float
        Ponto de meia-saturação da Hill > 0.

    Retorna
    -------
    np.ndarray
        Série transformada ∈ [0, 1).
    """
    adstock = geometric_adstock(spend, lambda_decay)
    return hill_saturation(adstock, gamma, kappa)


def transform_all_channels(
    df_spend: pd.DataFrame,
    channel_params: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """
    Aplica o pipeline adstock → Hill a todos os canais de mídia.

    Parâmetros
    ----------
    df_spend : pd.DataFrame
        DataFrame com uma coluna de spend por canal.
        Exemplo de colunas: ['spend_paid_search', 'spend_meta_ads', ...]
    channel_params : dict
        Dicionário no formato:
        {
            "nome_canal": {
                "lambda_decay": float,
                "gamma": float,
                "kappa": float,
            },
            ...
        }

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas 'media_<canal>' — série transformada
        pronta para a regressão.
    """
    result = {}
    for ch_name, params in channel_params.items():
        col_key  = ch_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        spend_col = f"spend_{col_key}"

        if spend_col not in df_spend.columns:
            raise KeyError(f"Coluna '{spend_col}' não encontrada no DataFrame.")

        transformed = transform_channel(
            df_spend[spend_col].values,
            lambda_decay=params["lambda_decay"],
            gamma=params["gamma"],
            kappa=params["kappa"],
        )
        result[f"media_{col_key}"] = transformed

    return pd.DataFrame(result, index=df_spend.index)


# ═══════════════════════════════════════════════════════════════
#  4.  GRID SEARCH / RANDOM SEARCH DE PARÂMETROS
# ═══════════════════════════════════════════════════════════════

def random_search_params(
    spend: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    n_iter: int = 200,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Busca aleatória dos parâmetros de adstock e Hill para um único canal.

    Minimiza o RSS (soma dos quadrados dos resíduos) de uma regressão
    simples canal + controles sobre a série `y`.

    Parâmetros
    ----------
    spend : np.ndarray
        Gastos brutos semanais do canal, shape (T,).
    y : np.ndarray
        Variável dependente (novos assinantes), shape (T,).
    controls : np.ndarray
        Matriz de variáveis de controle (sazonalidade, tendência etc.),
        shape (T, k).
    n_iter : int
        Número de combinações aleatórias a testar.
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    Tuple[float, float, float, float]
        (best_lambda, best_gamma, best_kappa, best_rss)
    """
    rng = np.random.default_rng(seed)

    lambda_grid = rng.uniform(0.01, 0.80, n_iter)
    gamma_grid  = rng.uniform(0.30, 1.50, n_iter)
    kappa_grid  = rng.uniform(spend.mean() * 0.3, spend.mean() * 3.0, n_iter)

    best_rss    = np.inf
    best_lambda = lambda_grid[0]
    best_gamma  = gamma_grid[0]
    best_kappa  = kappa_grid[0]

    for lam, gam, kap in zip(lambda_grid, gamma_grid, kappa_grid):
        transformed = transform_channel(spend, lam, gam, kap).reshape(-1, 1)

        # Regressão OLS mínima (X = [transformed | controls | intercept])
        X = np.hstack([
            transformed,
            controls,
            np.ones((len(y), 1)),
        ])
        try:
            coef, rss_arr, _, _ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(rss_arr[0]) if len(rss_arr) > 0 else np.sum((y - X @ coef) ** 2)
        except np.linalg.LinAlgError:
            continue

        if rss < best_rss:
            best_rss    = rss
            best_lambda = lam
            best_gamma  = gam
            best_kappa  = kap

    return best_lambda, best_gamma, best_kappa, best_rss


# ═══════════════════════════════════════════════════════════════
#  5.  UTILITÁRIOS DE DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════

def saturation_curve_points(
    gamma: float,
    kappa: float,
    n_points: int = 200,
    x_max_multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera pontos para plotar a curva de saturação de Hill.

    Retorna
    -------
    Tuple[np.ndarray, np.ndarray]
        (x_values, saturation_values)
    """
    x = np.linspace(0, kappa * x_max_multiplier, n_points)
    s = hill_saturation(x, gamma, kappa)
    return x, s


def marginal_return(
    x: float,
    gamma: float,
    kappa: float,
) -> float:
    """
    Derivada da função de Hill em relação a x.

    dS/dx = gamma · kappa^gamma · x^(gamma-1) / (x^gamma + kappa^gamma)^2

    Interpreta-se como: quantos assinantes incrementais são gerados
    por cada R$ 1 adicional de spend (antes do coeficiente de regressão).
    """
    if x <= 0:
        return 0.0
    num = gamma * (kappa ** gamma) * (x ** (gamma - 1))
    den = (x ** gamma + kappa ** gamma) ** 2
    return num / den
