"""
generate_synthetic_data.py
==========================
Gerador da base semanal sintética do MMM Video+.

Processo:
  1. Para cada canal, sorteia spend semanal realista
  2. Aplica adstock geométrico (carryover)
  3. Aplica saturação de Hill (retornos decrescentes)
  4. Combina via regressão linear com controles (sazonalidade, tendência, reajuste)
  5. Adiciona ruído gaussiano para simular variância residual real
  6. Salva em data/MMM_Synth_Weekly_Data.csv

Saída: 104 semanas (02/jan/2023 – 23/dez/2024)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reprodutibilidade ────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Parâmetros de cada canal ─────────────────────────────────────────────────
# Chave: nome do canal
# lambda_decay : taxa de decaimento do adstock geométrico
# gamma        : formato (shape) da curva de Hill
# kappa        : ponto de meia-saturação da curva de Hill
# spend_mean   : gasto semanal médio (R$)
# spend_std    : desvio-padrão do gasto semanal (R$)
# coef_true    : coeficiente verdadeiro (assinantes por unidade de saturação)
# cpa_target   : CPA-alvo de planejamento (R$) — referência de negócio

CHANNELS = {
    "Paid Search (Google)": {
        "lambda_decay": 0.05,
        "gamma": 0.70,
        "kappa": 150_000,
        "spend_mean": 400_000,
        "spend_std":  80_000,
        "coef_true":  9_500,
        "cpa_target": 45.0,
    },
    "Social Media (Meta Ads)": {
        "lambda_decay": 0.30,
        "gamma": 0.80,
        "kappa": 200_000,
        "spend_mean": 350_000,
        "spend_std":  70_000,
        "coef_true":  7_200,
        "cpa_target": 70.0,
    },
    "Social Media (YouTube Ads)": {
        "lambda_decay": 0.60,
        "gamma": 0.75,
        "kappa": 180_000,
        "spend_mean": 300_000,
        "spend_std":  60_000,
        "coef_true":  6_500,
        "cpa_target": 70.0,
    },
    "Affiliate Marketing": {
        "lambda_decay": 0.10,
        "gamma": 0.85,
        "kappa": 80_000,
        "spend_mean": 120_000,
        "spend_std":  25_000,
        "coef_true": 18_000,
        "cpa_target": 25.0,
    },
    "Email Marketing": {
        "lambda_decay": 0.05,
        "gamma": 0.90,
        "kappa": 30_000,
        "spend_mean":  50_000,
        "spend_std":  10_000,
        "coef_true": 22_000,
        "cpa_target": 30.0,
    },
    "E-commerce Company Site": {
        "lambda_decay": 0.15,
        "gamma": 0.80,
        "kappa": 100_000,
        "spend_mean": 180_000,
        "spend_std":  35_000,
        "coef_true": 13_000,
        "cpa_target": 70.0,
    },
    "Call Center": {
        "lambda_decay": 0.05,
        "gamma": 0.92,
        "kappa": 40_000,
        "spend_mean":  80_000,
        "spend_std":  15_000,
        "coef_true": 21_000,
        "cpa_target": 40.0,
    },
}

N_WEEKS   = 104
START_DATE = pd.Timestamp("2023-01-02")

# ── Funções de transformação ─────────────────────────────────────────────────

def geometric_adstock(spend: np.ndarray, lambda_decay: float) -> np.ndarray:
    """
    Adstock geométrico: carrega o efeito residual de semanas anteriores.

    adstock_t = spend_t + lambda * adstock_{t-1}
    """
    adstock = np.zeros_like(spend, dtype=float)
    adstock[0] = spend[0]
    for t in range(1, len(spend)):
        adstock[t] = spend[t] + lambda_decay * adstock[t - 1]
    return adstock


def hill_saturation(x: np.ndarray, gamma: float, kappa: float) -> np.ndarray:
    """
    Função de Hill (retornos decrescentes):

    S(x) = x^gamma / (x^gamma + kappa^gamma)

    Resultado no intervalo [0, 1).
    """
    x_g   = np.power(np.maximum(x, 0), gamma)
    k_g   = np.power(kappa, gamma)
    return x_g / (x_g + k_g)


# ── Índice de sazonalidade semanal ───────────────────────────────────────────
def seasonality_index(weeks: np.ndarray) -> np.ndarray:
    """
    Captura sazonalidade anual via combinação de sinusoides.
    Picos típicos: início de ano (jan), Copa/Olimpíadas (meio do ano),
    Black Friday/Natal (nov-dez).
    """
    angle = 2 * np.pi * weeks / 52
    seasonal = (
        1.0
        + 0.15 * np.sin(angle)
        + 0.10 * np.cos(angle)
        + 0.07 * np.sin(2 * angle)
        + 0.04 * np.cos(2 * angle)
    )
    return seasonal


# ── Geração principal ────────────────────────────────────────────────────────

def generate_data() -> pd.DataFrame:
    weeks      = np.arange(N_WEEKS)
    week_dates = [START_DATE + pd.Timedelta(weeks=w) for w in weeks]

    # Tendência linear leve (crescimento orgânico da plataforma)
    trend = 1.0 + 0.003 * weeks                       # +0,3% a.s. orgânico

    # Sazonalidade
    seasonal = seasonality_index(weeks)

    # Flag de reajuste de preço (semana 26 ≈ jul/2023 e semana 78 ≈ jul/2024)
    price_hike = np.zeros(N_WEEKS)
    price_hike[26] = 1
    price_hike[78] = 1

    # Baseline orgânico (assinantes sem mídia paga)
    baseline = 5_000 * trend * seasonal

    # Impacto negativo pontual do reajuste de preço
    baseline -= 800 * price_hike

    data = {"week": week_dates}
    channel_contributions = []

    for ch_name, params in CHANNELS.items():
        # Spend semanal (truncado em zero para evitar valores negativos)
        spend_raw = rng.normal(params["spend_mean"], params["spend_std"], N_WEEKS)
        spend_raw = np.maximum(spend_raw, 0)

        # Adstock
        ads = geometric_adstock(spend_raw, params["lambda_decay"])

        # Saturação de Hill
        sat = hill_saturation(ads, params["gamma"], params["kappa"])

        # Contribuição em assinantes (escala pelo coeficiente verdadeiro)
        contrib = params["coef_true"] * sat

        col_spend  = f"spend_{ch_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        col_contrib = f"contrib_{ch_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"

        data[col_spend]  = spend_raw
        data[col_contrib] = contrib
        channel_contributions.append(contrib)

    # Assinantes totais (soma das contribuições + baseline + ruído)
    total_signal = baseline + sum(channel_contributions)
    noise        = rng.normal(0, 500, N_WEEKS)
    new_subscribers = np.maximum(np.round(total_signal + noise), 0).astype(int)

    data["seasonality_index"] = seasonal
    data["trend"]             = trend
    data["price_hike_flag"]   = price_hike.astype(int)
    data["new_subscribers"]   = new_subscribers

    return pd.DataFrame(data)


# ── Ponto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_path = Path(__file__).parent / "MMM_Synth_Weekly_Data.csv"
    df = generate_data()
    df.to_csv(out_path, index=False)

    print(f"✅  Base sintética gerada: {out_path}")
    print(f"   {len(df)} semanas · {df.shape[1]} colunas")
    print(f"   Período: {df['week'].min().date()} → {df['week'].max().date()}")
    print(f"\n   Assinantes/semana — média: {df['new_subscribers'].mean():,.0f} "
          f"| min: {df['new_subscribers'].min():,} | max: {df['new_subscribers'].max():,}")
    total_spend = sum(
        df[c].sum()
        for c in df.columns if c.startswith("spend_")
    )
    print(f"   Spend total (2 anos): R$ {total_spend/1e6:.2f} M")
