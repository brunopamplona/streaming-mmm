"""
model.py
========
Wrapper do modelo de regressão MMM com erro-padrão HAC (Newey-West).

Responsabilidades:
  - Montar a matriz de features (mídia transformada + controles)
  - Ajustar OLS com erro-padrão HAC via statsmodels
  - Calcular VIF (diagnóstico de multicolinearidade)
  - Decompor contribuições por canal
  - Calcular CPA implícito por canal
  - Exportar sumário formatado

Uso típico:
    from src.model import MMMModel
    mmm = MMMModel(channel_params=CHANNEL_PARAMS, hac_lags=4)
    mmm.fit(df_train)
    summary = mmm.summary()
    contributions = mmm.decompose()
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ── Parâmetros canônicos de adstock e Hill para o modelo Video+ ──────────────

DEFAULT_CHANNEL_PARAMS: Dict[str, Dict] = {
    "Paid Search (Google)": {
        "lambda_decay": 0.05,
        "gamma": 0.70,
        "kappa": 150_000,
        "cpa_target": 45.0,
    },
    "Social Media (Meta Ads)": {
        "lambda_decay": 0.30,
        "gamma": 0.80,
        "kappa": 200_000,
        "cpa_target": 70.0,
    },
    "Social Media (YouTube Ads)": {
        "lambda_decay": 0.60,
        "gamma": 0.75,
        "kappa": 180_000,
        "cpa_target": 70.0,
    },
    "Affiliate Marketing": {
        "lambda_decay": 0.10,
        "gamma": 0.85,
        "kappa": 80_000,
        "cpa_target": 25.0,
    },
    "Email Marketing": {
        "lambda_decay": 0.05,
        "gamma": 0.90,
        "kappa": 30_000,
        "cpa_target": 30.0,
    },
    "E-commerce Company Site": {
        "lambda_decay": 0.15,
        "gamma": 0.80,
        "kappa": 100_000,
        "cpa_target": 70.0,
    },
    "Call Center": {
        "lambda_decay": 0.05,
        "gamma": 0.92,
        "kappa": 40_000,
        "cpa_target": 40.0,
    },
}


def _col_key(name: str) -> str:
    """Normaliza o nome do canal para nome de coluna."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


class MMMModel:
    """
    Marketing Mix Model com OLS + erro-padrão HAC (Newey-West).

    Parâmetros
    ----------
    channel_params : dict, opcional
        Dicionário com parâmetros de adstock/Hill por canal.
        Padrão: DEFAULT_CHANNEL_PARAMS.
    hac_lags : int
        Número de lags do corretor HAC (Newey-West). Padrão: 4.
    intercept : bool
        Se True, inclui intercepto na regressão. Padrão: True.
    """

    def __init__(
        self,
        channel_params: Optional[Dict] = None,
        hac_lags: int = 4,
        intercept: bool = True,
    ) -> None:
        self.channel_params = channel_params or DEFAULT_CHANNEL_PARAMS
        self.hac_lags       = hac_lags
        self.intercept      = intercept

        # Atributos preenchidos após .fit()
        self.result_         = None   # OLSResultsWrapper do statsmodels
        self.X_              = None   # DataFrame de features (treino)
        self.y_              = None   # Série dependente (treino)
        self.feature_cols_   = []     # Nomes das colunas de feature
        self.channel_cols_   = []     # Apenas colunas de mídia (sem controles)
        self._is_fitted      = False

    # ──────────────────────────────────────────────────────────────
    #  Transformações internas
    # ──────────────────────────────────────────────────────────────

    def _adstock(self, spend: np.ndarray, lam: float) -> np.ndarray:
        ads = np.empty_like(spend, dtype=float)
        ads[0] = float(spend[0])
        for t in range(1, len(spend)):
            ads[t] = float(spend[t]) + lam * ads[t - 1]
        return ads

    def _hill(self, x: np.ndarray, gamma: float, kappa: float) -> np.ndarray:
        x_safe = np.maximum(x, 0.0)
        x_g    = np.power(x_safe, gamma)
        k_g    = np.power(kappa, gamma)
        return x_g / (x_g + k_g)

    def _transform(self, spend: np.ndarray, params: dict) -> np.ndarray:
        ads = self._adstock(spend, params["lambda_decay"])
        return self._hill(ads, params["gamma"], params["kappa"])

    # ──────────────────────────────────────────────────────────────
    #  Construção da matriz de features
    # ──────────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Monta o DataFrame de features a partir dos dados brutos.

        Espera no `df`:
          - spend_<col_key>         → gasto bruto por canal
          - seasonality_index       → índice de sazonalidade
          - trend                   → tendência linear
          - price_hike_flag         → flag de reajuste de preço
        """
        features: Dict[str, np.ndarray] = {}

        # Canais de mídia (adstock → Hill)
        for ch_name, params in self.channel_params.items():
            key       = _col_key(ch_name)
            spend_col = f"spend_{key}"
            if spend_col not in df.columns:
                raise KeyError(
                    f"Coluna '{spend_col}' não encontrada. "
                    f"Colunas disponíveis: {list(df.columns)}"
                )
            features[f"media_{key}"] = self._transform(df[spend_col].values, params)

        self.channel_cols_ = [c for c in features]

        # Variáveis de controle
        for ctrl in ["seasonality_index", "trend", "price_hike_flag"]:
            if ctrl in df.columns:
                features[ctrl] = df[ctrl].values
            else:
                warnings.warn(f"Coluna de controle '{ctrl}' não encontrada. Ignorando.")

        X = pd.DataFrame(features, index=df.index)

        if self.intercept:
            X = sm.add_constant(X, has_constant="add")

        self.feature_cols_ = list(X.columns)
        return X

    # ──────────────────────────────────────────────────────────────
    #  Ajuste do modelo
    # ──────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, target_col: str = "new_subscribers") -> "MMMModel":
        """
        Ajusta o modelo OLS com erro-padrão HAC.

        Parâmetros
        ----------
        df : pd.DataFrame
            Dados de treino com colunas de spend, controles e target.
        target_col : str
            Nome da coluna dependente. Padrão: 'new_subscribers'.

        Retorna
        -------
        self
        """
        if target_col not in df.columns:
            raise KeyError(f"Coluna alvo '{target_col}' não encontrada.")

        self.y_ = df[target_col].values.astype(float)
        self.X_ = self._build_features(df)

        ols   = sm.OLS(self.y_, self.X_)
        self.result_ = ols.fit(
            cov_type="HAC",
            cov_kwds={"maxlags": self.hac_lags},
        )
        self._is_fitted = True
        return self

    # ──────────────────────────────────────────────────────────────
    #  Previsão
    # ──────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Retorna previsão de novos assinantes para `df`."""
        self._check_fitted()
        X = self._build_features(df)
        return self.result_.predict(X)

    # ──────────────────────────────────────────────────────────────
    #  VIF — diagnóstico de multicolinearidade
    # ──────────────────────────────────────────────────────────────

    def vif(self) -> pd.DataFrame:
        """
        Calcula o Variance Inflation Factor (VIF) para cada feature.

        VIF < 5  → baixa multicolinearidade (aceitável)
        VIF 5–10 → moderada (monitorar)
        VIF > 10 → alta (problemática)
        """
        self._check_fitted()
        X_arr = self.X_.values
        cols  = self.X_.columns.tolist()

        vif_values = [
            variance_inflation_factor(X_arr, i)
            for i in range(X_arr.shape[1])
        ]
        return pd.DataFrame({"feature": cols, "VIF": vif_values}).sort_values("VIF", ascending=False)

    # ──────────────────────────────────────────────────────────────
    #  Decomposição de contribuições
    # ──────────────────────────────────────────────────────────────

    def decompose(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Decompõe a série ajustada em contribuições por canal e baseline.

        Contribuição do canal i na semana t:
            contrib_it = coef_i × X_it

        Baseline = constante + controles (sazonalidade, tendência, reajuste).

        Parâmetros
        ----------
        df : pd.DataFrame, opcional
            Se None, usa os dados de treino.

        Retorna
        -------
        pd.DataFrame
            DataFrame com colunas:
              - contrib_<canal>  → contribuição do canal em assinantes
              - baseline         → baseline orgânico + controles
              - fitted           → soma (deve igualar predict())
        """
        self._check_fitted()
        X = self._build_features(df) if df is not None else self.X_
        coef = self.result_.params

        result: Dict[str, np.ndarray] = {}
        baseline = np.zeros(len(X))

        for col in X.columns:
            contribution = coef[col] * X[col].values
            if col in self.channel_cols_:
                ch_label = col.replace("media_", "contrib_")
                result[ch_label] = contribution
            else:
                baseline += contribution

        result["baseline"] = baseline
        result["fitted"]   = sum(result.values())

        return pd.DataFrame(result, index=X.index)

    # ──────────────────────────────────────────────────────────────
    #  CPA implícito
    # ──────────────────────────────────────────────────────────────

    def cpa_implied(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o CPA implícito do MMM por canal:

            CPA_implícito = spend_total / assinantes_atribuídos

        Parâmetros
        ----------
        df : pd.DataFrame
            DataFrame com dados de spend (mesmo período do treino).

        Retorna
        -------
        pd.DataFrame
            Tabela com: canal, spend_total, subs_atribuídos, CPA_implícito,
            CPA_target e razão implícito/target.
        """
        self._check_fitted()
        decomp = self.decompose(df)
        rows: List[dict] = []

        for ch_name, params in self.channel_params.items():
            key       = _col_key(ch_name)
            spend_col = f"spend_{key}"
            contrib_col = f"contrib_{key}"

            if spend_col not in df.columns or contrib_col not in decomp.columns:
                continue

            spend_total = df[spend_col].sum()
            subs_attr   = decomp[contrib_col].sum()
            cpa_imp     = spend_total / subs_attr if subs_attr > 0 else np.nan
            cpa_target  = params.get("cpa_target", np.nan)

            rows.append({
                "canal":               ch_name,
                "spend_total_R$":      round(spend_total, 2),
                "subs_atribuidos":     round(subs_attr, 0),
                "CPA_implicito_R$":    round(cpa_imp, 2),
                "CPA_target_R$":       cpa_target,
                "ratio_imp_vs_target": round(cpa_imp / cpa_target, 2) if cpa_target else np.nan,
            })

        return pd.DataFrame(rows).sort_values("CPA_implicito_R$")

    # ──────────────────────────────────────────────────────────────
    #  Sumário
    # ──────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Retorna o sumário completo do modelo statsmodels."""
        self._check_fitted()
        return str(self.result_.summary())

    def coef_table(self) -> pd.DataFrame:
        """
        Tabela de coeficientes com IC 95% e p-valor (HAC).

        Retorna
        -------
        pd.DataFrame
            feature | coef | std_err | t_stat | p_value | ci_lower | ci_upper
        """
        self._check_fitted()
        res = self.result_
        ci  = res.conf_int(alpha=0.05)

        return pd.DataFrame({
            "feature": res.params.index,
            "coef":    res.params.values,
            "std_err": res.bse.values,
            "t_stat":  res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_lower": ci.iloc[:, 0].values,
            "ci_upper": ci.iloc[:, 1].values,
        })

    # ──────────────────────────────────────────────────────────────
    #  Métricas de ajuste
    # ──────────────────────────────────────────────────────────────

    def metrics(self, df: Optional[pd.DataFrame] = None, target_col: str = "new_subscribers") -> dict:
        """
        Calcula R², R² ajustado, MAPE e RMSE.

        Se `df` for fornecido, calcula as métricas out-of-sample.
        """
        self._check_fitted()

        if df is not None:
            y_true = df[target_col].values.astype(float)
            y_pred = self.predict(df)
            n      = len(y_true)
            k      = len(self.feature_cols_)
        else:
            y_true = self.y_
            y_pred = self.result_.fittedvalues
            n      = len(y_true)
            k      = len(self.feature_cols_)

        ss_res  = np.sum((y_true - y_pred) ** 2)
        ss_tot  = np.sum((y_true - y_true.mean()) ** 2)
        r2      = 1 - ss_res / ss_tot
        r2_adj  = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        mape    = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        rmse    = np.sqrt(np.mean((y_true - y_pred) ** 2))

        return {"R2": round(r2, 4), "R2_adj": round(r2_adj, 4),
                "MAPE_%": round(mape, 4), "RMSE": round(rmse, 2)}

    # ──────────────────────────────────────────────────────────────
    #  Utilitário interno
    # ──────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Modelo não ajustado. Execute .fit() primeiro.")
