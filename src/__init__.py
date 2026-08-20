"""
src — Módulos do pipeline MMM Video+
=====================================
Pacote com os módulos principais do projeto de Marketing Mix Modeling.

Módulos:
  - transformations : adstock geométrico e saturação de Hill
  - model           : wrapper OLS + HAC e decomposição de contribuições
  - validation      : split temporal, métricas e visualização de holdout
"""

from .transformations import (
    geometric_adstock,
    hill_saturation,
    transform_channel,
    transform_all_channels,
    random_search_params,
    saturation_curve_points,
    marginal_return,
)

from .model import MMMModel, DEFAULT_CHANNEL_PARAMS

from .validation import (
    temporal_split,
    compute_metrics,
    holdout_week_by_week,
    durbin_watson,
    plot_holdout,
    validation_report,
)

__all__ = [
    # transformations
    "geometric_adstock",
    "hill_saturation",
    "transform_channel",
    "transform_all_channels",
    "random_search_params",
    "saturation_curve_points",
    "marginal_return",
    # model
    "MMMModel",
    "DEFAULT_CHANNEL_PARAMS",
    # validation
    "temporal_split",
    "compute_metrics",
    "holdout_week_by_week",
    "durbin_watson",
    "plot_holdout",
    "validation_report",
]
