# 📺 Video+ MMM — Media Mix Model para Plataforma de Streaming

> **Pipeline estatístico de Marketing Mix Modeling (MMM)** com decomposição de contribuição por canal, curvas de saturação, efeitos de carryover e validação out-of-sample — aplicado a uma base sintética inspirada em uma plataforma de streaming (SVOD) brasileira, a **Video+**.

---

## Visão geral

Este projeto documenta um pipeline de MMM de ponta a ponta aplicado a um negócio de assinatura de streaming de vídeo. O modelo quantifica a contribuição marginal de cada canal de mídia paga para a aquisição de novos assinantes, controla por baseline orgânico, sazonalidade e tendência, e reporta o CPA (custo por aquisição) implícito de cada canal — a métrica que efetivamente orienta decisões de realocação de budget.

**Por que isso importa para um negócio de streaming:** diferente de e-commerce, o crescimento de assinaturas em streaming é marcado por ciclos de conversão com atraso (adstock/carryover), forte sazonalidade (calendário esportivo, lançamentos de conteúdo, datas comerciais) e canais de performance com retornos decrescentes — padrões que atribuição linear simples não captura.

O repositório cobre duas frentes que conversam entre si:

1. **A planilha de modelagem** (`MMM_CLV_ROI_VIDEO_MODEL_vFinal.xlsx`), com os dados brutos, o modelo estatístico ajustado e sua validação.
2. **A apresentação executiva** (`MMM_Video+_Achados_2024.pptx`), que traduz os resultados do modelo em leitura de negócio e recomendações.

---

## Metodologia estatística

O modelo segue uma especificação de MMM linear com transformações não lineares por canal, no padrão adotado por frameworks como Meta Robyn e Google LightweightMMM:

```
Spend bruto por canal (semanal)
        │
        ▼
Adstock geométrico (carryover / efeito residual)
   adstock_t = spend_t + λ × adstock_(t-1)
   λ (decay) calibrado por canal — de 0,05 (Call Center, Paid Search)
   a 0,60 (YouTube Ads)
        │
        ▼
Saturação de Hill (retornos decrescentes)
   saturação(x) = x^γ / (x^γ + κ^γ)
   γ (formato da curva) e κ (ponto de meia-saturação)
   otimizados por canal via busca aleatória/grid search
        │
        ▼
Regressão linear (OLS) com erro-padrão robusto HAC (Newey-West, 4 lags)
   variáveis de controle: sazonalidade, tendência, flag de reajuste de preço
        │
        ├──► Coeficientes por canal + intervalo de confiança (95%) + p-valor
        ├──► Diagnóstico de multicolinearidade (VIF)
        │
        └──► Validação out-of-sample (holdout temporal)
                  treino: semanas 1–80 · holdout: semanas 81–104
                  parâmetros de adstock/Hill nunca veem o holdout
```

**Por que HAC/Newey-West e não OLS comum?** Séries temporais de spend e conversão carregam autocorrelação residual (Durbin–Watson ≈ 1,30 no ajuste final). Erro-padrão OLS comum subestima essa correlação e infla artificialmente a significância dos coeficientes. O erro-padrão HAC corrige isso, produzindo p-valores mais conservadores e confiáveis.

**Por que holdout temporal?** Um modelo pode ter R² alto simplesmente por ter graus de liberdade suficientes para "decorar" a série de treino. Reservar as últimas 24 semanas — nunca usadas na escolha de γ, κ ou λ — é o único jeito de checar se o modelo generaliza para dados fora da amostra.

**Diagnóstico de multicolinearidade (VIF):** todos os canais de mídia apresentam VIF entre 1,2 e 1,8 (baixa colinearidade); a variável de tendência (`trend`) tem VIF de 6,38 — ainda alto, sinalizado como limitação a corrigir na próxima iteração.

---

## Dados usados (`MMM_CLV_ROI_VIDEO_MODEL_vFinal.xlsx`)

A planilha tem 14 abas, divididas em dois blocos:

### Bloco 1 — Contexto de negócio (inputs de planejamento)
| Aba | Conteúdo |
|---|---|
| `Platform Traffic Distribution` | Distribuição de tráfego da plataforma por origem |
| `Acquisition Channels` | CPA-alvo, budget planejado, leads gerados e estratégia de lance por canal |
| `Customer Segments` | 8 segmentos de assinantes (ex.: Famílias Tradicionais, Maratonistas de Séries, Torcedores Fanáticos), com CLV, ARPU, canal prioritário e budget sugerido |
| `Subscription plans` | Planos comerciais (Telecine, Combate, HBO, Claro Shows, App Claro TV+ com Globoplay etc.), preço mensal, tempo médio de assinatura e CLV por plano |
| `CAC \| CLV \| ROI` | Cálculo de CAC, CLV e ROI consolidado |
| `Customer Attribution` | Atribuição de assinantes por canal e segmento |
| `Campaign Results` | Resultados da campanha *Monthly Campaign Video+*, com receita e ROI por segmento |
| `churn_frequency_subscribers` | Frequência de churn por coorte |
| `subs_evolution_over_time` | Evolução da base de assinantes ao longo do tempo (granularidade diária) |

### Bloco 2 — Modelo estatístico MMM v2
| Aba | Conteúdo |
|---|---|
| `MMM_Synth_Weekly_Data` | Base semanal de treino do modelo: **104 semanas, de 02/jan/2023 a 23/dez/2024** — spend por canal, assinantes adquiridos, índice de sazonalidade, flag de reajuste de preço |
| `MMM_v2_Model_Output_HAC` | Coeficientes finais, p-valor (HAC), IC 95% e VIF por variável |
| `MMM_v2_Validation_Holdout` | Previsão semana a semana no período de holdout, com erro absoluto (%) |
| `MMM_v2_Channel_Summary` | Parâmetros de adstock/Hill, spend total, assinantes atribuídos e CPA implícito por canal |
| `MMM_v2_Methodology_Notes` | Documentação da metodologia estatística, premissas e limitações conhecidas |

**Canais modelados:** Paid Search (Google), Social Media (Meta Ads), Social Media (YouTube Ads), Affiliate Marketing, Email Marketing, E-commerce Company Site e Call Center.

> ⚠️ **Nota de transparência:** a aba `MMM_v2_Methodology_Notes` declara explicitamente que os 104 pontos semanais são **100% sintéticos** — gerados por um processo conhecido (adstock + saturação + sazonalidade + ruído) calibrado nos CPAs-alvo já existentes na planilha de canais. Os dados **não refletem** tráfego, spend ou assinaturas reais. O valor do projeto está na metodologia estatística (como o modelo foi especificado e validado), não nos números de negócio em si.

---

## Principais achados (apresentação `MMM_Video+_Achados_2024.pptx`)

A apresentação traduz o modelo em leitura executiva para o período **jan/2023–dez/2024**, com 2024 como último ano completo da base. Os achados centrais:

### Qualidade de ajuste e validação
- **R² = 99,2%** no ajuste completo (99,1% ajustado); **Durbin–Watson = 1,30**, corrigido via erro-padrão HAC (4 lags).
- **Validação out-of-sample:** R² de 97,4% e **MAPE de 1,14%** nas 24 semanas de holdout (jul–dez/2024), nunca usadas para calibrar o modelo — RMSE de ±431 assinantes/semana.

### Significância por canal
- Os **7 canais pagos são estatisticamente significativos a 5%** (p < 0,05), com destaque para Affiliate Marketing e Call Center no maior coeficiente marginal — ressalvado que a escala de saturação (γ/κ) difere por canal, então coeficientes não são diretamente comparáveis em R$.
- VIF abaixo de 2 na maioria dos canais de mídia, indicando baixa multicolinearidade.

### CPA implícito (MMM) vs. CPA assumido no planejamento
| Canal | CPA implícito (MMM) | CPA assumido | Leitura |
|---|---|---|---|
| Paid Search (Google) | R$ 88,48 | R$ 45,00 | custa quase 2x o assumido |
| Social Media (Meta Ads) | R$ 57,05 | R$ 70,00 | mais eficiente que o assumido |
| Social Media (YouTube Ads) | R$ 53,96 | R$ 70,00 | mais eficiente que o assumido |
| Affiliate Marketing | R$ 21,58 | R$ 25,00 | eficiente |
| Email Marketing | R$ 18,71 | R$ 30,00 | um dos mais baratos |
| E-commerce Company Site | R$ 28,32 | R$ 70,00 | mais eficiente que o assumido |
| Call Center | R$ 18,71 | R$ 40,00 | um dos mais baratos |

### ROI de campanha por segmento (2024)
Famílias Tradicionais lidera com **ROI de 2,97x**, seguida por Fãs de Artes Marciais (1,70x), Maratonistas de Séries (1,64x) e Torcedores Fanáticos (1,32x). Já **Amantes de Música (-0,10x)** e **Amantes de Cinema (-0,41x)** apresentam ROI negativo — campanhas destruindo valor nesses segmentos.

### 2024 em números (vs. 2023)
- **1.476.809 novos assinantes** (+10,9% vs. 1.331.354 em 2023).
- **R$ 60,99 M** investidos em mídia (+7,6% vs. R$ 56,70 M em 2023).
- **CPA médio ponderado (blended) de R$ 41,30**, queda de 3,0% vs. R$ 42,59 em 2023.
- **R$ 10,27 M** em receita gerada pela campanha *Monthly Campaign Video+*, somando os 7 segmentos ativos.

### Recomendações estratégicas
1. Realocar verba para Affiliate Marketing, Call Center e Email Marketing — maior fronteira de eficiência disponível hoje.
2. Revisar o mix de Paid Search e E-commerce Site, cujo CPA real roda acima do CPA-alvo.
3. Priorizar budget em Famílias Tradicionais e Maratonistas de Séries — maior CLV e maior ROI de campanha.
4. Pausar ou reestruturar campanhas para Amantes de Cinema e Amantes de Música — ROI de campanha negativo.

---

## Limitações identificadas e roteiro (v3)

- Dado **real** de spend semanal por canal é o maior gap atual — tudo na base é sintético.
- Migrar de OLS frequentista para um framework Bayesiano hierárquico (Robyn / PyMC-Marketing) com priors informativos.
- Calibração externa via testes de incrementalidade (geo-experiments) para ancorar os coeficientes na realidade.
- Ortogonalizar a variável de tendência frente às demais variáveis de controle (VIF de trend ainda em 6,38).

---

## Estrutura do projeto

```
video-plus-mmm/
│
├── data/
│   ├── generate_synthetic_data.py       # Gerador da base sintética
│   └── MMM_CLV_ROI_VIDEO_MODEL_vFinal.xlsx  # Planilha de dados e modelo
│
├── notebooks/
│   ├── 01_eda.ipynb                     # Análise exploratória
│   ├── 02_adstock_saturation.ipynb      # Pipeline de transformação (adstock + Hill)
│   ├── 03_mmm_model_hac.ipynb           # Ajuste do modelo e erro-padrão HAC
│   └── 04_holdout_validation.ipynb      # Validação out-of-sample
│
├── src/
│   ├── transformations.py               # Funções de adstock geométrico e saturação de Hill
│   ├── model.py                         # Wrapper de regressão MMM (OLS + HAC)
│   └── validation.py                    # Split treino/holdout e métricas (R², MAPE, RMSE)
│
├── presentation/
│   └── MMM_Video+_Achados_2024.pptx     # Apresentação executiva dos achados
│
├── outputs/
│   ├── channel_significance.png
│   ├── cpa_implied_vs_assumed.png
│   └── campaign_roi_by_segment.png
│
├── requirements.txt
└── README.md
```

---

## Stack técnica

| Camada | Ferramentas |
|---|---|
| Geração de dados | Python, NumPy, pandas |
| Modelagem | statsmodels (OLS + HAC), scikit-learn |
| Validação | Split temporal treino/holdout, métricas R² / MAPE / RMSE |
| Visualização | matplotlib, seaborn, python-pptx |
| Planilha de apoio | openpyxl |

---

## Como rodar

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/video-plus-mmm.git
cd video-plus-mmm

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Gerar a base sintética semanal
python data/generate_synthetic_data.py

# 4. Rodar os notebooks em ordem
jupyter notebook notebooks/
```

---

## Contexto

Este projeto faz parte de um **portfólio de Marketing Science**, construído para demonstrar a capacidade de conduzir um MMM de ponta a ponta — da especificação estatística (adstock, saturação de Hill, erro-padrão robusto, validação out-of-sample) até a tradução dos resultados em recomendações de alocação de budget para um negócio de assinatura de streaming.

---

## Autor

**Bruno Pamplona** — Marketing Science Lead
Foco em Growth Analytics, Marketing Mix Modeling e inferência causal para negócios de assinatura.

[LinkedIn]([https://www.linkedin.com/in/bruno-pamplona]) • [GitHub](https://github.com/brunopamplona)

---

*Dados sintéticos. Todos os números são ilustrativos e não representam a performance real de nenhuma empresa.*
