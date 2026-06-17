<!-- RESULTS.md -->
# Model Results

All metrics evaluated on the held-out **test set** (15% of data, never seen during training).  
Primary metric: **F1 Score** (precision-recall balance — accuracy is meaningless with 8:1 class imbalance).

---

## Baseline Models (Week 2)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Random Forest (default) | — | — | — | — | — |
| Random Forest (tuned)   | — | — | — | — | — |
| XGBoost (tuned)         | — | — | — | — | — |

*Run `notebooks/05_hyperparameter_tuning.ipynb` with real Kepler data (to be filled in)*

## Cross-Validation Results (5-fold Stratified)

| Model | F1 mean ± std | AUC mean ± std | Notes |
|-------|--------------|----------------|-------|
| Random Forest | — ± — | — ± — | 300 trees, balanced weights |
| XGBoost | — ± — | — ± — | scale_pos_weight=8.3 |

## Feature Importance Findings (Week 2)

Top features by permutation importance on validation set:
1. `folded_transit_depth` — strongest single signal
2. `n_dips_detected` — periodicity indicator  
3. `dip_depth_consistency` — separates planets (consistent) from binaries (variable)
4. `flux_skew` — transit dips create negative skew
5. `frac_below_3sigma` — proxy for transit frequency

Features with near-zero importance (candidates for removal):
- `flux_mean` — always ~1.0 after normalization
- `flux_max` — noise-dominated

---

## Deep Learning Models (Week 3)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| 1D-CNN | — | — | — | — | — |
| LSTM | — | — | — | — | — |
| Ensemble (best) | — | — | — | — | — |

---

## Cross-Validation Results (5-fold)

| Model | F1 mean ± std | AUC mean ± std |
|-------|--------------|----------------|
| Random Forest | — | — |
| XGBoost | — | — |

---

*Updated as the project progresses. See `mlruns/` for full experiment tracking history.*