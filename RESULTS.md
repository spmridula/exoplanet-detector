<!-- RESULTS.md -->
# Model Results

All metrics evaluated on the held-out **test set** (15% of data, never seen during training).  
Primary metric: **F1 Score** (precision-recall balance — accuracy is meaningless with 8:1 class imbalance).

---

## Baseline Models (Week 2)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

*To Fill in after running `notebooks/02_baseline_models.ipynb` with real Kepler data*

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