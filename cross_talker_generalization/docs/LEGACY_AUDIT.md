# Historical implementation audit

This document records consequential differences among the earlier paper, manuscript text, and notebook implementation. Its purpose is not to invalidate historical results, but to prevent different estimands from being conflated.

| Topic | Published or claimed | Historical implementation | Production handling |
|---|---|---|---|
| DTW normalization | Optimal path length | Mean of the two sequence lengths | Explicit profile; mean-length reproduction plus path-length sensitivity |
| Multi-talker aggregation | Maximum similarity in the earlier best model | Several notebooks average raw distance | Explicit `mean_distance` and `min_distance` profiles |
| `k` selection | Optimized against behavior | Training-fold Wald z was optimized, then fold-specific `k` values were averaged | Not selected in confirmatory analysis; fixed grid only in compatibility profile |
| Cross-validation | Held-out evaluation | GLMM was refit on held-out rows and its z reported | Primary analysis freezes the training fit and scores held-out log loss |
| Fold reproducibility | Three folds | Some `StratifiedKFold` calls lacked `random_state` | Fixed seed and saved fold table |
| Similarity meaning | Information supplied by exposure | Often a same-content counterfactual recording rather than a heard token | Named `same_content_talker_proxy` |
| t-SNE scaling | 3-D representation | Some legacy paths z-scored each t-SNE axis | Primary profile leaves t-SNE geometry unchanged |
| B23 HVE | Actual exposure variability | Exact multi-talker assignment is absent | Actual multi-talker estimand is blocked; proxies require separate names |

Additional risks in the historical utilities included broad exception swallowing, dependence on filesystem traversal order, notebook-global state, unseeded folds, duplicated dataset-specific code, and R fits launched inside layer workers. The production implementation uses manifest IDs, deterministic task tables, one HDF5 open per layer worker, tidy outputs, and fail-closed validation.
