## Challenger Plan

### Objectives
- Keep the legacy champion workflow untouched outside `main_alpha原始_overlay.py`.
- Use `main_alpha原始_overlay.py` as a dedicated challenger entry.
- Replace the old expanding walk-forward split with a shorter rolling split.
- Make frozen clipping windows configurable at factor-generation time instead of hardcoded in factor files.
- Add a new regime-oriented factor file without changing old factor files' public names.

### Time Split
- Use rolling windows with:
  - training about 13-15 months
  - validation 3 months
  - test at least 3 months
- The new `Window 3` is designed to stay close to the current regime and still include a full recent spring sample.

### Frozen Clip
- `run_generate_factors_mode` passes:
  - `frozen_clip_train_start`
  - `frozen_clip_train_end`
  through `context_extra`.
- Active frozen factor definitions read these values from `ctx.extra`.
- If not provided, they fall back to the legacy default window so old workflows remain reproducible.

### New Regime Factors
- Add a new factor file with regime/gating style factors using only fields already used elsewhere:
  - market data: `close`, `daily_change_pct`, `amount` / `volume`
  - stock data: `pct_chg`, `is_limit_up`, `is_limit_down`, `is_zhaban`, `stock_code`, `date`
- Register new factors in:
  - `_get_stock_factors`
  - `_get_original_factors`
  - `_get_zscore_factors` if needed

### Champion vs Challenger
- Do not implement both models in one training loop.
- Champion stays in the legacy file / legacy factors.
- Challenger is trained and tested from `main_alpha原始_overlay.py`.
- Comparison is done externally by comparing the same OOS dates.

### Overlay
- The previous top1 overlay rule is not a valid challenger design.
- Disable it by default so challenger walk-forward is not polluted by a degenerate half-position rule.
