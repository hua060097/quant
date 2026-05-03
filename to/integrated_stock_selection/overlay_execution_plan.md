# Top1 Execution Overlay Plan

## Goal

Build a production-safe execution overlay on top of the existing `Top1` rank strategy in `main_alpha原始_overlay.py`.

This overlay is **not** a `Top1 -> Top2` switcher and **not** a newly trained model on the current Window 3 test segment.
It is a conservative execution-layer controller for:

- `full`
- `half`
- `skip`

on the daily selected `Top1`.

## Why This Direction

Recent diagnostics showed:

- `Top1` still beats `Top2` on real minute execution, even though `Top2` can look better in label-side JSON.
- `Top2` improves after stricter downside open-gap filtering, but it is still not a stable replacement.
- `daily_rank_stats` is a label-side view, not the final execution target.
- Using the same Window 3 test segment to train a new switch model would be too close to post-hoc fitting.

Therefore the professional action is:

1. keep `Top1` as the main alpha carrier
2. add a lightweight execution overlay
3. reduce exposure when `Top1` is weak / ambiguous
4. avoid promoting `Top2` into a fallback trading target

## Overlay Design

### Scope

Overlay applies only to `Top1` strategy (`stock_count == 1`).

Other rank strategies remain unchanged.

### Inputs

Use only same-day observable quantities already available in prediction output:

- `predicted_return`
- `std_pred`
- `high_confidence`
- optional `p_cls`
- cross-sectional relation between `Top1` and `Top2`

No new training is introduced in this version.

### Candidate Universe

Overlay diagnostics are computed on the same tradable universe style used by the minute strategy pre-filter:

- `is_can_buy == 1`
- exclude BJ
- exclude ST
- reasonable close price
- liquidity threshold
- exclude limit-up / limit-down / zhaban

This keeps overlay diagnostics aligned with actual minute execution selection.

### Core Signals

For each trading date:

- `overlay_margin12 = pred(top1) - pred(top2)`
- `overlay_margin12_rel = margin12 / abs(pred(top1))`
- `overlay_std_certainty_pct`
  - cross-sectional percentile where lower `std_pred` means higher certainty
- `overlay_high_confidence`
- optional `p_cls(top1)`

### Decision Logic

The rule is intentionally conservative:

- `skip`
  - when dominance over `Top2` is weak **and** confidence is weak
- `half`
  - when one side is weak but not both
- `full`
  - when `Top1` clearly dominates and uncertainty is acceptable

This is designed as exposure control, not signal replacement.

## Code Modification Plan

### In `main_alpha原始_overlay.py`

1. Add overlay config constants.
2. Add helper to compute tradable-universe diagnostics by date.
3. Annotate prediction dataframe with:
   - `overlay_action`
   - `overlay_scale`
   - `overlay_reason`
   - `overlay_margin12`
   - `overlay_margin12_rel`
   - `overlay_std_certainty_pct`
   - `rank_pos_all`
   - `rank_pct_all`
4. Call the annotation helper before backtest.
5. Add concise overlay summary logs.

### In `rqalpha_minute_datasource/strategy_backtest_lead_fusion_rank_binary_v2.py`

1. Read overlay fields from selected `Top1` row.
2. If action is `skip`, record a skipped buy attempt and do not buy.
3. If action is `half`, multiply existing `position_pct`.
4. Preserve original open-gap risk control and sell logic.
5. Add overlay stats to summary logs.

## Validation Standard

This version is validated by:

- syntax compilation
- deterministic logs
- preserved original strategy path for non-Top1 strategies

This version is **not** claimed as a new trained alpha model.
It is a safer execution overlay implementation for further out-of-sample evaluation.
