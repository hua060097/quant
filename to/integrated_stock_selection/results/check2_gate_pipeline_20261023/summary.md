# Check 2: 13 个 July 极端门控 + 2 个对照门控

## 1. pipeline_all G0/G2/G3 / 淘汰原因

| gate | in pipeline | g0 | neutral | dir_corr | median_b | same_frac | folds | ci_lo | G2 | G3 | g3_p | score | bank-v2 | 淘汰原因 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GM_newhigh_ratio_level | 1 | pass | False | 0.8258 | 0.00921 | 0.855 | 3 | 0.00326 | False | True | 0.0198 | 0.524 | 0 | G2 fail; G3 pass; not bank-v2 (G4/max-market cap) |
| GM_nh60_prem_chg | 0 | — | — | nan | 0.00608 | 0.790 |  | nan | — | True | nan | 0.885 | 1 | not in local 40-gate pipeline_all; server bank-v2 entry available |
| GM_sec_disp_ratio | 1 | pass | True | 0.1543 | 0.00118 | 0.532 | 3 | -0.00520 | False | False | 0.7723 | 0.690 | 0 | G2 fail; G3 fail |
| GM_vol_med_ratio | 1 | pass | True | 0.1966 | -0.00115 | 0.548 | 2 | -0.00905 | False | False | 0.7822 | 0.677 | 0 | G2 fail; G3 fail |
| GM_amp_med_ratio | 1 | pass | True | 0.2556 | -0.00325 | 0.589 | 3 | -0.01032 | False | False | 0.3663 | 0.670 | 0 | G2 fail; G3 fail |
| GM_xs_disp_ratio | 1 | pass | True | 0.0564 | 0.00620 | 0.645 | 3 | -0.00101 | False | True | 0.1782 | 0.801 | 0 | G2 fail; G3 pass; not bank-v2 (G4/max-market cap) |
| GM_xs_iqr_ratio | 1 | pass | True | 0.1239 | 0.00008 | 0.500 | 1 | -0.00749 | False | False | 0.9901 | 0.688 | 0 | G2 fail; G3 fail |
| GM_sec_disp_level | 1 | pass | True | 0.0952 | 0.00074 | 0.524 | 3 | -0.00654 | False | False | 0.8119 | 0.715 | 0 | G2 fail; G3 fail |
| GM_xs_disp_level | 1 | pass | True | 0.2052 | 0.00434 | 0.613 | 3 | -0.00296 | False | False | 0.2772 | 0.708 | 0 | G2 fail; G3 fail |
| GM_amp_med_level | 1 | pass | True | 0.2398 | -0.00662 | 0.734 | 3 | -0.01350 | False | False | 0.1584 | 0.754 | 0 | G2 fail; G3 fail |
| GM_xs_iqr_level | 1 | pass | True | 0.0962 | -0.00162 | 0.540 | 2 | -0.00871 | False | False | 0.7030 | 0.724 | 0 | G2 fail; G3 fail |
| GM_amt_hhi_level | 1 | pass | True | 0.0965 | 0.00271 | 0.565 | 2 | -0.00452 | False | False | 0.5446 | 0.737 | 0 | G2 fail; G3 fail |
| GM_vol_med_level | 1 | pass | True | 0.1866 | -0.00223 | 0.677 | 1 | -0.00983 | False | False | 0.5743 | 0.748 | 0 | G2 fail; G3 fail |
| GM_newhigh_ratio_chg | 1 | pass | False | 0.6053 | 0.01096 | 0.839 | 4 | 0.00408 | True | True | 0.0099 | 0.628 | 1 | passed to bank-v2 |
| GM_sec_top5_persist_level | 0 | — | — | nan | 0.01422 | 0.952 |  | nan | — | True | nan | 0.801 | 1 | not in local 40-gate pipeline_all; server bank-v2 entry available |

## 2. 月均 p：2024-01 / 2024-02 / 2026-01..09

| gate | 2024-01 | 2024-02 | 2026-01 | 2026-02 | 2026-03 | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 | H1 mean | July | Aug-Sep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GM_newhigh_ratio_level | 0.177 | 0.024 | 0.669 | 0.738 | 0.520 | 0.107 | 0.607 | 0.250 | 0.081 | 0.060 | 0.355 | 0.482 | 0.081 | 0.208 |
| GM_nh60_prem_chg | 0.709 | 0.356 | 0.588 | 0.427 | 0.600 | 0.498 | 0.826 | 0.506 | 0.129 | 0.508 | 0.719 | 0.574 | 0.129 | 0.614 |
| GM_sec_disp_ratio | 0.676 | 0.956 | 0.860 | 0.974 | 0.841 | 0.261 | 0.206 | 0.688 | 0.878 | 0.517 | 0.141 | 0.638 | 0.878 | 0.329 |
| GM_vol_med_ratio | 0.601 | 0.997 | 0.618 | 0.897 | 0.889 | 0.951 | 0.700 | 0.801 | 0.913 | 0.679 | 0.048 | 0.809 | 0.913 | 0.363 |
| GM_amp_med_ratio | 0.898 | 0.996 | 0.601 | 0.777 | 0.771 | 0.782 | 0.675 | 0.985 | 0.942 | 0.328 | 0.010 | 0.765 | 0.942 | 0.169 |
| GM_xs_disp_ratio | 0.270 | 0.785 | 0.921 | 0.919 | 0.674 | 0.492 | 0.908 | 0.987 | 0.968 | 0.388 | 0.008 | 0.817 | 0.968 | 0.198 |
| GM_xs_iqr_ratio | 0.707 | 0.993 | 0.718 | 0.870 | 0.644 | 0.508 | 0.821 | 0.991 | 0.976 | 0.389 | 0.011 | 0.758 | 0.976 | 0.200 |
| GM_sec_disp_level | 0.452 | 0.904 | 0.777 | 0.980 | 0.888 | 0.590 | 0.526 | 0.866 | 0.976 | 0.771 | 0.555 | 0.771 | 0.976 | 0.663 |
| GM_xs_disp_level | 0.423 | 0.810 | 0.624 | 0.723 | 0.606 | 0.527 | 0.928 | 0.999 | 0.981 | 0.769 | 0.431 | 0.734 | 0.981 | 0.600 |
| GM_amp_med_level | 0.768 | 0.998 | 0.460 | 0.769 | 0.755 | 0.793 | 0.801 | 0.998 | 0.991 | 0.775 | 0.553 | 0.763 | 0.991 | 0.664 |
| GM_xs_iqr_level | 0.385 | 0.978 | 0.636 | 0.898 | 0.813 | 0.708 | 0.947 | 0.997 | 0.991 | 0.816 | 0.576 | 0.833 | 0.991 | 0.696 |
| GM_amt_hhi_level | 0.024 | 0.103 | 0.654 | 0.582 | 0.566 | 0.754 | 0.925 | 1.000 | 0.996 | 0.920 | 0.682 | 0.747 | 0.996 | 0.801 |
| GM_vol_med_level | 0.282 | 0.989 | 0.314 | 0.763 | 0.844 | 0.896 | 0.821 | 0.921 | 0.998 | 0.927 | 0.621 | 0.760 | 0.998 | 0.774 |
| GM_newhigh_ratio_chg | 0.396 | 0.552 | 0.853 | 0.269 | 0.288 | 0.544 | 0.654 | 0.339 | 0.344 | 0.689 | 0.697 | 0.491 | 0.344 | 0.693 |
| GM_sec_top5_persist_level | 0.477 | 0.460 | 0.614 | 0.651 | 0.091 | 0.100 | 0.700 | 0.496 | 0.330 | 0.328 | 0.398 | 0.442 | 0.330 | 0.363 |

## 3. 读数

- **2026 H1 误报**：高波动/振幅/离散度/集中度这一组在 2026-01~06 已经长期高位，不是 7 月才出现；7 月只是继续/加剧。若把它们当作“7 月退潮信号”，H1 会大量误报。
- `GM_newhigh_ratio_level` 不同：H1 均值明显高于 7 月，7 月才是低分位；`GM_nh60_prem_chg` 也在 7 月最低（H1 内有 5 月 .826 的高位）。
- **2024-01 vs 2026-07**：2024-01 并没有普遍极端；高波动/离散度门控大多在 .4~.9。**2024-02** 才出现与 2026-07 类似的高位簇（vol/amp/xs_iqr/amt_hhi 近 1）。
- 因此这组高波动门控更像“波动/扩散状态”，会在 2024-02 和 2026-07 同时亮起；它们不是 7 月特有的退潮证据。真正与 7 月绑定的，是 `GM_newhigh_ratio_level` 与 `GM_nh60_prem_chg` 的低位。
- G2/G3：11 个高波动/离散度门控除 `GM_xs_disp_ratio` 外均 G2 fail 且 G3 fail；`GM_xs_disp_ratio` G2 fail 但 G3 pass。`GM_newhigh_ratio_level` G2 fail、G3 pass，但 bank-v2 未收录（G4/10 名额截断）。
- `GM_nh60_prem_chg` / `GM_sec_top5_persist_level` 不在本地 40-gate pipeline_all；server bank-v2 中 `GM_nh60_prem_chg` score .8845、`GM_sec_top5_persist_level` score .8012。
