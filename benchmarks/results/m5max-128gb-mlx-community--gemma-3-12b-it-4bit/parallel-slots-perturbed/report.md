# jevmlx eval report

## Environment

| key | value |
| --- | --- |
| chip | Apple M5 Max |
| git_sha | dbb1ff1f04a0bce4b20817e4c09eae2b5a3d2ad3 |
| jevmlx_version | 0.1.0 |
| machine_model | Mac17,7 |
| macos_version | 26.6.2 |
| mlx_lm_version | 0.31.3 |
| mlx_version | 0.32.2 |
| python_version | 3.12.14 |
| ram_gb | 128.0000 |
| timestamp_utc | 2026-09-21T23:17:16+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.5475 [0.5064, 0.5891] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.0000 |
| case_exact_match | 0.0000 |
| brier | 0.8526 [0.8165, 0.8890] (case_cluster_bootstrap) |
| correctness_auroc | 0.6670 |
| ece_5bin_equal_mass | 0.3216 |
| tie_rate | 0.0174 |
| any_flip_rate[action] | 0.3833 |
| any_flip_rate[category] | 0.1333 |
| any_flip_rate[fraud] | 0.0208 |
| any_flip_rate[needs_human] | 0.0000 |
| any_flip_rate[priority] | 0.4208 |
| any_flip_rate[risk] | 0.5708 |
| balanced_accuracy[action] | 0.5475 |
| balanced_accuracy[category] | 0.7786 |
| balanced_accuracy[fraud] | 0.8773 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.3562 |
| balanced_accuracy[risk] | 0.2880 |
| macro_f1[action] | 0.4992 |
| macro_f1[category] | 0.7341 |
| macro_f1[fraud] | 0.7956 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.2895 |
| macro_f1[risk] | 0.2662 |
| mean_tvd[action] | 0.3616 |
| mean_tvd[category] | 0.1334 |
| mean_tvd[fraud] | 0.0455 |
| mean_tvd[needs_human] | 0.0130 |
| mean_tvd[priority] | 0.4022 |
| mean_tvd[risk] | 0.4288 |
| order_flip_rate[action] | 0.3833 |
| order_flip_rate[category] | 0.1333 |
| order_flip_rate[fraud] | 0.0208 |
| order_flip_rate[needs_human] | 0.0000 |
| order_flip_rate[priority] | 0.4208 |
| order_flip_rate[risk] | 0.5708 |
| perturbation_flip_rate | 0.1898 |
| valid_accuracy | 0.5475 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| priority | 48 | 0.2500 [0.1492, 0.3878] (wilson) † | 0.5833 |
| risk | 48 | 0.3333 [0.2168, 0.4746] (wilson) † | 0.4167 |
| action | 48 | 0.4792 [0.3447, 0.6167] (wilson) | 0.4167 |
| needs_human | 48 | 0.5833 [0.4428, 0.7115] (wilson) | 0.5833 |
| category | 48 | 0.7500 [0.6122, 0.8508] (wilson) | 0.3333 |
| fraud | 48 | 0.8125 [0.6806, 0.8981] (wilson) | 0.7500 |
