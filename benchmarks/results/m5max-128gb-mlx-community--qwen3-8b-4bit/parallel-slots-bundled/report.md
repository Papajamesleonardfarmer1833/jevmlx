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
| timestamp_utc | 2026-09-21T06:43:34+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.5417 [0.4560, 0.6250] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.0000 |
| case_exact_match | 0.0000 |
| brier | 0.9415 [0.8274, 1.0599] (case_cluster_bootstrap) |
| correctness_auroc | 0.7190 |
| ece_5bin_equal_mass | 0.3714 |
| tie_rate | 0.0093 |
| any_flip_rate[action] | 0.3833 |
| any_flip_rate[category] | 0.3833 |
| any_flip_rate[fraud] | 0.0167 |
| any_flip_rate[needs_human] | 0.0167 |
| any_flip_rate[priority] | 0.3667 |
| any_flip_rate[risk] | 0.5167 |
| balanced_accuracy[action] | 0.4074 |
| balanced_accuracy[category] | 0.3889 |
| balanced_accuracy[fraud] | 0.8056 |
| balanced_accuracy[needs_human] | 0.6548 |
| balanced_accuracy[priority] | 0.4537 |
| balanced_accuracy[risk] | 0.2979 |
| macro_f1[action] | 0.3915 |
| macro_f1[category] | 0.3618 |
| macro_f1[fraud] | 0.8489 |
| macro_f1[needs_human] | 0.5734 |
| macro_f1[priority] | 0.4573 |
| macro_f1[risk] | 0.2745 |
| mean_tvd[action] | 0.3658 |
| mean_tvd[category] | 0.3706 |
| mean_tvd[fraud] | 0.0131 |
| mean_tvd[needs_human] | 0.0190 |
| mean_tvd[priority] | 0.3550 |
| mean_tvd[risk] | 0.4382 |
| order_flip_rate[action] | 0.3833 |
| order_flip_rate[category] | 0.3833 |
| order_flip_rate[fraud] | 0.0167 |
| order_flip_rate[needs_human] | 0.0167 |
| order_flip_rate[priority] | 0.3667 |
| order_flip_rate[risk] | 0.5167 |
| valid_accuracy | 0.5417 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| category | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.3333 |
| risk | 12 | 0.4167 [0.1933, 0.6805] (wilson) | 0.4167 |
| action | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| needs_human | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| priority | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| fraud | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.7500 |
