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
| timestamp_utc | 2026-09-21T15:01:44+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.4560 [0.4207, 0.4919] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.0000 |
| case_exact_match | 0.0000 |
| brier | 0.8214 [0.7949, 0.8495] (case_cluster_bootstrap) |
| correctness_auroc | 0.6710 |
| ece_5bin_equal_mass | 0.2383 |
| tie_rate | 0.0046 |
| any_flip_rate[action] | 0.4667 |
| any_flip_rate[category] | 0.4542 |
| any_flip_rate[fraud] | 0.0000 |
| any_flip_rate[needs_human] | 0.0375 |
| any_flip_rate[priority] | 0.3917 |
| any_flip_rate[risk] | 0.5083 |
| balanced_accuracy[action] | 0.3081 |
| balanced_accuracy[category] | 0.1684 |
| balanced_accuracy[fraud] | 0.5000 |
| balanced_accuracy[needs_human] | 0.6089 |
| balanced_accuracy[priority] | 0.4511 |
| balanced_accuracy[risk] | 0.3333 |
| macro_f1[action] | 0.3028 |
| macro_f1[category] | 0.1471 |
| macro_f1[fraud] | 0.4286 |
| macro_f1[needs_human] | 0.5963 |
| macro_f1[priority] | 0.4491 |
| macro_f1[risk] | 0.2862 |
| mean_tvd[action] | 0.2610 |
| mean_tvd[category] | 0.3516 |
| mean_tvd[fraud] | 0.0240 |
| mean_tvd[needs_human] | 0.0240 |
| mean_tvd[priority] | 0.2504 |
| mean_tvd[risk] | 0.4151 |
| order_flip_rate[action] | 0.4667 |
| order_flip_rate[category] | 0.4542 |
| order_flip_rate[fraud] | 0.0000 |
| order_flip_rate[needs_human] | 0.0375 |
| order_flip_rate[priority] | 0.3917 |
| order_flip_rate[risk] | 0.5083 |
| perturbation_flip_rate | 0.2407 |
| valid_accuracy | 0.4560 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| category | 48 | 0.2083 [0.1173, 0.3426] (wilson) † | 0.3333 |
| action | 48 | 0.2708 [0.1657, 0.4100] (wilson) † | 0.4167 |
| risk | 48 | 0.2917 [0.1824, 0.4318] (wilson) † | 0.4167 |
| needs_human | 48 | 0.6250 [0.4836, 0.7478] (wilson) | 0.5833 |
| priority | 48 | 0.6458 [0.5044, 0.7657] (wilson) | 0.5833 |
| fraud | 48 | 0.7500 [0.6122, 0.8508] (wilson) | 0.7500 |
