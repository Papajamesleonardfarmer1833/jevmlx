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
| timestamp_utc | 2026-09-21T17:46:10+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6146 [0.5382, 0.6840] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.3854 |
| case_exact_match | 0.3854 |
| balanced_accuracy[action] | 0.5153 |
| balanced_accuracy[category] | 0.7604 |
| balanced_accuracy[fraud] | 0.7083 |
| balanced_accuracy[needs_human] | 0.5143 |
| balanced_accuracy[priority] | 0.5655 |
| balanced_accuracy[risk] | 0.4281 |
| macro_f1[action] | 0.5148 |
| macro_f1[category] | 0.8180 |
| macro_f1[fraud] | 0.7498 |
| macro_f1[needs_human] | 0.4606 |
| macro_f1[priority] | 0.6088 |
| macro_f1[risk] | 0.4278 |
| perturbation_flip_rate | 0.1574 |
| valid_accuracy | 0.6705 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 48 | 0.4583 [0.3258, 0.5971] (wilson) † | 0.5833 |
| action | 48 | 0.5208 [0.3833, 0.6553] (wilson) | 0.4167 |
| risk | 48 | 0.5208 [0.3833, 0.6553] (wilson) | 0.4167 |
| priority | 48 | 0.5833 [0.4428, 0.7115] (wilson) | 0.5833 |
| category | 48 | 0.7500 [0.6122, 0.8508] (wilson) | 0.3333 |
| fraud | 48 | 0.8542 [0.7283, 0.9275] (wilson) | 0.7500 |
