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
| timestamp_utc | 2026-09-21T12:32:48+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.8160 [0.7604, 0.8681] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.6354 |
| case_exact_match | 0.6354 |
| balanced_accuracy[action] | 0.8333 |
| balanced_accuracy[category] | 0.9167 |
| balanced_accuracy[fraud] | 0.9444 |
| balanced_accuracy[needs_human] | 0.7143 |
| balanced_accuracy[priority] | 0.6845 |
| balanced_accuracy[risk] | 0.6250 |
| macro_f1[action] | 0.8110 |
| macro_f1[category] | 0.9143 |
| macro_f1[fraud] | 0.8992 |
| macro_f1[needs_human] | 0.6571 |
| macro_f1[priority] | 0.6708 |
| macro_f1[risk] | 0.5606 |
| perturbation_flip_rate | 0.0139 |
| valid_accuracy | 0.8160 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 48 | 0.6667 [0.5254, 0.7832] (wilson) | 0.5833 |
| risk | 48 | 0.7500 [0.6122, 0.8508] (wilson) | 0.4167 |
| priority | 48 | 0.8125 [0.6806, 0.8981] (wilson) | 0.5833 |
| action | 48 | 0.8333 [0.7042, 0.9130] (wilson) | 0.4167 |
| category | 48 | 0.9167 [0.8045, 0.9671] (wilson) | 0.3333 |
| fraud | 48 | 0.9167 [0.8045, 0.9671] (wilson) | 0.7500 |
