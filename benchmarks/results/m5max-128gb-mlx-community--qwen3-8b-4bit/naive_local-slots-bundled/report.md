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
| timestamp_utc | 2026-09-21T12:26:42+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.8194 [0.7083, 0.9167] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.6250 |
| case_exact_match | 0.6250 |
| balanced_accuracy[action] | 0.8333 |
| balanced_accuracy[category] | 0.9167 |
| balanced_accuracy[fraud] | 0.9444 |
| balanced_accuracy[needs_human] | 0.7143 |
| balanced_accuracy[priority] | 0.6667 |
| balanced_accuracy[risk] | 0.6250 |
| macro_f1[action] | 0.8110 |
| macro_f1[category] | 0.9143 |
| macro_f1[fraud] | 0.8992 |
| macro_f1[needs_human] | 0.6571 |
| macro_f1[priority] | 0.6250 |
| macro_f1[risk] | 0.5606 |
| valid_accuracy | 0.8194 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| risk | 12 | 0.7500 [0.4677, 0.9111] (wilson) | 0.4167 |
| action | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.4167 |
| priority | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.5833 |
| category | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.3333 |
| fraud | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.7500 |
