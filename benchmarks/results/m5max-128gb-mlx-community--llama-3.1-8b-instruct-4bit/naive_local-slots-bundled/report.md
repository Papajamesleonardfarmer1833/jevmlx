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
| timestamp_utc | 2026-09-21T17:40:03+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6667 [0.5278, 0.7917] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.4167 |
| case_exact_match | 0.4167 |
| balanced_accuracy[action] | 0.4944 |
| balanced_accuracy[category] | 0.9167 |
| balanced_accuracy[fraud] | 0.6667 |
| balanced_accuracy[needs_human] | 0.6429 |
| balanced_accuracy[priority] | 0.6905 |
| balanced_accuracy[risk] | 0.4125 |
| macro_f1[action] | 0.5000 |
| macro_f1[category] | 0.9143 |
| macro_f1[fraud] | 0.7000 |
| macro_f1[needs_human] | 0.5556 |
| macro_f1[priority] | 0.6667 |
| macro_f1[risk] | 0.4167 |
| valid_accuracy | 0.6667 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| action | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| risk | 12 | 0.5000 [0.2538, 0.7462] (wilson) | 0.4167 |
| needs_human | 12 | 0.5833 [0.3195, 0.8067] (wilson) | 0.5833 |
| priority | 12 | 0.6667 [0.3906, 0.8619] (wilson) | 0.5833 |
| fraud | 12 | 0.8333 [0.5520, 0.9530] (wilson) | 0.7500 |
| category | 12 | 0.9167 [0.6461, 0.9851] (wilson) | 0.3333 |
