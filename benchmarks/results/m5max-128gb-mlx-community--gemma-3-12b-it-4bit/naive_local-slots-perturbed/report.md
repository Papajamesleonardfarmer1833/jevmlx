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
| timestamp_utc | 2026-09-22T04:38:32+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7604 [0.6910, 0.8229] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.5139 |
| exact record | 0.6146 |
| case_exact_match | 0.6146 |
| balanced_accuracy[action] | 0.7569 |
| balanced_accuracy[category] | 0.8542 |
| balanced_accuracy[fraud] | 0.7917 |
| balanced_accuracy[needs_human] | 0.6786 |
| balanced_accuracy[priority] | 0.7381 |
| balanced_accuracy[risk] | 0.6406 |
| macro_f1[action] | 0.7673 |
| macro_f1[category] | 0.8800 |
| macro_f1[fraud] | 0.8360 |
| macro_f1[needs_human] | 0.6268 |
| macro_f1[priority] | 0.7095 |
| macro_f1[risk] | 0.6246 |
| perturbation_flip_rate | 0.0972 |
| valid_accuracy | 0.7849 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| needs_human | 48 | 0.6250 [0.4836, 0.7478] (wilson) | 0.5833 |
| priority | 48 | 0.6458 [0.5044, 0.7657] (wilson) | 0.5833 |
| action | 48 | 0.7708 [0.6346, 0.8669] (wilson) | 0.4167 |
| risk | 48 | 0.7708 [0.6346, 0.8669] (wilson) | 0.4167 |
| category | 48 | 0.8542 [0.7283, 0.9275] (wilson) | 0.3333 |
| fraud | 48 | 0.8958 [0.7783, 0.9547] (wilson) | 0.7500 |
