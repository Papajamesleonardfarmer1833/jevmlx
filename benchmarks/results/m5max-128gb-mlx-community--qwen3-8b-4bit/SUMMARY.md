# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--qwen3-8b-4bit | naive_local | slots | bundled | 0.8194 | 0.5139 | 0.625 | 0.625 | 0.7834 | — | — | — | 433.86 | 24 | 24 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | naive_local | slots | perturbed | 0.816 | 0.5139 | 0.6354 | 0.6354 | 0.7864 | — | — | 0.0139 | 551.08 | 96 | 96 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | naive_local | slots | typesafe | 0.6932 | 0.8577 | 0.3864 | 0.3864 | 0.6718 | — | — | — | 1584.43 | 45 | 45 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | labels | bundled | 0.7847 | 0.5139 | 0.4583 | 0.4583 | 0.7677 | 0.1897 | 0.0222 | — | 233.935 | 24 | 24 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | labels | perturbed | 0.7795 | 0.5139 | 0.4583 | 0.4583 | 0.7516 | 0.1931 | 0.034 | 0.037 | 256.015 | 96 | 96 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | labels | typesafe | 0.8462 | 0.8601 | 0.1778 | 0.1818 | 0.7443 | 0.1246 | 0.0426 | — | 646.875 | 44 | 45 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | slots | bundled | 0.5417 | 0.5139 | 0 | 0 | 0.5014 | 0.3714 | 0.2806 | — | 256.95 | 24 | 24 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | slots | perturbed | 0.5486 | 0.5139 | 0 | 0 | 0.5113 | 0.3612 | 0.2778 | 0.1343 | 252.06 | 96 | 96 |
| m5max-128gb | mlx-community--qwen3-8b-4bit | parallel | slots | typesafe | 0.3473 | 0.8601 | 0.2 | 0.2045 | 0.5044 | 0.5284 | 0.1022 | — | 691 | 44 | 45 |
