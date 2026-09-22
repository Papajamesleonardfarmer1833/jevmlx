# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | naive_local | slots | bundled | 0.6667 | 0.5139 | 0.4167 | 0.4167 | 0.6373 | — | — | — | 458.55 | 24 | 24 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | naive_local | slots | perturbed | 0.6146 | 0.5139 | 0.3854 | 0.3854 | 0.582 | — | — | 0.1574 | 575.755 | 96 | 96 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | naive_local | slots | typesafe | 0.2548 | 0.8577 | 0.1591 | 0.1591 | 0.2735 | — | — | — | 1491.08 | 45 | 45 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | labels | bundled | 0.7454 | 0.5139 | 0.25 | 0.25 | 0.723 | 0.0547 | 0.0528 | — | 248.365 | 24 | 24 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | labels | perturbed | 0.7245 | 0.5139 | 0.2396 | 0.2396 | 0.7016 | 0.0559 | 0.0674 | 0.088 | 263.895 | 96 | 96 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | labels | typesafe | 0.6438 | 0.8601 | 0.0667 | 0.0682 | 0.6057 | 0.0987 | 0.0556 | — | 707.1 | 44 | 45 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | slots | bundled | 0.4329 | 0.5139 | 0 | 0 | 0.376 | 0.2633 | 0.3361 | — | 243.155 | 24 | 24 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | slots | perturbed | 0.456 | 0.5139 | 0 | 0 | 0.395 | 0.2383 | 0.3097 | 0.2407 | 262.435 | 96 | 96 |
| m5max-128gb | mlx-community--llama-3.1-8b-instruct-4bit | parallel | slots | typesafe | 0.6485 | 0.8601 | 0.0667 | 0.0682 | 0.575 | 0.0519 | 0.1093 | — | 705.405 | 44 | 45 |
