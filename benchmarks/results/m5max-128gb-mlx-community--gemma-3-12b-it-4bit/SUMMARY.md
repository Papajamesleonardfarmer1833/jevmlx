# Bench summary

| machine | model | track | scorer | dataset | field acc | case exact | bal acc mean | ECE | any-flip | perturb-flip | p50 latency (ms) | calls | n_cases |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | naive_local | slots | bundled | 0.7917 | 0.5139 | 0.625 | 0.625 | 0.7676 | — | — | — | 930.165 | 24 | 24 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | naive_local | slots | perturbed | 0.7604 | 0.5139 | 0.6146 | 0.6146 | 0.7433 | — | — | 0.0972 | 1019.755 | 96 | 96 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | naive_local | slots | typesafe | 0.4192 | 0.8577 | 0.3636 | 0.3636 | 0.4062 | — | — | — | 2923.7 | 45 | 45 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | labels | bundled | 0.7269 | 0.5139 | 0.4167 | 0.4167 | 0.7277 | 0.2346 | 0.0389 | — | 408.68 | 24 | 24 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | labels | perturbed | 0.7251 | 0.5139 | 0.375 | 0.375 | 0.7223 | 0.2336 | 0.0382 | 0.0509 | 442.48 | 96 | 96 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | labels | typesafe | 0.6185 | 0.8601 | 0.2667 | 0.2727 | 0.6355 | 0.2099 | 0.0692 | — | 1181.59 | 44 | 45 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | slots | bundled | 0.5509 | 0.5139 | 0 | 0 | 0.5834 | 0.3181 | 0.25 | — | 458.105 | 24 | 24 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | slots | perturbed | 0.5475 | 0.5139 | 0 | 0 | 0.5817 | 0.3216 | 0.2549 | 0.1898 | 447.48 | 96 | 96 |
| m5max-128gb | mlx-community--gemma-3-12b-it-4bit | parallel | slots | typesafe | 0.3649 | 0.8601 | 0.0889 | 0.0909 | 0.4749 | 0.4153 | 0.1025 | — | 1224.805 | 44 | 45 |
