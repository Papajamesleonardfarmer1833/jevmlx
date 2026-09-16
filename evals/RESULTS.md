# Full head-to-head results

Agreement with TypeSafe's reference (consensus of GPT-6 Astra + Fable 5.1) on all
publicly shipped cases of their four workflows: 20 cases, 373 reference question-pairs.

## All answered pairs

| Model | Overall | Security | AgentTrace | Invoice | CustomerSvc |
|---|---|---|---|---|---|
| Sol (published) | **88.7% (315/355)** | 31/37 | 39/49 | 170/184 | 75/85 |
| Opus (published) | **89.5% (324/362)** | 29/37 | 42/49 | 170/184 | 83/92 |
| DeepSeek v4.1 Flash (max) | **88.2% (328/372)** | 35/48 | 41/49 | 165/183 | 87/92 |
| Jev / TypeSafe (published) | **86.9% (305/351)** | 20/26 | 35/49 | 169/184 | 81/92 |
| local Qwen2.5-7B (M5 Air) | **73.2% (273/373)** | 30/48 | 30/49 | 149/184 | 64/92 |
| local Qwen3-8B (M5 Air) | **71.4% (135/189)** | 34/48 | 32/49 | – | 69/92 |

## Strict common subset (pairs answered by every model)

Sizes: Security 26, AgentTrace 49, Invoice 183, CustomerSvc 85

| Model | Overall | Security | AgentTrace | Invoice | CustomerSvc |
|---|---|---|---|---|---|
| Sol (published) | **89.2% (306/343)** | 23/26 | 39/49 | 169/183 | 75/85 |
| Opus (published) | **89.8% (308/343)** | 21/26 | 42/49 | 169/183 | 76/85 |
| DeepSeek v4.1 Flash (max) | **89.5% (307/343)** | 21/26 | 41/49 | 165/183 | 80/85 |
| Jev / TypeSafe (published) | **86.6% (297/343)** | 20/26 | 35/49 | 168/183 | 74/85 |
| local Qwen2.5-7B (M5 Air) | **73.8% (253/343)** | 16/26 | 30/49 | 148/183 | 59/85 |
| local Qwen3-8B (M5 Air) | **71.2% (114/160)** | 19/26 | 32/49 | – | 63/85 |
