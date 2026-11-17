# 真实 LLM 通路冒烟

- model: `deepseek-flash`；base_url: `https://api.tokenmeld.com/v1`
- 真实请求次数: 3；缓存命中: 0
- 每次调用的 `violation` 为 true 表示模型输出了合规标签词，分数已被清零

| 调用 | 返回 | 耗时(s) | 错误 |
|---|---|---:|---|
| valid-A | `{"valid": true, "confidence": 0.95, "why": "Target expressly states an exception to the consent requirement.", "_tag": "valid-A"}` | 8.4 | — |
| valid-B | `{"valid": false, "confidence": 0.86, "why": "Target restates consent requirement, not an exception to it.", "_tag": "valid-B"}` | 4.55 | — |
| reward | `{"reward": 0.75, "why": "Exception not triggered because royalties remain unpaid; consent requirement still applies.", "_tag": "reward"}` | 5.0 | — |

通过标准：三条都返回 JSON、`valid`/`reward` 字段可解析、没有 `violation`。
通过后再跑 `python -m experiments.run_baselines --condition H --split dev --limit 20 --llm real` 验证接到搜索链上。
