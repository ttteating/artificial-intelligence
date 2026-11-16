# 条款图覆盖率与 20 条边抽查（B 线 `data/graph.json`）

产物：`data/graph.json`（3600 个场景）、`data/graph_stats.json`

## 1. 覆盖率（分场景图口径）

| 指标 | 数值 |
|---|---:|
| 场景数 | 3600 |
| 节点（条款级）总数 | 5559 |
| 每合同平均条款数 | 1.54 |
| 功能边（非 same_section）总数 | 292 |
| 至少有一条功能边的合同占比 | 217/3600 = 6.03% |
| 条款级节点覆盖率（有功能边的节点/全部节点） | 493/5559 = 8.87% |
| 锚点未解析次数（累计） | 8382 |
| 孤立节点占比 | 0.4067 |

边类型分布：

| 边类型 | 条数 |
|---|---:|
| same_section | 5254 |
| cites | 228 |
| exception_of | 59 |
| conditions_on | 5 |

映射方式分布（`heading`=条款文本以编号开头，可信；`id_number`=编号直配，启发式；`fallback`=同合同兜底边）：

| 映射方式 | 条数 |
|---|---:|
| fallback | 5254 |
| id_number | 292 |

## 2. 抽查 20 条边

| # | 场景 | src | dst | 边类型 | 映射 | conf | 锚点 | src 文本 | dst 文本 | 判定 |
|---:|---|---|---|---|---|---:|---|---|---|---|
| 1 | 1092 | evidence_7 | evidence_3 | cites | id_number | 0.30 | in | If MBRK terminates the Agreement in years 2 or 3 prior to the period 6 | MBRK may terminate this Agreement upon 60 days notice for a Change of  | 存疑（编号推断） |
| 2 | 1711 | evidence_3 | evidence_4 | conditions_on | id_number | 0.60 | provided that | For two (2) years following the Effective Time, QIWI covenants that at | For two (2) years following the Effective Time, QIWI covenants that at | 存疑（编号推断） |
| 3 | 1398 | evidence_6 | evidence_4 | exception_of | id_number | 0.60 | subject to | Subject to the terms and conditions of this Agreement, as of the Distr | shall retain those licenses granted to it under this Agreement, provid | 存疑（编号推断） |
| 4 | 264 | evidence_2 | evidence_8 | same_section | fallback | 0.50 | same_contract | Disclosing Party may serve written request on Recipient for return or  | Disclosing Party may serve written request on Recipient for return or  | 仅结构兜底 |
| 5 | 2745 | evidence_10 | evidence_9 | same_section | fallback | 0.50 | same_contract | Except in the event of (i) Third Party Claims subject to a Party's ind | Upon termination of this Agreement by Customer in whole or in part or  | 仅结构兜底 |
| 6 | 1837 | evidence_1 | evidence_2 | same_section | fallback | 0.50 | same_contract | Upon receipt by the Recipient of a written demand from the Disclosers: | The Recipient acknowledges that ownership of the Confidential Informat | 仅结构兜底 |
| 7 | 2520 | evidence_9 | evidence_13 | same_section | fallback | 0.50 | same_contract | Any and all data, information, and material created, conceived, reduce | EXCEPT IN CONNECTION WITH A BREACH OF ARTICLE XIV HEREUNDER, AND WITH  | 仅结构兜底 |
| 8 | 227 | evidence_6 | evidence_5 | cites | id_number | 0.30 | in | SpinCo, for itself and as representative of all other members of the S | SpinCo, for itself and as representative of all other members of the S | 存疑（编号推断） |
| 9 | 2133 | evidence_5 | evidence_7 | same_section | fallback | 0.50 | same_contract | The Parties will reasonably cooperate with each other with respect to  | The Parties agree that any termination of a Study Order shall allow fo | 仅结构兜底 |
| 10 | 1519 | evidence_9 | evidence_8 | conditions_on | id_number | 0.60 | provided that | Except as expressly provided in Section 3(a), Affiliate shall not have | Network hereby grants Affiliate during the Term a royalty-free, fully  | 存疑（编号推断） |
| 11 | 406 | evidence_1 | evidence_2 | cites | id_number | 0.60 | pursuant to | In addition to any outstanding Guarantee payments which Licensee shall | Licensee shall develop and distribute the following Wireless Products  | 存疑（编号推断） |
| 12 | 1711 | evidence_5 | evidence_4 | conditions_on | id_number | 0.60 | provided that | For two (2) years following the Effective Time, QIWI covenants that at | For two (2) years following the Effective Time, QIWI covenants that at | 存疑（编号推断） |
| 13 | 1208 | evidence_3 | evidence_1 | same_section | fallback | 0.50 | same_contract | The parties expect to make available to one another certain non-public | Promptly upon notice from either party that it does not wish to procee | 仅结构兜底 |
| 14 | 3057 | evidence_10 | evidence_9 | same_section | fallback | 0.50 | same_contract | Subject to the terms and conditions of this Agreement and such other s | Any and all data, information, and material created, conceived, reduce | 仅结构兜底 |
| 15 | 2254 | evidence_9 | evidence_8 | conditions_on | id_number | 0.60 | provided that | Except as expressly provided in Section 3(a), Affiliate shall not have | Network hereby grants Affiliate during the Term a royalty-free, fully  | 存疑（编号推断） |
| 16 | 249 | evidence_6 | evidence_4 | cites | id_number | 0.60 | pursuant to | Subject to this Section 4.8 and to applicable Legal Requirements, Depo | Any attempted assignment without consent shall be void. / This Agreeme | 存疑（编号推断） |
| 17 | 1083 | evidence_8 | evidence_9 | cites | id_number | 0.42 | under | Except for claims under Sections 9.4 [INDEMNIFICATION BY IMPRESSE] and | Except for claims under Sections 9.4 [INDEMNIFICATION BY IMPRESSE] and | 存疑（编号推断） |
| 18 | 2838 | evidence_9 | evidence_8 | conditions_on | id_number | 0.60 | provided that | Except as expressly provided in Section 3(a), Affiliate shall not have | Network hereby grants Affiliate during the Term a royalty-free, fully  | 存疑（编号推断） |
| 19 | 3008 | evidence_7 | evidence_8 | same_section | fallback | 0.50 | same_contract | In no event will either party be liable to the other for special, inci | Regardless of the basis on which Customer is entitled to claim damages | 仅结构兜底 |
| 20 | 1887 | evidence_6 | evidence_2 | exception_of | id_number | 0.60 | subject to | Without limiting the foregoing, but subject to the restrictions set fo | A Termination Event shall have occurred if: / (c) Supplier either: (i) | 存疑（编号推断） |

## 3. 结论

- 锚点式抽取在全量 train 上的功能边覆盖率只有 **8.87%**（292 条功能边 / 5559 个条款节点），绝大多数边是 `same_section` 兜底边——ACE 的条款文本只给 `evidence_N` 编号，没有章节号，`Section 9` 这类引用无法可靠映射回条款，这正是文档 §1.4 所说"保守锚点抽取、边稀疏但精确"的实测代价。
- 抽查中凡 `mapping=fallback` 的边都只是"同一合同"的结构连接，不承载 DEFINES/EXCEPTIONS 等法律语义；`id_number` 边需要用条款文本人工复核。
- 因此实验链路的条款图采用 `data/facet_graph.py` 的 facet 级图（DEFINES/COMPLEMENTS/EXCEPTIONS/DEPENDS_ON/CONFLICTS/SEQUENCE/CITES 七类边，覆盖 600 个 dev 场景 / 2,534 三元组），B 线 `graph.json` 作为"锚点基线 + 覆盖率评估"交付，不参与搜索。
