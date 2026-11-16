# 边人工抽查（dev，随机 20 条）

抽检要点：① 边类型与两端文本的语义是否匹配；② `src/dst` 是否落在同一场景的条款上；③ `w`（边权 = 目标条款与场景的相似度）是否随相关度变化。

| # | sid | src | etype | dst | w | src 角色/文本片段 | dst 角色/文本片段 |
|---:|---|---|---|---|---:|---|---|
| 1 | tr01708 | evidence_5#s0 | CITES | evidence_7#s1 | 0.193 | COND: If we do not exercise our right of first refusal, the transf | OBL: (b) you must pay any amounts owed for purchases from us and  |
| 2 | tr00785 | evidence_9#s0 | DEPENDS_ON | evidence_9#s1 | 0.071 | OBL: in (a) or (b) above, the otherwise applicable royalty rate w | COND: In the event that Array (a) [ * ] during [ * ], or (b) inten |
| 3 | tr02016 | evidence_6#s1 | EXCEPTIONS | evidence_6#s0 | 0.129 | OBL: Bellicum's Permitted Use of the Miltenyi Products shall be l | EXC: The supply of the Miltenyi Products hereunder conveys to Bel |
| 4 | tr00371 | evidence_1#s0 | CITES | evidence_2#s0 | 0.052 | EXC: This Agreement shall be automatically renewed for successive | EXC: This Agreement shall be automatically renewed for successive |
| 5 | tr00445 | evidence_7#s0 | COMPLEMENTS | evidence_10#s0 | 0.212 | EXC: [***] hereby grants to [***] a non-exclusive, non-transferab | REMEDY: diagnostic industry practices. | Each Party shall maintain,  |
| 6 | tr03138 | evidence_5#s1 | EXCEPTIONS | evidence_5#s0 | 0.051 | OBL: (b) patents, patent applications, plant variety rights, plan | EXC: CERES and IGER shall use reasonable efforts to avoid any act |
| 7 | tr00617 | evidence_14#s0 | CITES | evidence_10#s2 | 0.109 | EXC: EXCEPT IN THE CASE OF (a) ANY BREACH OF SECTION 10 (CONFIDEN | OBL: (c) [***]; |
| 8 | tr01885 | evidence_7#s0 | EXCEPTIONS | evidence_7#s1 | 0.037 | DEF: 3.8. "Representatives" means, in relation to a Party to the  | EXC: Recipient shall: | 5.1.3. not copy or reproduce (or permit t |
| 9 | tr03407 | evidence_5#s0 | EXCEPTIONS | evidence_5#s4 | 0.046 | COND: Customer may, without Manufacturer's consent, assign the rig | EXC: In the event that Manufacturer or any of its Affiliates, dir |
| 10 | tr00383 | evidence_4#s0 | EXCEPTIONS | evidence_5#s0 | 0.212 | DEF: (i) the other Party ceases or threatens to cease to carry on | EXC: Subject to Section 20.2, neither Party shall, nor shall it p |
| 11 | tr02858 | evidence_10#s1 | CITES | evidence_10#s0 | 0.082 | OBL: Such policies shall name AIRSOPURE as an additional insured  | SCOPE: The evidence of insurance shall include a statement by the i |
| 12 | tr01209 | evidence_6#s0 | CONFLICTS | evidence_7#s0 | 0.044 | OBL: Know-How, Achaogen Patents, and Achaogen Materials to resear | EXC: [***] hereby grants to [***] a non-exclusive, non-transferab |
| 13 | tr00277 | evidence_1#s0 | CITES | evidence_2#s0 | 0.234 | OBL: Thereafter, this Agreement shall automatically renew for sub | OBL: Thereafter, this Agreement shall automatically renew for sub |
| 14 | tr00586 | evidence_10#s3 | EXCEPTIONS | evidence_10#s2 | 0.355 | OBL: (a) provided, that it is understood that the Company-Skype B | EXC: subject to the provisions of Section 5), license during the  |
| 15 | tr02236 | evidence_13#s0 | COMPLEMENTS | evidence_13#s2 | 0.101 | OBL: 3. | OBL: Neither Party may assign, transfer or sell any of its rights |
| 16 | tr02157 | evidence_6#s0 | EXCEPTIONS | evidence_6#s4 | 0.121 | OBL: Certegy hereby grants, and will cause the other members of t | EXC: subject to the following: (i) Certegy shall not sublicense,  |
| 17 | tr00445 | evidence_8#s0 | COMPLEMENTS | evidence_10#s0 | 0.212 | EXC: EXCEPT WITH RESPECT TO DAMAGES THAT ARISE DUE TO A PARTY'S B | REMEDY: diagnostic industry practices. | Each Party shall maintain,  |
| 18 | tr01299 | evidence_2#s0 | COMPLEMENTS | evidence_2#s3 | 0.345 | OBL: 1. | OBL: Ann. |
| 19 | tr00617 | evidence_11#s0 | CITES | evidence_10#s2 | 0.109 | EXC: ENERGOUS hereby grants DIALOG a non-exclusive, non-transfera | OBL: (c) [***]; |
| 20 | tr03245 | evidence_13#s0 | COMPLEMENTS | evidence_13#s1 | 0.021 | COND: Throughout the entire Term, you must maintain such types of  | REMEDY: (b) comprehensive automobile liability for owned, hired and  |

判定口径（人工复核时填）：

- 语义匹配：边类型与两端文本的语义关系一致（如 EXCEPTIONS 两端确实是"主句 vs 例外句"）
- 可加载性：jsonl 每行是一个场景的 `{sid, version, entry, nodes, triples}`，共 600 行
