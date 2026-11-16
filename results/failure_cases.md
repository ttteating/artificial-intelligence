# 失败案例（自动挖掘）
## 偏离计数（dev + test 合计）
| 现象 | 条数 |
|---|---:|
| A1≠B0 且 B0 对 | 11 |
| H≠A1 且 A1 对 | 12 |
| 规则过约束（B1≠B0 + H 剪枝主导） | 8 |
| M0≠A0 且 A0 对 | 28 |

## heuristic_misled
### tr00117（dev，gold=Compliant）
```json
{
 "sid": "tr00117",
 "split": "dev",
 "gold": "Compliant",
 "A1": "Non-Compliant",
 "B0": "Compliant",
 "A0": "Compliant",
 "A1_expanded": 16,
 "extra_expanded": 0,
 "n_clauses": 4,
 "oracle_hops": 4
}
```
- 场景摘录：Supplier, under a distribution agreement governed by the laws of New York, is obligated to adhere to several clauses involving the ambiguous phrase 'in connection with'. Supplier appoints Reseller to market and sell its products. In compliance with clause evid…
- 条款 `evidence_3`：This Agreement shall be governed by the laws of the State of New York and all disputes and controversies arising out of or in connection with the Agreement shall be brought exclusi…
- 条款 `evidence_6`：In connection with the foregoing appointment, Supplier hereby grants Reseller a non-transferable, revocable, limited right to resell, market, promote, stimulate interest in, and so…
- 条款 `evidence_7`：In connection with the foregoing appointment, Supplier hereby grants Reseller a non-transferable, revocable, limited right to resell, market, promote, stimulate interest in, and so…
### tr00715（dev，gold=Compliant）
```json
{
 "sid": "tr00715",
 "split": "dev",
 "gold": "Compliant",
 "A1": "Not-Applicable",
 "B0": "Not-Applicable",
 "A0": "Compliant",
 "A1_expanded": 16,
 "extra_expanded": 0,
 "n_clauses": 3,
 "oracle_hops": 2
}
```
- 场景摘录：The franchisee ('you'), pursuant to the transfer clause, intended to sell their franchise to a Proposed New Owner. To comply with the obligation requiring payment of all outstanding amounts owed to the franchisor before the transfer (Clause 1(b)), the franchis…
- 条款 `evidence_7`：If you propose to Transfer this Agreement, the Franchise or its assets, or any Interest, or if any of your Principal Owners proposes to Transfer a controlling Interest in you or ma…
- 条款 `evidence_9`：You agree not to use any Mark as part of any corporate or trade name or with any prefix, suffix, or other modifying words, terms, designs, or symbols (other than logos and addition…
- 条款 `evidence_14`：You must provide us with copies of policies evidencing the existence of such insurance concurrently with execution of this Agreement and prior to each subsequent renewal date of ea…
### tr00736（dev，gold=Compliant）
```json
{
 "sid": "tr00736",
 "split": "dev",
 "gold": "Compliant",
 "A1": "Non-Compliant",
 "B0": "Compliant",
 "A0": "Compliant",
 "A1_expanded": 16,
 "extra_expanded": 0,
 "n_clauses": 4,
 "oracle_hops": 3
}
```
- 场景摘录：ETI, on behalf of Entrust, was obligated under the Reseller Agreement to ensure that NTL received terms for the Enterprise License that were 'no less favorable' than those offered to other licensees receiving 'substantially similar rights,' taking into account…
- 条款 `evidence_2`：During the life of the Reseller Agreement, ----------------------- it is the intention of ETI that the terms of the Reseller Agreement shall be no less favourable to Nortel than th…
- 条款 `evidence_4`：following rights: (a) to make, use, lease, sell or otherwise dispose of, maintain and repair, Licensed Products, to license the use of Licensed Products made by or for Grantee, to …
- 条款 `evidence_6`：Any rights or license granted under this Article VI to a ----------------- corporation or other legal entity which becomes a Subsidiary of a Party at a date later than the Effectiv…
## llm_hallucination
### te00262（test，gold=Compliant）
```json
{
 "sid": "te00262",
 "split": "test",
 "gold": "Compliant",
 "H": "Non-Compliant",
 "A1": "Compliant",
 "llm_calls": 223,
 "llm_rejected": 43,
 "H_expanded": 16
}
```
- 场景摘录：PB entered into a definitive agreement for a Change of Control and exercised its right to make a Change of Control Buy-Out Payment to SFJ within the 120-day timeframe stipulated in evidence_4. However, instead of providing formal written notice via a tradition…
- 条款 `evidence_2`：Notwithstanding any of the foregoing, without the consent of PB, which consent may be withheld in PB's sole discretion, SFJ shall not sell, assign, sublicense or otherwise transfer…
- 条款 `evidence_4`：Within one hundred and twenty (120) days following the closing of a Change of Control, PB or its successor shall have the right to make a one-time payment (the "Change of Control B…
- 条款 `evidence_5`：PB shall not sell, transfer or assign, directly or indirectly, in whole or in part, any rights to receive payments of royalties or license fees with respect to the Product or the P…
### te00737（test，gold=Non-Compliant）
```json
{
 "sid": "te00737",
 "split": "test",
 "gold": "Non-Compliant",
 "H": "Compliant",
 "A1": "Non-Compliant",
 "llm_calls": 153,
 "llm_rejected": 0,
 "H_expanded": 16
}
```
- 场景摘录：NCM operates a service that licenses its Marks and Derived Works to Network Affiliate under specific contractual terms. During the Term of the Agreement, Network Affiliate proposes using an NCM Mark in an advertising campaign targeting a niche demographic. Ins…
- 条款 `evidence_10`：Subject to the terms and conditions of this Agreement and such other standards, trademark usage guidelines and specifications as are prescribed by NCM during the term of this Agree…
- 条款 `evidence_11`：Subject to the terms and conditions of this Agreement and such other standards, trademark usage guidelines and specifications as are prescribed by NCM during the term of this Agree…
- 条款 `evidence_16`：Neither party will at any time, except to the extent necessary to assert or defend its rights under this Agreement: (i) challenge or otherwise do anything inconsistent with the oth…
### te00776（test，gold=Non-Compliant）
```json
{
 "sid": "te00776",
 "split": "test",
 "gold": "Non-Compliant",
 "H": "Compliant",
 "A1": "Non-Compliant",
 "llm_calls": 40,
 "llm_rejected": 11,
 "H_expanded": 11
}
```
- 场景摘录：Company A and Company B enter into negotiations for a potential joint venture involving chemical informatics. During the discussions, Company A (the 'Receiving Party') receives proprietary information from Company B (the 'Disclosing Party') about its chemical …
- 条款 `evidence_3`：Except to the extent required by law, neither party shall disclose the existence or subject matter of the negotiations or business relationship contemplated by this Agreement.…
- 条款 `evidence_4`：Each undersigned party (the "Receiving Party") understands that the other party (the "Disclosing Party") has disclosed or may disclose information relating to the Disclosing Party'…
- 条款 `evidence_8`：Immediately upon | (i) the decision by either party not to enter into a relationship as a result of the exchange of information hereunder, or | (ii) a request by the Disclosing Par…
## rule_overconstraint
### tr00117（dev，gold=Compliant）
```json
{
 "sid": "tr00117",
 "split": "dev",
 "gold": "Compliant",
 "B1": "Non-Compliant",
 "B0": "Compliant",
 "pruned": 4,
 "steps": 3,
 "expanded": 16,
 "source": "B1≠B0"
}
```
- 场景摘录：Supplier, under a distribution agreement governed by the laws of New York, is obligated to adhere to several clauses involving the ambiguous phrase 'in connection with'. Supplier appoints Reseller to market and sell its products. In compliance with clause evid…
- 条款 `evidence_3`：This Agreement shall be governed by the laws of the State of New York and all disputes and controversies arising out of or in connection with the Agreement shall be brought exclusi…
- 条款 `evidence_6`：In connection with the foregoing appointment, Supplier hereby grants Reseller a non-transferable, revocable, limited right to resell, market, promote, stimulate interest in, and so…
- 条款 `evidence_7`：In connection with the foregoing appointment, Supplier hereby grants Reseller a non-transferable, revocable, limited right to resell, market, promote, stimulate interest in, and so…
### te00262（test，gold=Compliant）
```json
{
 "sid": "te00262",
 "split": "test",
 "gold": "Compliant",
 "B1": "Non-Compliant",
 "B0": "Compliant",
 "pruned": 101,
 "steps": 2,
 "expanded": 16,
 "source": "B1≠B0"
}
```
- 场景摘录：PB entered into a definitive agreement for a Change of Control and exercised its right to make a Change of Control Buy-Out Payment to SFJ within the 120-day timeframe stipulated in evidence_4. However, instead of providing formal written notice via a tradition…
- 条款 `evidence_2`：Notwithstanding any of the foregoing, without the consent of PB, which consent may be withheld in PB's sole discretion, SFJ shall not sell, assign, sublicense or otherwise transfer…
- 条款 `evidence_4`：Within one hundred and twenty (120) days following the closing of a Change of Control, PB or its successor shall have the right to make a one-time payment (the "Change of Control B…
- 条款 `evidence_5`：PB shall not sell, transfer or assign, directly or indirectly, in whole or in part, any rights to receive payments of royalties or license fees with respect to the Product or the P…
### te00859（test，gold=Not-Applicable）
```json
{
 "sid": "te00859",
 "split": "test",
 "gold": "Not-Applicable",
 "B1": "Compliant",
 "B0": "Not-Applicable",
 "pruned": 17,
 "steps": 3,
 "expanded": 16,
 "source": "B1≠B0"
}
```
- 场景摘录：In 2023, Buyer Affiliates are obligated under the contract to purchase a minimum of 10,000 metric tons of Product from ExxonMobil Selling Affiliates as stipulated in the Multi-Conditional Obligation clause. To fulfill this obligation, Buyer Affiliates execute …
- 条款 `evidence_2`：Promptly after any public announcement regarding any proposed transaction that would result in a Change in Circumstances, Buyer/Buyer Affiliates shall notify Seller/ExxonMobil Sell…
- 条款 `evidence_6`：IN NO EVENT SHALL SELLER BE LIABLE FOR SPECIAL, INCIDENTAL, CONSEQUENTIAL OR EXEMPLARY DAMAGES. | Any cause of action that Buyer may have against Seller and which may arise in conn…
## mcts_wander
### tr00479（dev，gold=Compliant）
```json
{
 "sid": "tr00479",
 "split": "dev",
 "gold": "Compliant",
 "M0": "Not-Applicable",
 "A0": "Compliant",
 "expanded": 16
}
```
- 场景摘录：Vyera, seeking to restructure its business operations, assigns the Agreement to a newly created subsidiary, Vyera SubCo, which is wholly owned and operated by Vyera. Vyera SubCo is designed to exclusively handle all contractual obligations related to this Agre…
- 条款 `evidence_3`：CytoDyn may assign this Agreement and its rights and obligations hereunder, in whole but not in part, to any Third Party not in a materially worse (financially and otherwise) of pe…
- 条款 `evidence_10`：Each Party shall provide a certificate of insurance (or evidence of self-insurance) evidencing such coverage to the other Party upon reques | Each Party shall provide the other Par…
### tr00647（dev，gold=Compliant）
```json
{
 "sid": "tr00647",
 "split": "dev",
 "gold": "Compliant",
 "M0": "Not-Applicable",
 "A0": "Compliant",
 "expanded": 16
}
```
- 场景摘录：ACSI and Company are engaged in a collaborative project involving the development of a Joint Work. As part of their agreement, ACSI seeks to sublicense a portion of the Company Intellectual Property (CIP) to an affiliate, ACSI Digital Solutions (ADS), in order…
- 条款 `evidence_5`：registration of any Intellectual Property Rights associated with the Joint Works without the other Party's prior written consent; (b) make, manufacture, assemble, produce, market, …
- 条款 `evidence_6`：Company hereby grants to ACSI, during the Term, a non-exclusive, non-transferable (except in accordance with Section 11.7 [Assignment]) license, which ACSI may sublicense only to i…
- 条款 `evidence_7`：Company hereby grants to ACSI, during the Term, a non-exclusive, non-transferable (except in accordance with Section 11.7 [Assignment]) license, which ACSI may sublicense only to i…
### tr00715（dev，gold=Compliant）
```json
{
 "sid": "tr00715",
 "split": "dev",
 "gold": "Compliant",
 "M0": "Not-Applicable",
 "A0": "Compliant",
 "expanded": 16
}
```
- 场景摘录：The franchisee ('you'), pursuant to the transfer clause, intended to sell their franchise to a Proposed New Owner. To comply with the obligation requiring payment of all outstanding amounts owed to the franchisor before the transfer (Clause 1(b)), the franchis…
- 条款 `evidence_7`：If you propose to Transfer this Agreement, the Franchise or its assets, or any Interest, or if any of your Principal Owners proposes to Transfer a controlling Interest in you or ma…
- 条款 `evidence_9`：You agree not to use any Mark as part of any corporate or trade name or with any prefix, suffix, or other modifying words, terms, designs, or symbols (other than logos and addition…
- 条款 `evidence_14`：You must provide us with copies of policies evidencing the existence of such insurance concurrently with execution of this Agreement and prior to each subsequent renewal date of ea…
