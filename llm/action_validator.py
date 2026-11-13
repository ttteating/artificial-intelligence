import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from llm.llm_client import LLMClient
from state.interface import State, Action, Evaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ActionValidator(Evaluator):
    def __init__(self, client: LLMClient, clause_text_lookup: Optional[Dict[str, str]] = None):
        self.client = client
        self.clause_text_lookup = clause_text_lookup or {}

    def set_clause_lookup(self, lookup: Dict[str, str]):
        """Updates the clause text lookup dictionary."""
        self.clause_text_lookup = lookup

    def _get_clause_text(self, clause_id: str) -> str:
        return self.clause_text_lookup.get(clause_id, f"[Text for {clause_id} not found]")

    def action_valid(self, state: State, action: Action, question: str) -> Tuple[bool, float]:
        """
        Judges if a search action is valid using LLM.
        Returns (valid, confidence).
        """
        system_prompt = "你是合同合规推理的专家。你只判断搜索动作的合法性，不判断最终合规标签。"
        
        # Prepare context
        current_clauses_summary = "\n".join([f"- {cid}: {self._get_clause_text(cid)}" for cid in state.clauses])
        src_text = self._get_clause_text(action.src)
        dst_text = self._get_clause_text(action.dst)
        
        user_prompt = f"""
场景描述（Question）:
{question}

当前已审查条款集合:
{current_clauses_summary}

待验证动作（Action）:
- 起点条款 (Source): {action.src} ({src_text})
- 边类型 (Edge Type): {action.edge_type}
- 目标条款 (Destination): {action.dst} ({dst_text})

任务:
判断在合同合规推理过程中，从起点条款通过该类型的边扩展到目标条款是否具有逻辑合理性。
注意：
1. 只判断搜索路径的合法性，**严禁**输出最终的合规标签（如 Compliant, Non-Compliant, Not-Applicable）。
2. 给出你的判定（valid: true/false）、置信度（0.0-1.0）以及理由。

输出格式要求:
必须输出如下 JSON 格式：
{{
  "valid": bool,
  "confidence": float,
  "reason": "..."
}}
"""
        schema = {
            "valid": "bool",
            "confidence": "float",
            "reason": "str"
        }
        
        try:
            res = self.client.call_structured(user_prompt, schema, system=system_prompt)
            if "_parse_error" in res:
                logger.error(f"Action validation failed to parse: {res['raw']}")
                return False, 0.0
            
            valid = res.get("valid", False)
            confidence = res.get("confidence", 0.0)
            reason = res.get("reason", "")
            
            logger.info(f"Action Validation: {action.src} -> {action.dst} | Valid: {valid}, Conf: {confidence}, Reason: {reason}")
            
            return valid, confidence
            
        except Exception as e:
            logger.exception(f"Action validation error: {e}")
            return False, 0.0

    def batch_validate(self, state: State, actions: List[Action], question: str) -> List[Tuple[bool, float]]:
        """
        Validates multiple actions in a single LLM call.
        """
        if not actions:
            return []
            
        system_prompt = "你是合同合规推理的专家。你只判断搜索动作的合法性，不判断最终合规标签。"
        
        current_clauses_summary = "\n".join([f"- {cid}: {self._get_clause_text(cid)}" for cid in state.clauses])
        
        actions_desc = []
        for i, act in enumerate(actions):
            src_t = self._get_clause_text(act.src)
            dst_t = self._get_clause_text(act.dst)
            actions_desc.append({
                "id": i,
                "src": act.src,
                "src_text": src_t,
                "edge": act.edge_type,
                "dst": act.dst,
                "dst_text": dst_t
            })
            
        user_prompt = f"""
场景描述（Question）:
{question}

当前已审查条款集合:
{current_clauses_summary}

待验证动作列表:
{json.dumps(actions_desc, indent=2, ensure_ascii=False)}

任务:
对列表中的每个动作，判断其在合同合规推理中是否合理。
注意：
1. **严禁**输出合规标签（Compliant, Non-Compliant, Not-Applicable）。
2. 返回每个动作的判定结果。

输出格式要求:
必须输出如下 JSON 格式：
{{
  "results": [
    {{
      "id": int,
      "valid": bool,
      "confidence": float,
      "reason": "..."
    }},
    ...
  ]
}}
"""
        schema = {
            "results": "list"
        }
        
        try:
            res = self.client.call_structured(user_prompt, schema, system=system_prompt)
            if "_parse_error" in res:
                logger.warning("Batch validation parse error, falling back to individual calls.")
                return [self.action_valid(state, act, question) for act in actions]
            
            results = res.get("results", [])
            # Map back to original order
            mapped_results = [None] * len(actions)
            for r in results:
                idx = r.get("id")
                if idx is not None and 0 <= idx < len(actions):
                    mapped_results[idx] = (r.get("valid", False), r.get("confidence", 0.0))
            
            # Fill missing results with fallback
            for i in range(len(mapped_results)):
                if mapped_results[i] is None:
                    mapped_results[i] = self.action_valid(state, actions[i], question)
            
            return mapped_results
            
        except Exception as e:
            logger.warning(f"Batch validation error: {e}. Falling back to individual calls.")
            return [self.action_valid(state, act, question) for act in actions]

    # Implement remaining Evaluator methods as stubs
    def path_reward(self, path: List[Action], question: str) -> float:
        return 0.0
        
    def success_prob(self, state: State, question: str) -> float:
        return 0.5

if __name__ == "__main__":
    from llm.llm_client import LLMClient
    from pathlib import Path
    import shutil
    
    # Setup test cache
    test_cache_dir = Path("data/test_validator_cache")
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)
        
    # Mock lookup
    mock_clauses = {
        "evidence_1": "The Recipient shall keep the Information confidential.",
        "evidence_2": "Information includes any data shared under this Agreement.",
        "evidence_3": "The Recipient may share Information with its employees who have a need to know.",
        "evidence_4": "The Agreement terminates in 2 years.",
        "evidence_5": "The term 'Information' shall mean all technical data."
    }
    
    client = LLMClient(model="gpt-3.5-turbo", cache_dir=test_cache_dir)
    validator = ActionValidator(client, mock_clauses)
    
    question = "Is the Recipient allowed to share the data with a junior assistant?"
    state = State(clauses=frozenset(["evidence_1"]), used_edges=(), depth=0, label_hypothesis="")
    
    print("\n=== Action Validator Smoke Test ===")
    
    # 1. Obvious valid: evidence_1 (confidentiality) -> defines -> evidence_2 (definition)
    print("\nCase 1: Obvious Valid (cites/defines)")
    act1 = Action(src="evidence_1", edge_type="defines", dst="evidence_2")
    # Add hint for mock
    q1 = question + " (Case 1: Obvious Valid)"
    res1 = validator.action_valid(state, act1, q1)
    print(f"Result: {res1}")
    
    # 2. Obvious invalid: evidence_1 -> cites -> evidence_4 (termination, irrelevant to confidentiality)
    print("\nCase 2: Obvious Invalid (irrelevant link)")
    act2 = Action(src="evidence_1", edge_type="cites", dst="evidence_4")
    # Add hint for mock
    q2 = question + " (Case 2: Obvious Invalid)"
    res2 = validator.action_valid(state, act2, q2)
    print(f"Result: {res2}")
    
    # 3. Borderline: evidence_1 -> conditions_on -> evidence_3 (need to know condition)
    print("\nCase 3: Borderline (conditional expansion)")
    act3 = Action(src="evidence_1", edge_type="conditions_on", dst="evidence_3")
    # Add hint for mock
    q3 = question + " (Case 3: Borderline)"
    res3 = validator.action_valid(state, act3, q3)
    print(f"Result: {res3}")
    
    # 4. Batch validation test
    print("\nTesting Batch Validation...")
    actions = [act1, act2, act3]
    res_batch = validator.batch_validate(state, actions, question)
    print(f"Batch Result: {res_batch}")

    # 5. Cache Hit Test
    print("\nTesting Cache Hit (repeating Case 1)...")
    res1_cache = validator.action_valid(state, act1, q1)
    print(f"Result (should be same): {res1_cache}")
    
    stats = client.stats()
    print(f"\nStats: {stats}")
    if stats["cache_hits"] > 0:
        print("Success: Cache hit detected!")
    else:
        print("Failure: Cache hit not detected.")

    # Cleanup
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)
