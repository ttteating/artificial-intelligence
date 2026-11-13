import json
import logging
from typing import Any, Dict, List, Optional

from llm.llm_client import LLMClient
from state.interface import Action

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CLAUSE_SUMMARY_MAX_CHARS = 200
BATCH_KEY = "rewards"

SYSTEM_PROMPT = "你是合同合规推理的评估器。你只评估推理路径的结构合理性，不判断最终合规标签。"

_SINGLE_TEMPLATE = """
场景描述（Question）:
{question}

推理路径（按扩展顺序，含条款摘要）:
{path_desc}

任务:
评估这条推理路径在多大程度上接近一个可判定的合规结论。判断要点：
1. 路径是否已经触及关键条款：例外条款（exception_of）、条件条款（conditions_on / cites）、或义务判定点（defines 出义务主体与范围的条款）。
2. 路径中是否出现结构上不合理的跳跃（例如在语义无关的条款之间扩展）。
3. 路径累积的信息是否足以支撑下一步判定。

明确禁止：
- 严禁输出合规标签（Compliant / Non-Compliant / Not-Applicable）。
- 严禁输出"答案应该是X"之类的最终结论。
- 只评估路径结构合理性，不判定合规与否。

输出格式要求:
必须输出如下 JSON 格式（reward 为 0.0~1.0 的浮点数，越大表示越接近可判定点）：
{{
  "reward": float,
  "touched_key_clause": bool,
  "brief_reason": str
}}
"""

_BATCH_TEMPLATE = """
场景描述（Question）:
{question}

待评估路径列表（按扩展顺序，含条款摘要）:
{paths_desc}

任务:
对列表中的每条路径，评估其在多大程度上接近一个可判定的合规结论。判断要点：
1. 路径是否已经触及关键条款：例外条款（exception_of）、条件条款（conditions_on / cites）、或义务判定点（defines 出义务主体与范围的条款）。
2. 路径中是否出现结构上不合理的跳跃（例如在语义无关的条款之间扩展）。
3. 路径累积的信息是否足以支撑下一步判定。

明确禁止：
- 严禁输出合规标签（Compliant / Non-Compliant / Not-Applicable）。
- 严禁输出"答案应该是X"之类的最终结论。
- 只评估路径结构合理性，不判定合规与否。

输出格式要求:
必须输出如下 JSON 格式（每条路径一项，reward 为 0.0~1.0 的浮点数）：
{{
  "{batch_key}": [
    {{
      "id": int,
      "reward": float,
      "touched_key_clause": bool,
      "brief_reason": str
    }},
    ...
  ]
}}
"""


class PathReward:
    """
    Evaluates how close a reasoning path is to a decidable compliance conclusion.

    Implements the Evaluator protocol's path_reward method (duck-typed).
    Only returns a scalar reward in [0, 1]; no truncation/aggregation is done here
    (that is the MCTS side's responsibility).
    """

    def __init__(self, client: LLMClient, clause_text_lookup: Optional[Dict[str, str]] = None):
        self.client = client
        self.clause_text_lookup = clause_text_lookup or {}

    def set_clause_lookup(self, lookup: Dict[str, str]) -> None:
        """Updates the clause text lookup dictionary."""
        self.clause_text_lookup = lookup

    def _summarize_clause(self, clause_id: str) -> str:
        """Returns a truncated summary of the clause text for prompt building."""
        text = self.clause_text_lookup.get(clause_id)
        if not text:
            return f"[条款 {clause_id} 文本缺失]"
        text = " ".join(text.split())
        if len(text) > CLAUSE_SUMMARY_MAX_CHARS:
            return text[:CLAUSE_SUMMARY_MAX_CHARS] + "..."
        return text

    def _describe_path(self, path: List[Action]) -> str:
        lines = []
        for i, act in enumerate(path):
            lines.append(
                f"{i + 1}. {act.src} (摘要: {self._summarize_clause(act.src)})"
                f" --[{act.edge_type}]--> "
                f"{act.dst} (摘要: {self._summarize_clause(act.dst)})"
            )
        return "\n".join(lines) if lines else "(空路径)"

    def _parse_single(self, res: Dict[str, Any]) -> Optional[float]:
        if "_parse_error" in res:
            return None
        try:
            reward = float(res.get("reward", 0.0))
        except (TypeError, ValueError):
            return None
        touched = res.get("touched_key_clause", False)
        reason = res.get("brief_reason", "")
        logger.info(f"PathReward: reward={reward:.3f}, touched_key_clause={touched}, reason={reason}")
        return reward

    def path_reward(self, path: List[Action], question: str) -> float:
        """
        Judges how close a (partial or complete) path is to a decidable
        compliance conclusion. Returns a scalar reward in [0, 1].

        Never returns a label; the LLM is explicitly forbidden from emitting one.
        """
        if not path:
            logger.warning("path_reward called with an empty path; returning 0.0 without an LLM call.")
            return 0.0

        user_prompt = _SINGLE_TEMPLATE.format(
            question=question,
            path_desc=self._describe_path(path),
        )

        schema = {
            "reward": "float",
            "touched_key_clause": "bool",
            "brief_reason": "str",
        }

        try:
            res = self.client.call_structured(user_prompt, schema, system=SYSTEM_PROMPT)
        except Exception as e:
            logger.exception(f"path_reward LLM call failed: {e}")
            return 0.0

        reward = self._parse_single(res)
        if reward is None:
            logger.error(f"path_reward failed to parse response: {res.get('raw', res)}")
            return 0.0
        return reward

    def batch_reward(self, paths: List[List[Action]], question: str) -> List[float]:
        """
        Evaluates multiple rollout paths (e.g. MCTS rollouts) in a single LLM call.
        Falls back to per-path calls on parse failure.
        """
        if not paths:
            return []

        # Fast path: all empty paths need no LLM call.
        if all(len(p) == 0 for p in paths):
            return [0.0] * len(paths)

        paths_desc = []
        for i, path in enumerate(paths):
            paths_desc.append({
                "id": i,
                "steps": self._describe_path(path).split("\n"),
            })

        user_prompt = _BATCH_TEMPLATE.format(
            question=question,
            paths_desc=json.dumps(paths_desc, indent=2, ensure_ascii=False),
            batch_key=BATCH_KEY,
        )

        schema = {BATCH_KEY: "list"}

        try:
            res = self.client.call_structured(user_prompt, schema, system=SYSTEM_PROMPT)
        except Exception as e:
            logger.warning(f"batch_reward LLM call failed: {e}. Falling back to individual calls.")
            return [self.path_reward(p, question) for p in paths]

        if "_parse_error" in res:
            logger.warning("batch_reward parse error, falling back to individual calls.")
            return [self.path_reward(p, question) for p in paths]

        entries = res.get(BATCH_KEY, [])
        mapped: List[Optional[float]] = [None] * len(paths)
        for entry in entries:
            try:
                idx = int(entry.get("id"))
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(paths):
                try:
                    reward = float(entry.get("reward", 0.0))
                except (TypeError, ValueError):
                    continue
                touched = entry.get("touched_key_clause", False)
                reason = entry.get("brief_reason", "")
                logger.info(
                    f"Batch PathReward [{idx}]: reward={reward:.3f}, "
                    f"touched_key_clause={touched}, reason={reason}"
                )
                mapped[idx] = reward

        # Fallback to individual calls for paths the batch response missed.
        results: List[float] = []
        for i, path in enumerate(paths):
            if mapped[i] is None:
                logger.warning(f"batch_reward missing result for path {i}, falling back to individual call.")
                results.append(self.path_reward(path, question))
            else:
                results.append(mapped[i])
        return results


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    # Setup test cache
    test_cache_dir = Path("data/test_path_reward_cache")
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)

    mock_clauses = {
        "evidence_1": "The Recipient shall keep the Information confidential and shall not disclose it to any third party.",
        "evidence_2": "Information means any technical, financial or business data disclosed under this Agreement.",
        "evidence_3": "Notwithstanding the foregoing, the Recipient may disclose Information to its employees who have a need to know.",
        "evidence_4": "This Agreement terminates two years after the Effective Date.",
        "evidence_5": "Provided that the Recipient implements reasonable security measures, the confidentiality obligations shall apply.",
    }

    client = LLMClient(model="gpt-3.5-turbo", cache_dir=test_cache_dir)
    reward_fn = PathReward(client, mock_clauses)

    question = "Is the Recipient allowed to share the data with a junior assistant?"

    # Case 1: clearly approaching a decision point (exception + condition)
    path_close = [
        Action(src="evidence_1", edge_type="exception_of", dst="evidence_3"),
        Action(src="evidence_3", edge_type="conditions_on", dst="evidence_5"),
    ]
    # Case 2: clearly off-track (termination clause, irrelevant to the question)
    path_off = [
        Action(src="evidence_4", edge_type="same_section", dst="evidence_2"),
    ]
    # Case 3: partially touching (definition reached, but no exception/condition yet)
    path_partial = [
        Action(src="evidence_1", edge_type="defines", dst="evidence_2"),
    ]

    print("\n=== PathReward Smoke Test ===")

    print("\nCase 1: 明显接近判定点 (exception_of + conditions_on)")
    q1 = question + " (Case 1: 接近判定点)"
    r1 = reward_fn.path_reward(path_close, q1)
    print(f"Reward: {r1:.3f}")

    print("\nCase 2: 明显偏离 (irrelevant termination clause)")
    q2 = question + " (Case 2: 明显偏离)"
    r2 = reward_fn.path_reward(path_off, q2)
    print(f"Reward: {r2:.3f}")

    print("\nCase 3: 部分触及 (definition only)")
    q3 = question + " (Case 3: 部分触及)"
    r3 = reward_fn.path_reward(path_partial, q3)
    print(f"Reward: {r3:.3f}")

    if r1 > r3 > r2:
        print("Success: Reward ordering is reasonable (接近判定点 > 部分触及 > 偏离).")
    else:
        print(f"Failure: Unexpected ordering: r1={r1:.3f}, r3={r3:.3f}, r2={r2:.3f}")

    # Batch interface test
    print("\nTesting batch_reward (3 paths in one call)...")
    batch_q = question + " (batch)"
    batch_results = reward_fn.batch_reward([path_close, path_off, path_partial], batch_q)
    print(f"Batch rewards: {[f'{r:.3f}' for r in batch_results]}")

    # Empty path test (no LLM call expected)
    print("\nTesting empty path (should return 0.0 without LLM call)...")
    r_empty = reward_fn.path_reward([], question)
    print(f"Empty path reward: {r_empty}")

    # Cache hit test: repeat Case 1
    print("\nTesting cache (repeating Case 1)...")
    calls_before = client.stats()["total_calls"]
    r1_repeat = reward_fn.path_reward(path_close, q1)
    calls_after = client.stats()["total_calls"]
    cache_hits = client.stats()["cache_hits"]
    print(f"Repeat reward (should be {r1:.3f}): {r1_repeat:.3f}")
    print(f"total_calls before/after: {calls_before}/{calls_after}, cache_hits: {cache_hits}")
    if r1_repeat == r1 and cache_hits > 0:
        print("Success: Cache hit works.")
    else:
        print("Failure: Cache hit not detected.")

    stats = client.stats()
    print(f"\nClient stats: {stats}")

    # Cleanup
    if test_cache_dir.exists():
        shutil.rmtree(test_cache_dir)
