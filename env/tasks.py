import copy
from typing import Any, Dict, List


class BaseTask:
    difficulty = "easy"

    def __init__(self) -> None:
        self.code = self._initial_code()
        self.issues = self._ground_truth_issues()

    def _initial_code(self) -> str:
        raise NotImplementedError

    def _ground_truth_issues(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def initial_state(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "issues": copy.deepcopy(self.issues),
            "difficulty": self.difficulty,
        }


class EasyTask(BaseTask):
    difficulty = "easy"

    def _initial_code(self) -> str:
        return (
            "def sum_items(items):\n"
            "    total = 0\n"
            "    for i in range(len(items)):\n"
            "        total += items[i]\n"
            "    return total\n"
        )

    def _ground_truth_issues(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "E1",
                "description": "Loop style can be simplified to direct iteration.",
                "severity": "low",
                "type": "style",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
            },
            {
                "id": "E2",
                "description": "No guard for None input may raise TypeError.",
                "severity": "medium",
                "type": "logic",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
            },
        ]


class MediumTask(BaseTask):
    difficulty = "medium"

    def _initial_code(self) -> str:
        return (
            "def find_user(users, target_id):\n"
            "    for user in users:\n"
            "        if user['id'] == target_id:\n"
            "            return user\n"
            "    return {}\n"
            "\n"
            "def compute_rate(total, count):\n"
            "    return total / count\n"
        )

    def _ground_truth_issues(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "M1",
                "description": "Returning mutable empty dict may hide missing-user state.",
                "severity": "medium",
                "type": "logic",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "context_hint": "ambiguous API contract",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "M2",
                "description": "Potential division by zero in compute_rate.",
                "severity": "high",
                "type": "security",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "requires": ["M1"],
                "context_hint": "input validation dependency",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "M3",
                "description": "Linear scan should be indexed for repeated lookups.",
                "severity": "medium",
                "type": "performance",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "context_hint": "hot path under repeated access",
                "actionable_with": ["optimize_code", "suggest_fix"],
            },
            {
                "id": "M4",
                "description": "find_user assumes every record has key 'id', leading to KeyError risk.",
                "severity": "medium",
                "type": "logic",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "context_hint": "schema inconsistency edge case",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "M5",
                "description": "No normalization on target_id causes inconsistent matching behavior.",
                "severity": "medium",
                "type": "logic",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "requires": ["M4"],
                "context_hint": "cross-path id format mismatch",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "M6",
                "description": "Raw dictionary access without schema checks can raise runtime exceptions.",
                "severity": "medium",
                "type": "security",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "requires": ["M4"],
                "context_hint": "validation order matters",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
        ]


class HardTask(BaseTask):
    difficulty = "hard"

    def _initial_code(self) -> str:
        return (
            "def execute_query(conn, user_input, table=\"users\"):\n"
            "    safe = user_input.replace(\";\", \"\").strip()\n"
            "    # misleading: this looks safe but quote payloads still pass through\n"
            "    query = \"SELECT * FROM \" + table + \" WHERE name = '\" + safe + \"'\"\n"
            "    return conn.run(query)\n"
            "\n"
            "def transform(items, cache={}, retries=2):\n"
            "    # noisy signal: retry and cache logic hides shared mutable state\n"
            "    if not items:\n"
            "        return []\n"
            "    key = tuple(items)\n"
            "    if key in cache:\n"
            "        return cache[key]\n"
            "    out = []\n"
            "    for item in items:\n"
            "        try:\n"
            "            out.append(int(item) * 2)\n"
            "        except Exception:\n"
            "            # misleading: keeps flow smooth while masking data errors\n"
            "            out.append(0)\n"
            "    if retries > 0:\n"
            "        cache[key] = out\n"
            "    return out\n"
            "\n"
            "def verify_token(expected, provided):\n"
            "    # misleading: normalization appears defensive for secrets\n"
            "    return expected.lower() == provided.lower()\n"
            "\n"
            "def dedupe_records(records):\n"
            "    # noisy signal: correctness is fine but complexity is quadratic\n"
            "    unique = []\n"
            "    for record in records:\n"
            "        exists = False\n"
            "        for saved in unique:\n"
            "            if saved['id'] == record['id']:\n"
            "                exists = True\n"
            "                break\n"
            "        if not exists:\n"
            "            unique.append(record)\n"
            "    return unique\n"
            "\n"
            "def maybe_log(debug, msg, pii):\n"
            "    if debug and msg:\n"
            "        print(msg + ':' + pii)\n"
        )

    def _ground_truth_issues(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "H1",
                "description": "SQL injection risk due to string concatenation query build.",
                "severity": "high",
                "type": "security",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "misleading_pattern": "partial sanitization comment",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "H2",
                "description": "Mutable default argument in transform(cache={}).",
                "severity": "high",
                "type": "logic",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "misleading_pattern": "cache appears intentional optimization",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "H3",
                "description": "Broad exception handling hides conversion defects.",
                "severity": "medium",
                "type": "logic",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "noisy_signal": True,
                "requires": ["H2"],
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "H4",
                "description": "Case-insensitive token compare is unsafe for secret validation.",
                "severity": "high",
                "type": "security",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "misleading_pattern": "appears user-friendly but weakens auth checks",
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "H5",
                "description": "Quadratic dedupe implementation causes performance degradation.",
                "severity": "medium",
                "type": "performance",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "noisy_signal": True,
                "actionable_with": ["optimize_code", "suggest_fix"],
            },
            {
                "id": "H6",
                "description": "Debug logging leaks PII through combined message output.",
                "severity": "high",
                "type": "security",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "noisy_signal": True,
                "requires": ["H4"],
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "H7",
                "description": "Dynamic table concatenation expands unsafe query surface area.",
                "severity": "high",
                "type": "security",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "misleading_pattern": "table parameter looks configurable but bypasses query safety",
                "requires": ["H1"],
                "actionable_with": ["flag_issue", "suggest_fix"],
            },
            {
                "id": "H8",
                "description": "Repeated string concatenation in dedupe path causes avoidable allocation overhead.",
                "severity": "medium",
                "type": "performance",
                "resolved": False,
                "flagged": False,
                "source": "ground_truth",
                "noisy_signal": True,
                "requires": ["H5"],
                "actionable_with": ["optimize_code", "suggest_fix"],
            },
        ]
