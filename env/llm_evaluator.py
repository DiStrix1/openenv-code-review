from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


class LLMEvaluator:
    def __init__(self, difficulty: str, mode: Optional[str] = None) -> None:
        requested_mode = (mode or os.getenv("CODE_REVIEW_LLM_EVAL_MODE", "heuristic")).strip().lower()
        if requested_mode not in {"off", "heuristic", "openai"}:
            requested_mode = "heuristic"

        remote_opt_in = os.getenv("CODE_REVIEW_ENABLE_REMOTE_LLM", "0").strip()
        self._remote_llm_enabled = remote_opt_in == "1"

        self.difficulty = difficulty
        self.mode = requested_mode
        self.model = os.getenv("CODE_REVIEW_LLM_MODEL", "gpt-4o-mini")
        self._client = None
        self._mode_reason = "configured"

        if self.mode == "openai" and not self._remote_llm_enabled:
            self.mode = "heuristic"
            self._mode_reason = "remote_llm_opt_in_not_enabled"

        if self.mode == "openai":
            self._initialize_client()

    def reset(self, difficulty: str) -> None:
        self.difficulty = difficulty

    def _initialize_client(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.mode = "heuristic"
            self._mode_reason = "missing_openai_api_key"
            return
        try:
            from openai import OpenAI
        except Exception:
            self.mode = "heuristic"
            self._mode_reason = "openai_package_unavailable"
            return

        self._client = OpenAI(api_key=api_key)
        self._mode_reason = "openai_enabled"

    def _heuristic_raw_score(
        self,
        action: str,
        action_effect: Dict[str, Any],
        developer_result: Dict[str, Any],
        remaining_before: int,
        remaining_after: int,
        step_count: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        score = 0.5
        tags: List[str] = []

        if action_effect.get("correct_detection", False):
            score += 0.12
            tags.append("correct_detection")
        if action_effect.get("high_severity", False):
            score += 0.10
            tags.append("high_severity_focus")
        if developer_result.get("fix_applied", False):
            if developer_result.get("fixed_issue_source") == "ground_truth":
                score += 0.18
                tags.append("ground_truth_fix")
            else:
                score += 0.08
                tags.append("introduced_bug_fix")
        if developer_result.get("fix_failed", False):
            score -= 0.10
            tags.append("fix_failure")
        if developer_result.get("introduced_bug_id") is not None:
            score -= 0.20
            tags.append("regression_introduced")
        if action_effect.get("false_positive", False):
            score -= 0.20
            tags.append("false_positive")
        if action_effect.get("unnecessary", False):
            score -= 0.15
            tags.append("unnecessary_action")
        if action_effect.get("repeated", False):
            score -= 0.06
            tags.append("repeated_action")
        if action_effect.get("repeat_target", False):
            score -= 0.08
            tags.append("repeat_target")

        if remaining_after < remaining_before:
            score += 0.10
            tags.append("issue_count_reduced")
        if remaining_after == 0:
            score += 0.20
            tags.append("complete_resolution")

        if step_count > max_steps // 2:
            score -= 0.03 * float(step_count - (max_steps // 2))
            tags.append("late_step_penalty")

        if self.difficulty == "hard":
            score -= 0.03
        elif self.difficulty == "easy":
            score += 0.02

        raw_score = float(_clamp(score, 0.0, 1.0))
        confidence = 0.70
        if tags:
            confidence += min(len(tags) * 0.03, 0.20)
        confidence = float(_clamp(confidence, 0.5, 0.95))

        rationale = " | ".join(tags[:4]) if tags else f"neutral_{action}"
        return {
            "raw_score": round(raw_score, 3),
            "confidence": round(confidence, 3),
            "rationale": rationale,
        }

    def _openai_raw_score(
        self,
        action: str,
        action_effect: Dict[str, Any],
        developer_result: Dict[str, Any],
        remaining_before: int,
        remaining_after: int,
        step_count: int,
        max_steps: int,
        total_issues: int,
    ) -> Optional[Dict[str, Any]]:
        if self._client is None:
            return None

        payload = {
            "difficulty": self.difficulty,
            "action": action,
            "action_effect": action_effect,
            "developer_result": developer_result,
            "remaining_before": remaining_before,
            "remaining_after": remaining_after,
            "step_count": step_count,
            "max_steps": max_steps,
            "total_issues": total_issues,
        }
        system_prompt = (
            "Evaluate one RL transition and return strict JSON with keys: "
            "raw_score (0..1), confidence (0..1), rationale."
        )
        user_prompt = json.dumps(payload, ensure_ascii=True)

        try:
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_content = completion.choices[0].message.content
            if raw_content is None:
                return None
            parsed = json.loads(raw_content)
            raw_score = float(_clamp(float(parsed.get("raw_score", 0.5)), 0.0, 1.0))
            confidence = float(_clamp(float(parsed.get("confidence", 0.7)), 0.0, 1.0))
            rationale = str(parsed.get("rationale", "llm_eval"))
            return {
                "raw_score": round(raw_score, 3),
                "confidence": round(confidence, 3),
                "rationale": rationale[:160],
            }
        except Exception:
            self.mode = "heuristic"
            self._client = None
            self._mode_reason = "openai_runtime_fallback"
            return None

    def _aligned_score(
        self,
        raw_score: float,
        action_effect: Dict[str, Any],
        developer_result: Dict[str, Any],
        remaining_before: int,
        remaining_after: int,
        total_issues: int,
        step_count: int,
        max_steps: int,
    ) -> Dict[str, Any]:
        total = max(total_issues, 1)
        completion_ratio = float(_clamp((total - remaining_after) / float(total), 0.0, 1.0))

        transition_delta = remaining_before - remaining_after
        if transition_delta > 0:
            transition_quality = 1.0
        elif transition_delta == 0:
            transition_quality = 0.5
        else:
            transition_quality = 0.0

        efficiency_ratio = 1.0 - (step_count / float(max(max_steps, 1)))
        efficiency_ratio = float(_clamp(efficiency_ratio, 0.0, 1.0))

        penalty = 0.0
        if action_effect.get("false_positive", False):
            penalty += 0.12
        if action_effect.get("unnecessary", False):
            penalty += 0.08
        if developer_result.get("introduced_bug_id") is not None:
            penalty += 0.10

        aligned = (
            (0.45 * completion_ratio)
            + (0.25 * transition_quality)
            + (0.20 * float(raw_score))
            + (0.10 * efficiency_ratio)
            - penalty
        )
        aligned = float(_clamp(aligned, 0.0, 1.0))
        return {
            "aligned_score": round(aligned, 3),
            "completion_ratio": round(completion_ratio, 3),
            "transition_quality": round(transition_quality, 3),
            "efficiency_ratio": round(efficiency_ratio, 3),
            "penalty": round(penalty, 3),
        }

    def evaluate(
        self,
        action: str,
        action_effect: Dict[str, Any],
        developer_result: Dict[str, Any],
        remaining_before: int,
        remaining_after: int,
        step_count: int,
        max_steps: int,
        total_issues: int,
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        _ = history
        if self.mode == "off":
            return {
                "mode": "off",
                "enabled": False,
                "strategic_score": 0.0,
                "raw_strategic_score": 0.0,
                "confidence": 0.0,
                "rationale": "llm_evaluation_disabled",
                "mode_reason": "explicitly_disabled",
                "fallback_used": False,
                "alignment": {
                    "completion_ratio": 0.0,
                    "transition_quality": 0.0,
                    "efficiency_ratio": 0.0,
                    "penalty": 0.0,
                },
            }

        heuristic_eval = self._heuristic_raw_score(
            action=action,
            action_effect=action_effect,
            developer_result=developer_result,
            remaining_before=remaining_before,
            remaining_after=remaining_after,
            step_count=step_count,
            max_steps=max_steps,
        )

        raw_score = float(heuristic_eval["raw_score"])
        confidence = float(heuristic_eval["confidence"])
        rationale = str(heuristic_eval["rationale"])
        fallback_used = False
        mode_used = "heuristic"

        if self.mode == "openai":
            openai_eval = self._openai_raw_score(
                action=action,
                action_effect=action_effect,
                developer_result=developer_result,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                step_count=step_count,
                max_steps=max_steps,
                total_issues=total_issues,
            )
            if openai_eval is None:
                fallback_used = True
            else:
                mode_used = "openai"
                raw_score = float(
                    _clamp(
                        (0.60 * float(openai_eval["raw_score"])) + (0.40 * raw_score),
                        0.0,
                        1.0,
                    )
                )
                confidence = float(
                    _clamp(
                        (0.70 * float(openai_eval["confidence"])) + (0.30 * confidence),
                        0.0,
                        1.0,
                    )
                )
                rationale = str(openai_eval["rationale"])

        alignment = self._aligned_score(
            raw_score=raw_score,
            action_effect=action_effect,
            developer_result=developer_result,
            remaining_before=remaining_before,
            remaining_after=remaining_after,
            total_issues=total_issues,
            step_count=step_count,
            max_steps=max_steps,
        )

        return {
            "mode": mode_used,
            "enabled": True,
            "strategic_score": alignment["aligned_score"],
            "raw_strategic_score": round(raw_score, 3),
            "confidence": round(confidence, 3),
            "rationale": rationale,
            "mode_reason": self._mode_reason,
            "fallback_used": fallback_used,
            "alignment": {
                "completion_ratio": alignment["completion_ratio"],
                "transition_quality": alignment["transition_quality"],
                "efficiency_ratio": alignment["efficiency_ratio"],
                "penalty": alignment["penalty"],
            },
        }
