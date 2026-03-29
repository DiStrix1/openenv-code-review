from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gradio as gr

from baseline.run_baseline import run_episode


def run_baseline_suite(max_steps: int) -> Tuple[List[List[Any]], Dict[str, Any]]:
    rows: List[List[Any]] = []
    summary: Dict[str, Any] = {
        "max_steps": max_steps,
        "results": {},
    }
    for task_name in ("easy", "medium", "hard"):
        result = run_episode(task_name, max_steps=max_steps)
        rows.append(
            [
                result["task"],
                result["steps"],
                round(result["total_reward"], 3),
                round(result["final_grade"], 3),
                result["remaining_issues"],
                result["remaining_total_issues"],
                round(result["avg_llm_score"], 3),
                result["llm_mode"],
            ]
        )
        summary["results"][task_name] = result
    return rows, summary


with gr.Blocks(title="Code Review Simulation Environment") as demo:
    gr.Markdown("# Code Review Simulation Environment")
    gr.Markdown(
        "Run deterministic baseline rollouts for Easy, Medium, and Hard tasks."
    )

    max_steps = gr.Slider(
        minimum=4,
        maximum=20,
        value=10,
        step=1,
        label="Requested Max Steps",
    )
    run_btn = gr.Button("Run Baseline")
    results_table = gr.Dataframe(
        headers=[
            "task",
            "steps",
            "total_reward",
            "final_grade",
            "remaining_ground_truth_issues",
            "remaining_total_issues",
            "avg_llm_score",
            "llm_mode",
        ],
        value=[],
        interactive=False,
        wrap=True,
    )
    raw_json = gr.JSON(label="Raw Results")

    run_btn.click(
        fn=run_baseline_suite,
        inputs=[max_steps],
        outputs=[results_table, raw_json],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
