import os
from typing import Optional

from openai import OpenAI

from env.environment import CodeReviewEnv
from env.grader import grade as grader
from env.models import Action
from env.tasks import EasyTask, MediumTask, HardTask

# Required env vars for evaluator compatibility.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional - if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


def get_api_key() -> str:
    return HF_TOKEN or ""


def build_client() -> Optional[OpenAI]:
    api_key = get_api_key()
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key, base_url=API_BASE_URL)
    except Exception:
        return None


def build_prompt(diff: str, comments: list[str], iteration: int) -> str:
    comments_text = "\n".join(comments) if comments else "None"
    return (
        "You are reviewing a pull request in a code review environment.\n"
        f"Iteration: {iteration}\n\n"
        "Diff:\n"
        f"{diff}\n\n"
        "Previous comments:\n"
        f"{comments_text}\n\n"
        "Choose exactly ONE next action from:\n"
        "comment(issue_id)\n"
        "approve\n"
        "request_changes\n\n"
        "Return ONLY the action text."
    )


def query_llm(client: Optional[OpenAI], model_name: str, prompt: str) -> str:
    if client is None or not model_name:
        return "request_changes"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return only one action."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        return content.strip()
    except Exception:
        # Fallback behavior when API call fails.
        return "request_changes"


def parse_action(response_text: str) -> Action:
    text = (response_text or "").strip().lower()

    if "approve" in text:
        return Action(action_type="approve")

    if "request" in text:
        return Action(action_type="request_changes")

    return Action(action_type="comment", issue_id=1)


def run_task(
    task_cls,
    task_id: str,
    client: Optional[OpenAI],
    model_name: str,
    max_steps: int = 5,
) -> float:
    print(f"[START] task={task_id}", flush=True)

    env = CodeReviewEnv(task=task_cls())
    observation = env.reset()
    step = 0

    for step in range(1, max_steps + 1):
        prompt = build_prompt(
            diff=observation.diff,
            comments=observation.comments,
            iteration=observation.iteration,
        )
        llm_response = query_llm(client=client, model_name=model_name, prompt=prompt)
        action = parse_action(llm_response)

        observation, reward, done, _ = env.step(action)
        print(f"[STEP] step={step} reward={reward.value:.4f}", flush=True)

        if done:
            break

    score = grader(env.state())
    print(f"[END] task={task_id} score={score:.4f} steps={step}", flush=True)
    return score


def main() -> None:
    client = build_client()
    model_name = MODEL_NAME

    easy_score = run_task(EasyTask, "easy", client, model_name)
    medium_score = run_task(MediumTask, "medium", client, model_name)
    hard_score = run_task(HardTask, "hard", client, model_name)

    print(f"EasyTask: {easy_score}")
    print(f"MediumTask: {medium_score}")
    print(f"HardTask: {hard_score}")


if __name__ == "__main__":
    main()
