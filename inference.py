import os
from typing import Optional

from openai import OpenAI

from env.environment import CodeReviewEnv
from env.grader import grade as grader
from env.models import Action
from env.tasks import EasyTask, MediumTask, HardTask


def get_api_key() -> str:
    return os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""


def build_client() -> Optional[OpenAI]:
    api_key = get_api_key()
    base_url = os.getenv("API_BASE_URL")

    if not api_key:
        return None

    try:
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
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


def run_task(task_cls, client: Optional[OpenAI], model_name: str, max_steps: int = 5) -> float:
    env = CodeReviewEnv(task=task_cls())
    observation = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        prompt = build_prompt(
            diff=observation.diff,
            comments=observation.comments,
            iteration=observation.iteration,
        )
        llm_response = query_llm(client=client, model_name=model_name, prompt=prompt)
        action = parse_action(llm_response)

        observation, reward, done, _ = env.step(action)
        total_reward += reward.value

        if done:
            break

    score = grader(env.state())
    return score


def main() -> None:
    client = build_client()
    model_name = os.getenv("MODEL_NAME", "")

    easy_score = run_task(EasyTask, client, model_name)
    medium_score = run_task(MediumTask, client, model_name)
    hard_score = run_task(HardTask, client, model_name)

    print(f"EasyTask: {easy_score}")
    print(f"MediumTask: {medium_score}")
    print(f"HardTask: {hard_score}")


if __name__ == "__main__":
    main()
