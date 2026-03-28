from env.environment import CodeReviewEnv
from env.tasks import EasyTask, MediumTask, HardTask
from env.models import Action
from env.grader import grade
import random

SEED = 42

def smart_agent(observation):
    import random

    diff = observation.diff.lower()

    # detect null issues
    if "null" in diff or "none" in diff:
        return Action(action_type="comment", issue_id=2, severity="high")

    # detect loops (NOT always correct now)
    if diff.count("for") > 1:
        if random.random() < 0.6:
            return Action(action_type="comment", issue_id=1, severity="medium")
        else:
            return Action(action_type="request_changes")

    # detect assignment bug (less reliable)
    if "=" in diff and "==" not in diff and "if" in diff:
        if random.random() < 0.6:
            return Action(action_type="comment", issue_id=1, severity="high")

    # fallback exploration
    return Action(
        action_type="comment",
        issue_id=random.choice([1, 2, 3]),
        severity="low"
    )
def run_task(task):
    env = CodeReviewEnv(task)
    obs = env.reset()
    done = False
    while not done:
        action = smart_agent(obs)
        obs,reward, done, _ = env.step(action)
    final_state = env.state()
    return grade(final_state)


def run_benchmark(seed=SEED):
    random.seed(seed)
    tasks = [EasyTask(), MediumTask(), HardTask()]
    scores = {}
    for task in tasks:
        scores[task.__class__.__name__] = run_task(task)

    easy_score = scores["EasyTask"]
    medium_score = scores["MediumTask"]
    hard_score = scores["HardTask"]
    if not (easy_score > medium_score > hard_score):
        print(
            "WARNING: Difficulty calibration failed: expected EasyTask > MediumTask > HardTask, "
            f"got EasyTask={easy_score}, MediumTask={medium_score}, HardTask={hard_score}"
        )

    return scores


if __name__ == "__main__":
    benchmark_scores = run_benchmark()
    for task_name in ["EasyTask", "MediumTask", "HardTask"]:
        print(f"{task_name}: {benchmark_scores[task_name]}")