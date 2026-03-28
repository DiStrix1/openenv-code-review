import gradio as gr

from env.environment import CodeReviewEnv
from env.tasks import EasyTask, MediumTask, HardTask
from baseline.run_baseline import smart_agent


def run_task(task):
    env = CodeReviewEnv(task)
    obs = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = smart_agent(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.value

    from env.grader import grade
    return grade(env.state())


def run_all():
    results = {}

    for task in [EasyTask(), MediumTask(), HardTask()]:
        score = run_task(task)
        results[task.__class__.__name__] = round(score, 2)

    return results


demo = gr.Interface(
    fn=run_all,
    inputs=[],
    outputs="json",
    title="Code Review OpenEnv Environment",
    description="Runs baseline agent on all tasks"
)


if __name__ == "__main__":
    demo.launch()