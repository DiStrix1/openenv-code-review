from fastapi import FastAPI
from fastapi.responses import JSONResponse
import gradio as gr

from env.environment import CodeReviewEnv
from env.tasks import EasyTask, MediumTask, HardTask
from baseline.run_baseline import smart_agent

app = FastAPI()


# ─────────────────────────────────────────────
# /reset  — required by the OpenEnv validator
# ─────────────────────────────────────────────
@app.post("/reset")
def reset():
    """Reset the environment and return the initial observation."""
    env = CodeReviewEnv(EasyTask())
    obs = env.reset()
    return JSONResponse(status_code=200, content=obs.model_dump())


# ─────────────────────────────────────────────
# Gradio UI  — runs all tasks with baseline agent
# ─────────────────────────────────────────────
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
        results[task.__class__.__name__] = score

    return results


demo = gr.Interface(
    fn=run_all,
    inputs=[],
    outputs="json",
    title="Code Review OpenEnv Environment",
    description="Runs baseline agent on all tasks"
)

# Mount Gradio at /ui so it lives alongside the FastAPI routes
app = gr.mount_gradio_app(app, demo, path="/")
