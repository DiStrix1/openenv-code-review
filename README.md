# OpenEnv Code Review Environment

This project simulates a multi-step code review loop where an automated reviewer
comments on issues in a pull request and a developer simulator may fix issues or
introduce new bugs over iterations.

## Environment Spaces

Observation space:

- `diff` (`string`): pull-request code diff visible to the agent.
- `comments` (`list[string]`): accumulated review comments (for example `ISSUE_1`).
- `iteration` (`int`): current interaction step in the episode.

Action space:

- `action_type` (`string`): one of `comment`, `approve`, `request_changes`.
- `issue_id` (`int | null`): target issue identifier when applicable.
- `severity` (`string | null`): optional level in `low`, `medium`, `high`.

Reward space:

- continuous scalar with configured range `[-5.0, 1.5]`

## Project Structure

- `env/`: Environment core, task definitions, simulator, models, and grading.
- `baseline/`: Baseline heuristic agent runner.
- `openenv.yaml`: High-level environment specification.
- `app.py`: Gradio entrypoint for Hugging Face Spaces.

## How It Works

1. A task generates a pull request (`diff` + labeled issues).
2. The agent chooses an action (`comment`, `approve`, `request_changes`).
3. The environment computes step reward and updates state.
4. The developer simulator probabilistically fixes flagged issues.
5. The episode ends after approval with no unresolved issues, or after 5 turns.

## Task Definitions

- Easy (`easy`): clear high-signal issues in a short diff; objective is to catch obvious correctness and safety defects.
- Medium (`medium`): fewer obvious patterns and more ambiguity; objective is to identify performance and safety issues in noisier code.
- Hard (`hard`): multiple interacting issues with mixed severity; objective is to balance correctness, performance, and safety under uncertainty.

The developer simulator uses task-specific behavior profiles so harder tasks are less responsive and can introduce more severe bugs.

## Setup

Prerequisites:

- Python 3.10+

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Run a single baseline benchmark locally:

```bash
python -m baseline.run_baseline
```

## Run Baseline

```bash
python -m baseline.run_baseline
```

Expected output format:

```text
EasyTask: <score>
MediumTask: <score>
HardTask: <score>
```

Scores are normalized to `[0.0, 1.0]` by severity-weighted issue resolution.

The baseline runner also enforces difficulty calibration (`EasyTask >= MediumTask >= HardTask`) and raises an error if it is violated.

## Baseline Results

Current deterministic baseline output (seeded run):

```text
EasyTask: 1.0
MediumTask: 0.54
HardTask: 0.25
```

Interpretation: higher is better, and scores are expected to follow Easy >= Medium >= Hard.

## Deployment

Docker:

```bash
docker build -t openenv-code-review .
docker run --rm openenv-code-review
```

Hugging Face Spaces:

- Set `app.py` as the Spaces entrypoint.
- Use `requirements.txt` for dependency installation.
- Space type: Gradio.

## Dependencies

- `pydantic`
- `pyyaml`
- `gradio`
