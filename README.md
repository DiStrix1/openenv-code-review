---
title: Code Review Environment
sdk: docker
app_port: 8000
---

# Code Review Simulation Environment

This repository implements a deterministic, multi-step reinforcement learning environment for code review simulation.

## Overview

The environment models an iterative review loop between an agent and a developer simulator:

1. The agent selects a review action.
2. The developer simulator responds by fixing issues, failing to fix, or introducing regressions.
3. The environment updates state, computes reward, and records transition metadata.
4. The episode ends when all ground-truth issues are resolved or `max_steps` is reached.

## Environment Interface

The core environment is implemented in `env/environment.py` as `CodeReviewEnv`.

- `reset()` initializes a task episode.
- `step(action)` applies action, simulates transition, computes reward, and returns `(next_state, reward, done, info)`.
- `state()` returns the current observation dictionary.

### State Schema

The observation includes the required fields:

```python
{
    "code": str,
    "issues": list,
    "history": list,
    "step_count": int,
}
```

Additional fields used by evaluation and APIs:

- `total_issues`
- `max_steps`
- `llm_evaluation_mode`

## Action Space

Defined in `env/actions.py`:

```python
ACTIONS = [
    "flag_issue",
    "suggest_fix",
    "optimize_code",
    "ignore",
]
```

Each action has explicit transition effects and validity checks.

## Transition Model

`env/developer.py` implements a seeded developer simulator with difficulty-aware behavior:

- probabilistic issue fixing
- failure outcomes
- bounded bug introduction and follow-up regressions
- context-dependent reliability adjustments

Difficulty modes:

- `easy`
- `medium`
- `hard`

## Reward Model

`env/reward.py` computes a deterministic, bounded score (`[-5, 10]`) using:

- correct fixes
- false positives
- unnecessary actions
- step efficiency
- completion progress
- aligned LLM transition signal
- completion and early-finish bonuses

Per-step reward is computed as score delta between consecutive states.

## Grading

`env/grader.py` computes final task completion score based on unresolved ground-truth issues.

- `1.0` indicates all ground-truth issues resolved.
- Lower values indicate partial completion.

## Task Suite

Defined in `env/tasks.py`:

- `EasyTask`
- `MediumTask`
- `HardTask`

Task difficulty scales through issue complexity, dependencies, and noisy/misleading patterns.

## LLM-Based Evaluation Integration

`env/llm_evaluator.py` provides step-level evaluation signals used by:

- reward shaping
- transition dynamics (`llm_guidance`)
- baseline strategy adaptation

Modes:

- deterministic heuristic mode (default)
- optional remote OpenAI mode (explicit opt-in)

Remote OpenAI mode requires:

```bash
set CODE_REVIEW_ENABLE_REMOTE_LLM=1
set CODE_REVIEW_LLM_EVAL_MODE=openai
set OPENAI_API_KEY=...
```

If remote mode is not available, the system falls back to deterministic heuristic mode.

## Determinism and Reproducibility

Randomness is controlled with seeded RNG usage (`random.seed(42)` and local seeded generators) across environment components.

Baseline rollouts are deterministic under the same configuration.

## Baseline Agent

`baseline/run_baseline.py` includes a strategy-adaptive baseline that uses:

- unresolved issue structure
- action history
- recent transition outcomes
- LLM score trend signals

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run baseline evaluation:

```bash
python baseline/run_baseline.py
```

Run Gradio app:

```bash
python app.py
```

Run FastAPI/OpenEnv server:

```bash
python inference.py
```

## OpenEnv and Deployment Files

- `openenv.yaml`
- `inference.py`
- `Dockerfile`
- `requirements.txt`

## Repository Structure

```text
env/
  __init__.py
  actions.py
  developer.py
  environment.py
  grader.py
  llm_evaluator.py
  models.py
  reward.py
  tasks.py

baseline/
  run_baseline.py

app.py
inference.py
openenv.yaml
requirements.txt
Dockerfile
README.md
```

## Current Baseline Snapshot

```text
easy   -> reward 9.45, grade 1.000
medium -> reward 4.70, grade 0.833
hard   -> reward 3.73, grade 0.625
```
