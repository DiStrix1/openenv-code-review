# OpenEnv Code Review Environment

A multi-step simulation environment for evaluating AI code review agents.

## Motivation

Most AI code review systems are evaluated using static, one-shot benchmarks. In practice, code review on platforms such as GitHub is iterative, interactive, and dependent on developer responses. This project addresses that gap by introducing a dynamic, multi-step evaluation environment.

## Problem

Existing evaluation methods often fail to capture:

- Iterative feedback loops
- Evolving code states
- Developer-agent interaction
- Partial progress over time

As a result, they can overestimate agent capability and underrepresent real-world behavior.

## Solution

This project provides an OpenEnv-compatible closed-loop workflow:

Agent reviews code -> developer responds -> code evolves -> repeat.

Agent actions directly influence future environment states.

## Environment Overview

### Interaction Loop

Observation -> Action -> Environment Update -> Reward -> Next Observation

### Observation Space

- `diff` (`string`): pull request code diff
- `comments` (`list[string]`): prior agent comments
- `iteration` (`int`): current step

Ground-truth issues are hidden from the agent.

### Action Space

- `comment(issue_id)`: flag an issue
- `approve`: accept PR
- `request_changes`: ask for revision

Optional:

- `severity` (`low`, `medium`, `high`)

### Hidden State

- Actual issues (ground truth)
- Severity levels
- Resolution status

### Developer Simulator

A probabilistic model that:

- Fixes issues based on agent feedback
- Prioritizes high-severity issues
- Introduces new issues occasionally
- Adapts behavior by task difficulty

## Tasks

### Easy

- Clear, high-signal issues
- Short diff
- Objective: detect obvious correctness issues

### Medium

- Fewer explicit patterns
- More ambiguity
- Objective: identify performance and safety issues

### Hard

- Multiple interacting issues
- Mixed severity
- Objective: prioritize under uncertainty

## Reward Design

Continuous reward range: `[-5.0, 1.5]`

Positive signals:

- Correct issue detection
- Higher reward for higher severity

Penalties:

- False positives
- Repeated comments
- Incorrect approval
- Unresolved critical issues
- Inefficient long loops

This design supports meaningful learning signals while reducing reward hacking.

## Results

Baseline (heuristic agent):

- `EasyTask`: 0.50
- `MediumTask`: 0.36
- `HardTask`: 0.33

LLM inference:

- `EasyTask`: 0.73
- `MediumTask`: 0.27
- `HardTask`: 0.21

Interpretation:

- Performance decreases with task difficulty
- No task is trivial
- No task collapses
- The environment differentiates agent capability

## Project Structure

```text
env/
  environment.py
  developer.py
  tasks.py
  models.py
  grader.py

baseline/
  run_baseline.py

app.py
inference.py
openenv.yaml
Dockerfile
requirements.txt
```

## Running the Project

Local:

```bash
pip install -r requirements.txt
python -m baseline.run_baseline
```

Inference (LLM):

```bash
python inference.py
```

Docker:

```bash
docker build -t openenv-code-review .
docker run openenv-code-review
```

Hugging Face Spaces:

- Entry point: `app.py`
- Framework: Gradio
- Output: evaluation results

## OpenEnv Compliance

- `step()` / `reset()` / `state()`
- Typed models with Pydantic
- Multi-step interaction
- Task-based evaluation (Easy -> Medium -> Hard)
- Deterministic grading (0.0-1.0)
- Dockerized execution
- Hugging Face deployment
- OpenAI-based inference script

## Key Design Challenges

1. Difficulty calibration
Easy > Medium > Hard
2. Agent-task alignment
Avoid pattern overfitting and trivial solutions
3. Developer behavior
Balance agent influence with environment realism
4. Reward shaping
Provide dense feedback with stable learning signals

## Key Insight

The core challenge is not only detecting issues, but evaluating how agents behave in dynamic, evolving systems.

## Conclusion

This environment provides a more realistic benchmark for AI code review by capturing multi-step interaction, uncertainty, and developer feedback loops.

## Links

Live Demo:

- [Hugging Face Space link](https://huggingface.co/spaces/DishuMahajan/openenv-code-review-env)

GitHub Repository:

- [Repository link](https://github.com/DiStrix1/openenv-code-review)
