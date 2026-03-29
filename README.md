# Code Review Simulation Environment (OpenEnv)

### Meta x HuggingFace OpenEnv Hackathon Submission

A reinforcement learning-style environment that simulates real-world code review interactions between an agent and a developer, enabling iterative improvement of code quality through structured feedback and rewards.

---

# Problem

Modern code review tools are static and one-shot:

* No interaction loop
* No evolving state
* No measurable learning signal

This makes them unsuitable for training intelligent agents.

---

# Solution

This project introduces a Code Review Simulation Environment where:

* An agent reviews code and takes actions
* A developer simulator responds by fixing issues or introducing new ones
* The environment evolves over multiple steps
* A reward system evaluates agent behavior
* A grader measures final code quality

This reframes code review as a sequential decision-making problem.

---

# Environment Design

## Interaction Loop

```python
obs = env.reset()

while not done:
    action = agent(obs)
    obs, reward, done, info = env.step(action)
```

---

## Action Space

```python
ACTIONS = [
    "flag_issue",
    "suggest_fix",
    "optimize_code",
    "ignore"
]
```

Each action has a clearly defined and deterministic effect.

---

## State Representation

```python
{
    "code": str,
    "issues": list,
    "history": list,
    "step_count": int
}
```

---

## Developer Simulator

Simulates realistic developer behavior:

* Fixes issues probabilistically
* May fail to fix
* May introduce new bugs (especially in hard mode)

Behavior varies by difficulty:

* Easy: high reliability
* Medium: balanced
* Hard: noisy and error-prone

---

## Reward Function

The reward function is aligned with actual performance:

* +2 for correct issue resolution
* +3 for high severity issues
* -2 for false positives
* -1 for unnecessary or repeated actions
* Step penalty for inefficiency
* Bonus for early completion

The reward signal reflects both code improvement and efficiency.

---

## Tasks

| Level  | Description                      |
| ------ | -------------------------------- |
| Easy   | Clear and obvious issues         |
| Medium | Subtle and mixed issues          |
| Hard   | Noisy, ambiguous, and misleading |

---

## Grading

Final score is computed as:

```python
grade = resolved_issues / total_issues
```

* 1.0 indicates fully clean code
* Lower values indicate partial improvement

---

# Results (Baseline Agent)

```text
easy   → reward 9.00, grade 1.000
medium → reward 3.50, grade 0.667
hard   → reward 3.00, grade 0.500
```

## Observations

* Clear separation across difficulty levels
* Reward is aligned with performance
* Hard task remains challenging but solvable

---

# Why This Approach Matters

This system:

* Models interactive code review loops
* Enables learning over multiple steps
* Captures developer unpredictability
* Provides measurable evaluation signals

It is not a static tool, but a structured environment for training and evaluating intelligent agents.

---

# Project Structure

```
env/
 ├── environment.py
 ├── developer.py
 ├── tasks.py
 ├── grader.py
 ├── models.py

baseline/
 ├── run_baseline.py

app.py
inference.py
openenv.yaml
Dockerfile
requirements.txt
```

---

# Running Locally

```bash
pip install -r requirements.txt
python app.py
```

---

# Deployment

* Compatible with HuggingFace Spaces
* OpenEnv API supported via FastAPI
* Deterministic and reproducible

---

# OpenEnv Compliance

* Implements reset(), step(), and state()
* Uses structured observation and action models
* Deterministic execution
* Container-ready setup

---

# Future Work

* Train RL agents (PPO, DQN)
* Integrate real-world code datasets
* Add static analysis tools
* Extend to multi-agent review systems

---

# Author

Dishu
B.Tech Student | Software Developer
