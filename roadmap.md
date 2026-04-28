# RLVR positional bias reduction — research plan

**Goal:** Train a language model to answer multiple-choice questions consistently regardless of option ordering, using RL with a verifiable reward. Monitor whether accuracy degrades, stays the same, or improves as a result.

---

## Core idea

The model receives the same question with its options shuffled into all possible orderings (6 permutations for 3-option questions). A good model should always pick the same answer regardless of where it appears on the page. We reward consistency and measure whether accuracy follows.

---

## Decision summary

| Decision | Choice | Why |
|---|---|---|
| Reward signal | Entropy-based consistency | No tie-breaking ambiguity, smooth signal |
| Reward granularity | 1 reward per question | Permutations are evidence, not independent rollouts |
| Baseline | Global EMA | Simple, cross-step signal, no per-example state needed |
| Temperature | 0.1–0.2 | Keeps REINFORCE valid, reduces noise |
| KL penalty | Small (0.01–0.05) | Protects base model knowledge |

---

## Step 1 — Reward function

Replace the current mode-count reward with an entropy-based one. This produces a single scalar per question in [0, 1] with no dependency on which answer wins a tie.

```python
import math
from collections import Counter

def entropy_consistency_reward(semantic_answers):
    valid = [a for a in semantic_answers if a is not None]
    if not valid:
        return 0.0
    counts = Counter(valid)
    n = len(valid)
    n_options = 3
    probs = [c / n for c in counts.values()]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    H_max = math.log(n_options)
    return 1.0 - (H / H_max)
```

- Perfectly consistent model → reward **1.0**
- Uniform split across all options → reward **0.0**
- No modal answer needed, no tie-breaking

---

## Step 2 — Baseline

Use a global exponential moving average (EMA) across training steps. One number, updated once per step.

```python
ema_baseline = 0.5   # neutral starting point
EMA_ALPHA = 0.1      # how fast it adapts

# inside the training loop, after computing batch rewards:
batch_mean = sum(batch_rewards) / len(batch_rewards)
ema_baseline = EMA_ALPHA * batch_mean + (1 - EMA_ALPHA) * ema_baseline

advantages = [r - ema_baseline for r in batch_rewards]
```

Update the EMA *after* computing advantages for the current step — not before.

---

## Step 3 — Training loop structure

The loop needs to collect all rewards across the batch before computing advantages. This is the key structural change from the current code.

```python
for step in range(1, num_steps + 1):

    batch = sample_batch()
    batch_rewards = []
    batch_log_probs = []

    for example in batch:
        # generate all 6 permutations
        perm_log_probs, semantic_answers = generate_permutations(example)

        # 1 reward per question
        reward = entropy_consistency_reward(semantic_answers)

        batch_rewards.append(reward)
        batch_log_probs.append(torch.stack(perm_log_probs).mean())

    # update baseline with this batch
    batch_mean = sum(batch_rewards) / len(batch_rewards)
    ema_baseline = EMA_ALPHA * batch_mean + (1 - EMA_ALPHA) * ema_baseline

    # compute advantages
    advantages = [r - ema_baseline for r in batch_rewards]

    # single loss over the batch
    loss = -sum(
        adv * lp for adv, lp in zip(advantages, batch_log_probs)
    ) / len(batch)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Step 4 — Temperature

Lower the sampling temperature from 0.7 to **0.1 or 0.2** in your generation call. This reduces noise from semantic uncertainty while keeping sampling valid for REINFORCE.

```python
outputs = model.generate(
    ...
    temperature=0.1,   # was 0.7
    do_sample=True,    # must stay True — greedy breaks REINFORCE
)
```

---

## Step 5 — KL penalty

Add a small KL coefficient to prevent the model from drifting too far from its base weights. Your current run reached KL ~1.05 with no penalty — adding a small term protects general accuracy as training runs longer.

```python
kl_coeff = 0.02   # start here, adjust if needed

loss = reinforce_loss + kl_coeff * mean_kl
```

If mean_kl exceeds ~2.0 during training, increase `kl_coeff`. If consistency stops improving, decrease it.

---

## Metrics to track

| Metric | Role |
|---|---|
| `entropy_consistency` | Primary training signal — should rise |
| `perfect_accuracy` | Most honest accuracy signal — watch for degradation |
| `overall_accuracy` | Secondary accuracy check |
| `mean_kl` | Drift monitor — stop or penalise if it exceeds 2.0 |
| `grad_norm` | Training stability — should trend downward |

---

## Experimental phases

### Phase 1 — Clean baseline (steps 1–1000)
Run with all the above changes, same hyperparameters otherwise. This is the apples-to-apples comparison with your current run.

**Success criterion:** entropy consistency reaches > 0.90, perfect accuracy holds or improves vs current run.

### Phase 2 — Longer training (steps 1000–3000)
Extend training to check whether improvements hold and KL drift is controlled. This is where the KL penalty earns its keep.

**Success criterion:** accuracy does not degrade as KL stabilises.

### Phase 3 — Out-of-distribution evaluation
Evaluate on a held-out dataset not seen during training (e.g. ARC, MMLU subset). Test whether the consistency improvement generalises.

---

## What you changed from the current run and why

| Was | Now | Reason |
|---|---|---|
| Mode-count reward (0/1 per permutation) | Entropy reward (1 scalar per question) | Removes tie-breaking bias, cleaner signal |
| Within-example mean baseline | Global EMA baseline | Cross-step signal, no zero-sum advantages |
| Temperature 0.7 | Temperature 0.1 | Reduces semantic noise in consistency measure |
| kl_coeff = 0.0 | kl_coeff = 0.02 | Protects base model from drift |