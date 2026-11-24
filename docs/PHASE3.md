# Phase 3: improving on the rule

Phase 2 ended with a learned policy that tied the rule (BeliefGreedy) in the
wilderness, lost 8–12 points in urban and 6–8 in flood, and won only under a
short radio range. Phase 3 tried three ways to do better. All results below
are 100 missions per condition, every method on the same missions, run on one
machine (`runs/p3_diag.sh`, `runs/p3_chain.sh`).

## Results (share of people found)

| Method | Wilderness | Urban | Flood (held out) | Wilderness, short radio |
|---|---|---|---|---|
| Lawnmower | 12% | 65% | 53% | 12% |
| **Rule (BeliefGreedy)** | **30%** | **71%** | **55%** | **15%** |
| Step-level copy, straight-line features (C) | 28 / 29% | 69 / 64% | 44 / 44% | – |
| Step-level copy, travel-distance features (E) | 31 / 27% | 68 / 64% | 50 / 50% | – |
| Rollout planner | 29% | 71% | 51% | 15% |
| Target policy, copied from the planner | 30 / 28% | 72 / 74% | 52 / 53% | 15 / 15% |
| Target policy, RL from the rule | 31 / 27% | 72 / 74% | 54 / 54% | 17 / 16% |

Pairs are sampled / greedy (temperature 1 / 0). With 100 missions a
difference of a few points is within noise.

## 1. Travel-distance planning features (diagnostic)

Hypothesis: the step-level policy loses to the rule where obstacles make
straight-line distance wrong. Giving it the rule's own travel-distance scores
(`--plan-path`, `plan_feats=2`) closed about half of the flood gap (44 → 50%)
and changed nothing in urban or wilderness. So the hypothesis is partly
right. At this small scale the urban gap was only 2–7 points, so the 10-point
urban loss at cluster scale has another cause, not pinned down.

## 2. Target-choosing policy (`swarmsearch/targets.py`)

Instead of choosing a move every step, the policy chooses which (height, cell)
to fly to, each time the rule would. Its logits are the rule's log score times
a learned scale, plus a correction from a small fully convolutional network
whose last layer starts at zero. Untrained and greedy it is the rule, move
for move (tested). Flying, confirming alerts, the shield, battery and radio
returns stay the rule's. It is trained with decision-level PPO (a decision
lasts until the drone's next one; reward is the discounted sum in between)
with a KL penalty to the rule (`swarmsearch/targets_train.py`).

A drone makes about 250 such decisions per wilderness mission (it re-picks
about every 1.5 steps), so this does not shorten the credit-assignment
horizon much; its value is that learning starts from the rule rather than
from an imperfect copy.

Result: 60 iterations (16 missions each, about 1.7 h) ended in a tie with the
rule everywhere: within ±3 points, with no loss in any family, unlike every
step-level policy. The 8-mission validation peaked at 0.66 against the rule's
0.58, but the 100-mission evaluation shows that was noise.

## 3. Rollout planner (`swarmsearch/planner.py`)

For each drone needing a target, take the rule's top 6 diverse candidates
(always including the rule's choice), simulate the drone on its own copy of
the map for 60 steps (fly there, then follow the rule, removing the expected
detections of each look), and pick the candidate with the most discounted
expected detections.

Result: it overrides the rule in 50–58% of decisions and gains nothing (tie
in wilderness, urban and short radio; −4 in flood). Under the map's own model
its choices are better, so the model's predictions of what a look will find
must disagree with reality enough that the extra look-ahead only follows the
model's errors. Teammates' future looks are also not simulated. A policy
trained to copy the planner reached only 34–38% agreement (the choices are
close calls) and also ties the rule.

## Conclusion

The rule's one-target-ahead choice is close to the best that its search map
supports. Learning that starts from the rule no longer loses ground, but
neither RL nor look-ahead planning found a reliable gain. Gains are more
likely to come from a better map model (detection rates, survivor motion)
than from a better choice on top of the current one. Phase 2 also found that
growing the search area gives no gain.
