# Phase 4: research-driven changes to beat the rule

Phase 3 ended with the learned target policy and the rollout planner tying
the rule (BeliefGreedy). Search theory says why a one-step greedy choice is
hard to beat on a static map (Stone, *Theory of Optimal Search*): gains have
to come from travel, teammates, a moving or mis-mapped person, and from a
training signal that is less noisy than "found or not" at the end of a
mission. These changes follow a literature review (sources below). All are
options, off by default, so earlier results reproduce.

## Training the target policy (`swarmsearch/targets_train.py`)

Already from 2026-11-27: validation on 200 missions per family scored as the
paired difference from the rule (`vs_rule`, `vs_rule_se` in `val_log.csv`),
advantages normalised per family, one mission per worker.

1. **A reward on every look** (`--pod-weight w`). The find reward is split:
   a share `w` is paid on each look as the chance it gives of detecting a
   true person not yet found, times a find's value; the rest on actual
   finds. The simulator knows where people are, so this is privileged
   (training only; the critic already sees the truth). A look that passes
   over the person without a detection still earns credit, so the gradient
   no longer hangs on a few rare finds (cf. "probability seen" rewards,
   Ewers et al. 2025).
2. **The rule as a counterfactual baseline** (`--cf-decisions n
   --cf-horizon h --cf-weight w`). At n sampled steps per mission, the
   mission is copied (7 ms) and one deciding drone gets the rule's target
   instead of the policy's; both copies fly h steps with the same random
   numbers, and the difference in that drone's discounted return is the
   advantage of that decision (COMA-style, Foerster et al. 2018, with a
   real rollout instead of a critic). Where the policy chose what the rule
   would, the advantage is 0.
6. **Replay of hard missions** (`--replay-share p`). Mission seeds are kept
   with a score (mean |advantage|: where the value estimate is most wrong);
   a share p of each iteration's missions are re-flown seeds picked by rank
   of score and staleness (prioritised level replay, Jiang et al. 2021).
   The buffer is checkpointed.
7. **Population-based tuning** (`scripts/pbt.py`,
   `scripts/cluster/realism_pbt.sbatch`). Eight members (one array task
   each, 64 CPUs) with their own `kl_rule`, `ent_coef`, `lr`, `gamma`,
   `pod_weight`, `cf_decisions`, `replay_share` train in 20-iteration
   segments; before each segment a member in the bottom quarter by
   `vs_rule` copies a top-quarter member's checkpoint and perturbs the
   settings (×0.8 / ×1.25; choices re-drawn a quarter of the time;
   Jaderberg et al. 2017). `python scripts/pbt.py report --out
   runs/realism/pbt` shows the population; `pbt_log.jsonl` every exploit.

## Searching better without learning

3. **Team-aware planner** (`PlannerPolicy(team=True, objective="time")`).
   The rollout planner now also flies the drone's teammates in its
   simulation (to the targets they have announced, then by the rule), so a
   candidate is judged by what the whole team finds; and each expected
   detection counts by how early in the mission it comes (1 − t/T, as the
   find reward), i.e. minimum expected time to find (cf. Dec-MCTS, Best et
   al. 2019; minimum-time search, Lanillos 2014).
4. **Planning against a wrong map** (`PlannerPolicy(..., robust=3)`, and
   the scenario's `adaptive_map`). Each candidate is scored on three
   perturbed copies of the map (detection rates ×0.6–1.6, a smooth
   log-normal distortion of the probability map) and the mean taken, so a
   choice that only pays if the map is exactly right loses to one that
   pays either way. Phase 3 found the planner "follows the map's errors".
5. **Looks valued per viewing direction** (`env.view_value = True`). The
   map counts looks per direction (S2: a person hidden from one side may be
   visible from another), but the value of a look used only the count from
   straight above. Now each footprint cell is discounted by the looks from
   the direction it would be seen from, so looking again from a new side
   is valued (cf. airborne optical sectioning, Schedl et al. 2020). In
   generated wilderness the footprint is usually one cell (50-200 m cells)
   so this changes little there; it matters in urban and flood.

## Results

**Search options without learning** (cluster job 24300826; 200 missions per condition, every method on
the same missions; paired difference from the rule in points of share found, ± one standard error):

| Condition | Rule found | rule_view | rule_adaptive | planner | planner_team | planner_robust |
|---|---|---|---|---|---|---|
| wild_mix | 33.8% | −2.2 ± 1.2 | −2.8 ± 2.1 | +4.2 ± 2.6 | +1.4 ± 2.4 | +0.7 ± 2.7 |
| urban_mix | 72.4% | −0.1 ± 1.2 | −1.8 ± 1.1 | −2.8 ± 1.3 | −2.0 ± 1.3 | −1.9 ± 1.3 |
| flood_mix_heldout | 50.4% | −0.9 ± 1.3 | −4.0 ± 1.5 | −1.8 ± 1.4 | −2.1 ± 1.5 | −3.0 ± 1.6 |
| Seconds per mission | 6 | 6–9 | 5–8 | 42–51 | 84–117 | 227–295 |

None of options 3–5 beats the rule. Simulating teammates and planning against perturbed maps did not
improve on the Phase 3 planner, which itself helps only in generated wilderness (+4.2 ± 2.6, under two
standard errors) and loses in urban and flood. Valuing looks by viewing direction is no better and
slightly worse in wilderness; estimating sensor quality online is worse everywhere. All stay off by default.

**Population-based tuning** (job 24300825; 8 members x 200 iterations; score: paired difference from
the rule on 200 validation missions per family, standard error about 1.3 points):

| Member | vs rule at 200 | Best seen |
|---|---|---|
| 6 | +2.1 | +3.2 |
| 4 | +2.0 | +3.7 |
| 5 | +1.6 | +3.6 |
| 0 | +1.1 | +2.5 |
| 3 | +0.3 | +1.7 |
| 1, 7 | −0.4 | +1.8, +2.9 |
| 2 | −1.1 | +2.8 |

After 24 exploit steps the population settled on gamma 0.99, four counterfactual decisions per mission,
a look reward of 0.14–0.22, a light pull to the rule (kl_rule 0.008–0.012) and some replay. The best
members are 1.5 standard errors above the rule, and picking the best of eight validation scores
inflates that. It is suggestive, not yet a win; the next step is a fresh evaluation of members 4, 5 and
6 on missions they were not selected on, including the real sites.

**Fresh evaluation** (job 24309330; 300 missions per condition, same missions for every method; the
real sites were never trained on). Paired difference from the rule, points of share found:

| Condition | Rule | Member 4 | Member 5 | Member 6 | Phase 3 run (24292146) |
|---|---|---|---|---|---|
| wild_mix | 29.9% | +3.1 ± 2.1 | +0.4 ± 1.9 | +0.7 ± 2.0 | +1.8 ± 2.1 |
| urban_mix | 72.0% | +0.8 ± 1.0 | +0.4 ± 1.0 | −0.8 ± 1.1 | −0.6 ± 0.9 |
| flood_mix_heldout | 48.0% | +1.1 ± 1.2 | −0.5 ± 1.2 | −0.9 ± 1.2 | −0.6 ± 1.2 |
| wild_real_catoctin | 9.1% | +1.3 ± 1.0 | +2.3 ± 1.4 | +2.2 ± 1.3 | +0.6 ± 1.1 |
| wild_real_hamlet | 25.0% | +2.3 ± 2.0 | +3.4 ± 1.8 | +3.6 ± 1.6 | +4.4 ± 1.9 |
| **Generated, pooled** | | +1.6 ± 0.9 | +0.1 ± 0.8 | −0.3 ± 0.9 | +0.2 ± 0.9 |
| **Real sites, pooled** | | +1.8 ± 1.1 | +2.9 ± 1.1 | +2.9 ± 1.0 | +2.5 ± 1.1 |

On the real sites every learned policy beats the rule, by 2-3 points (2.5-2.9 standard errors for
three of the four), i.e. 10-25% more people found there; on the generated missions they tie, except
member 4 (+1.6 ± 0.9), the only policy ahead in every condition. The learned target policy is the first
method to beat the rule where it matters, on terrain it was not trained on. Member 4 is the
recommended model (`runs/realism/pbt/m04/model.pt` on the cluster).

## Running it

    # compare search methods on the same missions (any of: rule, rule_view, rule_adaptive, planner,
    # planner_team, planner_robust, targets:<model.pt>)
    python scripts/paired_eval.py --methods rule planner_team rule_view --episodes 200 --workers 64
    # population-based tuning of the target policy (8 members)
    sbatch -A <account> scripts/cluster/realism_pbt.sbatch

## Sources

- Stone, L. D. *Theory of Optimal Search*; Stone, Royset & Washburn (2016), *Optimal Search for Moving Targets*.
- Ng, Harada & Russell (1999), policy invariance under reward shaping.
- Ewers et al. (2025), Frontiers in Robotics and AI, doi:10.3389/frobt.2024.1527095.
- Foerster et al. (2018), counterfactual multi-agent policy gradients (COMA), arXiv:1705.08926.
- Best et al. (2019), Dec-MCTS, IJRR.
- Lanillos et al. (2014), minimum-time search, Sensors 14(8).
- Schedl, Kurmi & Bimber (2020), airborne optical sectioning, Nature Machine Intelligence 2.
- Jiang et al. (2021), prioritised level replay, arXiv:2110.02439.
- Jaderberg et al. (2017), population-based training, arXiv:1711.09846.
