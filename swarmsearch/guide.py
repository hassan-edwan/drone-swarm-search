"""Guides that suggest a target region for each drone during a run.

The RL policy gets the suggestion as an extra input (direction to the target
region) and, during training, a shaping bonus for moving toward it that fades
to zero, so the final policy optimizes the true reward and can ignore bad
advice (in the spirit of LLM-guided exploration, e.g. ELLM, Du et al. 2023).

Backends:
- "llm:<hf model id>": a pretrained language model downloaded from Hugging
  Face reads a text summary of the search and answers with JSON targets.
- "heuristic": fast rule-based assignment; used for development and as the
  fallback when the LLM's answer cannot be parsed.
"""

from __future__ import annotations

import re

import numpy as np

from .env import COARSE, GridSearchEnv


def region_of(env: GridSearchEnv, x: int, y: int) -> tuple[int, int]:
    """(row, col) of the coarse region holding cell (x, y)."""
    return int(y * COARSE // env.grid_size), int(x * COARSE // env.grid_size)


def region_center(env: GridSearchEnv, row: int, col: int) -> tuple[int, int]:
    """Cell (x, y) at the centre of a coarse region."""
    edges = np.linspace(0, env.grid_size, COARSE + 1)
    cx = int((edges[col] + edges[col + 1] - 1) / 2)
    cy = int((edges[row] + edges[row + 1] - 1) / 2)
    return cx, cy


def region_stats(env: GridSearchEnv) -> tuple[np.ndarray, np.ndarray]:
    """Per region: expected unfound survivors (belief mass) and recently searched fraction."""
    belief = env._coarse(env.belief, np.sum).reshape(COARSE, COARSE) * max(len(env.found), 1)
    searched = env._coarse(env.freshness(), np.mean).reshape(COARSE, COARSE)
    return belief, searched


def live_drones(env: GridSearchEnv) -> list[str]:
    return [a for a in env.agents if env.alive.get(a)]


class HeuristicGuide:
    """Greedy: each drone in turn takes the free region with the best
    value (unfound mass x staleness) minus a distance cost."""

    name = "heuristic"

    def __init__(self, dist_cost: float = 0.15):
        self.dist_cost = dist_cost

    def plan_one(self, env: GridSearchEnv) -> dict[str, tuple[int, int]]:
        belief, searched = region_stats(env)
        value = belief * (1.0 - searched)
        value = value / (value.max() + 1e-9)
        taken: set[tuple[int, int]] = set()
        out = {}
        for a in live_drones(env):
            r0, c0 = region_of(env, *env.pos[a])
            best, best_s = (r0, c0), -1e9
            for r in range(COARSE):
                for c in range(COARSE):
                    if (r, c) in taken:
                        continue
                    s = value[r, c] - self.dist_cost * max(abs(r - r0), abs(c - c0))
                    if s > best_s:
                        best, best_s = (r, c), s
            taken.add(best)
            out[a] = best
        return out

    def plan(self, envs: list[GridSearchEnv]) -> list[dict[str, tuple[int, int]]]:
        return [self.plan_one(e) for e in envs]


PROMPT = """You coordinate {n} search-and-rescue drones over a disaster zone split into a {k}x{k} grid of regions (row 0 is the top, column 0 is the left).
Each region shows "value/searched": value = expected survivors still unfound there (x10, rounded), searched = percent of its cells searched recently.
Survivor movement: {movement}. Time left: {left} of {total} steps.

{table}

Drones (row,col): {drones}

Assign every drone one target region. Give different drones different regions, prefer high value and low searched, prefer regions close to the drone.
Answer with one JSON object only, with a [row, col] pair for each of {ids}, in the form {example}"""


def describe(env: GridSearchEnv) -> str:
    belief, searched = region_stats(env)
    sc = env.scenario
    rows = []
    for r in range(COARSE):
        cells = " ".join(f"{int(round(10 * belief[r, c]))}/{int(round(100 * searched[r, c]))}" for c in range(COARSE))
        rows.append(f"row {r}: {cells}")
    prof = sc.observed_profile()
    movement = (f"{prof.kind} at {prof.speed_mps:.1f} m/s" + ("" if sc.profile_known else " (assumed default)"))
    drones = live_drones(env)
    pos = ", ".join("d{} ({},{})".format(a.split("_")[1], *region_of(env, *env.pos[a])) for a in drones)
    ids = ", ".join(f"d{a.split('_')[1]}" for a in drones)
    example = "{" + ", ".join(f'"d{a.split("_")[1]}": [row, col]' for a in drones[:2]) + ", ...}"
    return PROMPT.format(n=len(drones), k=COARSE, movement=movement, left=sc.max_steps - env.t,
                         total=sc.max_steps, table="\n".join(rows), drones=pos, ids=ids, example=example)


_PAIR = re.compile(r'"?d(\d+)"?\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]')


def parse(text: str, env: GridSearchEnv) -> dict[str, tuple[int, int]]:
    alive = set(live_drones(env))
    out = {}
    for i, r, c in _PAIR.findall(text):
        a = f"drone_{i}"
        r, c = int(r), int(c)
        if a in alive and 0 <= r < COARSE and 0 <= c < COARSE:
            out[a] = (r, c)
    return out


class LLMGuide:
    """Pretrained instruction-tuned LM from Hugging Face, batched over envs."""

    def __init__(self, model_id: str, device: str = "cpu", max_new_tokens: int = 96):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"llm:{model_id}"
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.fallback = HeuristicGuide()
        self.n_drones = 0
        self.n_parsed = 0

    def plan(self, envs: list[GridSearchEnv]) -> list[dict[str, tuple[int, int]]]:
        import torch

        prompts = [
            self.tok.apply_chat_template([{"role": "user", "content": describe(e)}],
                                         tokenize=False, add_generation_prompt=True)
            for e in envs
        ]
        batch = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
        texts = self.tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        plans = []
        for e, text in zip(envs, texts):
            got = parse(text, e)
            want = live_drones(e)
            self.n_drones += len(want)
            self.n_parsed += sum(a in got for a in want)
            if len(got) < len(want):  # fill drones the LLM skipped
                fb = self.fallback.plan_one(e)
                got = {a: got.get(a, fb[a]) for a in want}
            plans.append(got)
        return plans

    @property
    def parse_rate(self) -> float:
        return self.n_parsed / max(self.n_drones, 1)


def make_guide(spec: str | None, device: str = "cpu"):
    if not spec or spec == "none":
        return None
    if spec == "heuristic":
        return HeuristicGuide()
    if spec.startswith("llm:"):
        return LLMGuide(spec[4:], device=device)
    raise ValueError(f"unknown guide {spec!r}")


def apply_plans(envs: list[GridSearchEnv], plans: list[dict[str, tuple[int, int]]]):
    for e, p in zip(envs, plans):
        e.set_goals({a: region_center(e, r, c) for a, (r, c) in p.items()})
