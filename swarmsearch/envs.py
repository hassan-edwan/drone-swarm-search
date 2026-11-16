"""One constructor for every environment: the grid environment (M1-M7) or
the field environment (M8), chosen by the scenario or sampler."""

from __future__ import annotations


def is_field(x) -> bool:
    from .field.params import FieldSampler, FieldScenario
    return isinstance(x, (FieldScenario, FieldSampler))


def make_env(scenario=None, sampler=None, **kw):
    if is_field(scenario) or is_field(sampler):
        from .field.env import FieldEnv
        env = FieldEnv(scenario=scenario, sampler=sampler, seed=kw.get("seed"))
        env.full_map = bool(kw.get("full_map", False))
        env.plan_feats = bool(kw.get("plan_feats", False))
        return env
    from .env import GridSearchEnv
    kw.pop("plan_feats", None)
    return GridSearchEnv(scenario=scenario, sampler=sampler, **kw)
