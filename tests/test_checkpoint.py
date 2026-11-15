"""Training can stop and continue from its checkpoint."""

import torch

from swarmsearch.mappo import Config, train


def test_resume(tmp_path):
    cfg = Config(total_env_steps=4 * 128 * 2, n_envs=2, ckpt_every=2, out_dir=str(tmp_path))
    train(cfg)
    ck = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert ck["update"] == 4 and ck["env_steps"] == 1024 and "optimizer" in ck
    cfg2 = Config(total_env_steps=6 * 128 * 2, n_envs=2, ckpt_every=2, out_dir=str(tmp_path), resume=True)
    train(cfg2)
    ck = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert ck["update"] == 6 and ck["env_steps"] == 1536
    rows = (tmp_path / "train_log.csv").read_text().splitlines()
    assert rows[0].startswith("update") and sum(r.startswith("update") for r in rows) == 1


def test_workers_train_and_resume(tmp_path):
    cfg = Config(total_env_steps=2 * 128 * 4, n_envs=4, ckpt_every=1, workers=2, out_dir=str(tmp_path))
    train(cfg)
    cfg2 = Config(total_env_steps=3 * 128 * 4, n_envs=4, ckpt_every=1, workers=2, out_dir=str(tmp_path), resume=True)
    train(cfg2)
    ck = torch.load(tmp_path / "ckpt.pt", weights_only=False)
    assert ck["update"] == 3 and len(ck["env_rngs"]) == 4
