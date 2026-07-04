"""Smoke tests: package imports + schema round-trips before any real wiring."""
from ouroboros import schema


def test_trajectory_roundtrip():
    meta = schema.RunMeta(model="qwen36-27b", checkpoint=None, base_url="http://x/v1")
    t = schema.Trajectory(
        task_id="t1", domain="postgres", split="train", sample_index=0,
        messages=[{"role": "user", "content": "hi"}], meta=meta,
    )
    assert '"task_id": "t1"' in t.to_json()


def test_cli_parser_builds():
    from ouroboros import cli
    cli.build_parser()  # must not raise
