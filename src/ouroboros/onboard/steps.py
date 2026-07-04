"""The generic onboarding steps: analyze -> probe -> elicit.

analyze : scan a repo (generic heuristics) + let the matched adapter fill in
          benchmark-specific fields -> BenchmarkSpec
probe   : check env-file keys have values (WITHOUT reading the values) + run the
          adapter's endpoint/service probes -> ProbeResult
elicit  : turn whatever is missing into structured "please provide X" prompts
"""
from __future__ import annotations

from pathlib import Path

from ouroboros.onboard.adapters import find_adapter
from ouroboros.onboard.spec import BenchmarkSpec, ElicitItem, ProbeItem, ProbeResult


def analyze(repo_path: str) -> BenchmarkSpec:
    repo = Path(repo_path).expanduser().resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"benchmark repo not found: {repo}")
    adapter = find_adapter(repo)
    spec = BenchmarkSpec(name=repo.name, repo_path=str(repo),
                         adapter=adapter.name if adapter else "none")
    _generic_scan(repo, spec)
    if adapter:
        adapter.enrich(spec)
    else:
        spec.notes.append("未匹配到已知 adapter —— 需为该 benchmark 写一个 adapter 才能自动 provision/smoke")
    return spec


def _generic_scan(repo: Path, spec: BenchmarkSpec) -> None:
    for f in ("pyproject.toml", "requirements.txt", "setup.py", "environment.yml", "Dockerfile"):
        if (repo / f).exists():
            spec.deps.append(f)
    for f in (".mcp_env", ".env.example", "env.example", ".env.template"):
        if (repo / f).exists():
            spec.notes.append(f"env 模板/文件: {f}")
    for f in ("pipeline.py", "run.py", "main.py", "run-benchmark.sh"):
        if (repo / f).exists():
            spec.notes.append(f"疑似入口: {f}")


def probe(spec: BenchmarkSpec, model: str) -> ProbeResult:
    repo = Path(spec.repo_path)
    res = ProbeResult()
    set_keys: set[str] = set()
    if spec.env_file:
        env_path = repo / spec.env_file
        set_keys = _env_keys_with_values(env_path)
        res.items.append(ProbeItem(f"env file {spec.env_file}", env_path.exists(),
                                   f"{len(set_keys)} key(s) set"))
    for req in spec.required_env:
        if req.required and req.key not in set_keys:
            res.missing_env.append(req)
    adapter = find_adapter(repo)
    if adapter:
        res.items.extend(adapter.probe_env(spec, model))
    return res


def _env_keys_with_values(path: Path) -> set[str]:
    """Return keys that have a non-empty value — WITHOUT ever returning the values.

    This is the secret-safe way to check config completeness: we read the file
    but only surface which KEYS are set, never their contents.
    """
    keys: set[str] = set()
    if not path.exists():
        return keys
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if v.strip().strip('"').strip("'"):
            keys.add(k.strip())
    return keys


def elicit(spec: BenchmarkSpec, probe_res: ProbeResult) -> list[ElicitItem]:
    items: list[ElicitItem] = []
    for req in probe_res.missing_env:
        how = f"在 {spec.repo_path}/{spec.env_file} 里设置 {req.key}"
        if req.secret:
            how += "(secret:请在服务器终端安全填入,勿贴对话)"
        items.append(ElicitItem(req.key, req.purpose, how, req.secret))
    for it in probe_res.items:
        if it.name.startswith("endpoint") and not it.ok:
            items.append(ElicitItem("OPENAI_API_KEY / endpoint",
                                    f"模型端点不可用: {it.detail}",
                                    "换一个有效 key 或可用 endpoint,写入 .mcp_env 后重跑",
                                    secret=True))
    return items
