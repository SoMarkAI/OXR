"""Validate launch wiring without installing vLLM or invoking a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_launcher_uses_repository_plugin_and_root_template(tmp_path):
    model = tmp_path / "model with spaces"
    (model / "plugins/vllm").mkdir(parents=True)
    (model / "plugins/vllm/pyproject.toml").touch()
    (model / "chat_template.jinja").touch()
    bins = tmp_path / "bin"
    bins.mkdir()
    output = tmp_path / "calls.jsonl"
    for name in ("python3", "vllm"):
        stub = bins / name
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CALLS'], 'a') as f:\n"
            " f.write(json.dumps([sys.argv, os.getenv('VLLM_PLUGINS')]) + '\\n')\n"
        )
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bins}:{os.environ['PATH']}",
           "MODEL_PATH": str(model), "CALLS": str(output)}
    subprocess.run(["bash", str(ROOT / "services/oxr/start_vllm.sh"),
                    "--port", "9090"], env=env, check=True)
    install, serve = [json.loads(line) for line in output.read_text().splitlines()]
    assert install[0][1:] == ["-m", "pip", "install", str(model / "plugins/vllm")]
    args, plugins = serve
    assert plugins == "oxr"
    assert args[1:3] == ["serve", str(model)]
    assert args[args.index("--chat-template") + 1] == str(model / "chat_template.jinja")
    assert args[args.index("--served-model-name") + 1] == "oxr"
    assert args[args.index("--max-model-len") + 1] == "16384"
    assert "--trust-remote-code" in args
    assert json.loads(args[args.index("--attention-config") + 1]) == {"backend": "TRITON_ATTN"}
    assert args[-2:] == ["--port", "9090"]


def test_launcher_rejects_old_or_incomplete_layout(tmp_path):
    result = subprocess.run(
        ["bash", str(ROOT / "services/oxr/start_vllm.sh")],
        env={**os.environ, "MODEL_PATH": str(tmp_path)},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Expected an oxr-model repository root" in result.stderr
