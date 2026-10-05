import asyncio
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import httpx
import pytest


SCRIPT_PATH = Path(__file__).parents[2] / "docker" / "allinone_entrypoint.py"
SPEC = importlib.util.spec_from_file_location("allinone_entrypoint", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
AllInOneConfig = MODULE.AllInOneConfig


def _config(**overrides):
    env = {"MODEL_PATH": "/models/oxr"}
    env.update(overrides)
    return AllInOneConfig.from_env(env)


def test_commands_use_loopback_model_services_and_public_api():
    config = _config()

    assert config.dfine_command() == [
        "/opt/dfine-venv/bin/python",
        "-m",
        "uvicorn",
        "dfine_la.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--workers",
        "1",
    ]
    assert config.vllm_command()[:5] == [
        "/opt/vllm-venv/bin/python",
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "/models/oxr",
    ]
    assert "127.0.0.1" in config.vllm_command()
    assert config.api_command() == [
        "/opt/oxr-venv/bin/oxr",
        "serve",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "1",
        "--layout-analyze-model-url",
        "http://127.0.0.1:8081",
        "--oxr-model-url",
        "http://127.0.0.1:8080/v1",
        "--oxr-model-name",
        "oxr",
    ]


def test_dfine_and_api_environments_are_scoped():
    config = _config(DFINE_MODEL_PATH="/weights/layout.pth", DFINE_PORT="9001")

    dfine_env = config.dfine_env({"PYTHONPATH": "/existing"})
    assert dfine_env["MODEL_PATH"] == "/weights/layout.pth"
    assert dfine_env["MODEL_METADATA_PATH"] == "/weights/metadata.json"
    assert dfine_env["PORT"] == "9001"
    assert dfine_env["PYTHONPATH"] == "/app/services/layout_analyze:/existing"

    api_env = config.api_env({})
    assert api_env["OXR__LAYOUT_ANALYZE_MODEL__URL"] == "http://127.0.0.1:9001"
    assert api_env["OXR__OXR_MODEL__URL"] == "http://127.0.0.1:8080/v1"
    assert api_env["OXR__OXR_MODEL__MODEL_NAME"] == "oxr"
    assert config.vllm_env({"VLLM_PLUGINS": "other"})["VLLM_PLUGINS"] == "oxr"


def test_unified_repository_paths_and_frozen_inference_options():
    config = _config()
    assert config.dfine_model_path == "/models/oxr/layout_analysis/last.pth"
    assert config.dfine_env({})["MODEL_METADATA_PATH"] == "/models/oxr/layout_analysis/metadata.json"
    command = config.vllm_command()
    assert command[command.index("--chat-template") + 1] == "/models/oxr/chat_template.jinja"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert json.loads(command[command.index("--attention-config") + 1]) == {"backend": "TRITON_ATTN"}
    assert command[command.index("--max-num-seqs") + 1] == "16"


@pytest.fixture
def model_repository(tmp_path):
    for name in (
        "config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
        "plugins/vllm/pyproject.toml", "layout_analysis/last.pth",
        "layout_analysis/metadata.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture", encoding="utf-8")
    return tmp_path


def test_preflight_accepts_complete_repository(model_repository):
    config = _config(MODEL_PATH=str(model_repository))
    assert MODULE._validate_model_repository(config) == model_repository / "plugins/vllm"


@pytest.mark.parametrize("name", ["model.safetensors", "tokenizer.json", "layout_analysis/last.pth"])
def test_preflight_rejects_unresolved_lfs(model_repository, name):
    (model_repository / name).write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError, match="Unresolved Git LFS"):
        MODULE._validate_model_repository(_config(MODEL_PATH=str(model_repository)))


def test_preflight_rejects_missing_metadata(model_repository):
    (model_repository / "layout_analysis/metadata.json").unlink()
    with pytest.raises(ValueError, match="Missing model repository file"):
        MODULE._validate_model_repository(_config(MODEL_PATH=str(model_repository)))


def test_plugin_installs_from_temporary_copy_without_dependency_changes(model_repository, monkeypatch):
    async def check():
        commands = []
        staged_path = None
        original = (model_repository / "plugins/vllm/pyproject.toml").read_bytes()

        async def start(name, command, env=None):
            nonlocal staged_path
            commands.append(command)
            assert env["VLLM_PLUGINS"] == "oxr"
            if name == "OXR plugin installation":
                staged_path = Path(command[-1])
                assert staged_path != model_repository / "plugins/vllm"
                assert (staged_path / "pyproject.toml").read_bytes() == original
                assert "--no-deps" in command and "--no-build-isolation" in command
            async def wait():
                return 0
            return MODULE.ManagedProcess(name, SimpleNamespace(returncode=0, wait=wait))

        async def shutdown(*args):
            pass

        monkeypatch.setattr(MODULE, "_start_process", start)
        monkeypatch.setattr(MODULE, "_shutdown_processes", shutdown)
        processes = []
        await MODULE._prepare_plugin(
            _config(MODEL_PATH=str(model_repository)), processes, asyncio.Event(),
        )
        assert "0.25.1" in commands[0][-1] and "2.11.0+cu130" in commands[0][-1]
        assert "5.13.1" in commands[0][-1]
        assert processes == []
        assert not staged_path.exists()
        assert (model_repository / "plugins/vllm/pyproject.toml").read_bytes() == original

    asyncio.run(check())


def test_vllm_extra_args_are_parsed_without_shell_evaluation():
    config = _config(VLLM_EXTRA_ARGS="--enforce-eager --max-num-batched-tokens 8192")

    assert config.vllm_command()[-3:] == ["--enforce-eager", "--max-num-batched-tokens", "8192"]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"MODEL_PATH": ""}, "MODEL_PATH must be set"),
        ({"VLLM_GPU_MEMORY_UTILIZATION": "1"}, "must be greater than 0 and less than 1"),
        ({"DFINE_PORT": "8080"}, "must be distinct"),
        ({"DFINE_PORT": "65536"}, "must be at most 65535"),
        ({"DFINE_MODEL_PATH": ""}, "DFINE_MODEL_PATH must be set"),
        ({"VLLM_EXTRA_ARGS": "--port 9000"}, "cannot override managed options"),
        ({"VLLM_EXTRA_ARGS": "--port=9000"}, "cannot override managed options"),
        ({"VLLM_EXTRA_ARGS": "--attention-config {}"}, "cannot override managed options"),
        ({"VLLM_EXTRA_ARGS": "-ac {}"}, "cannot override managed options"),
        ({"VLLM_EXTRA_ARGS": "--attention-config.backend=FLASH_ATTN"}, "cannot override managed options"),
        ({"VLLM_EXTRA_ARGS": "--attention-backend FLASH_ATTN"}, "cannot override managed options"),
        ({"ALLINONE_STARTUP_TIMEOUT": "0"}, "must be greater than zero"),
    ],
)
def test_invalid_configuration_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


@pytest.mark.parametrize("name, payload", [
    ("D-FINE", {"status": "loading"}),
    ("D-FINE", []),
    ("vLLM", {"data": [{"id": "other"}]}),
    ("vLLM", {"data": "oxr"}),
    ("vLLM", ["oxr"]),
])
def test_http_200_is_not_sufficient_readiness(name, payload):
    response = httpx.Response(200, json=payload, request=httpx.Request("GET", "http://test"))
    with pytest.raises(ValueError):
        MODULE._validate_response(name, response)


def test_expected_model_identity_is_configurable():
    response = httpx.Response(
        200, json={"data": [{"id": "custom"}]},
        request=httpx.Request("GET", "http://test"),
    )
    MODULE._validate_response("vLLM", response, "custom")
    with pytest.raises(ValueError):
        MODULE._validate_response("vLLM", response, "oxr")


def test_startup_rechecks_process_after_http_response():
    async def check():
        process = SimpleNamespace(returncode=None)
        managed = MODULE.ManagedProcess("vLLM", process)

        def handler(request):
            process.returncode = 7
            return httpx.Response(200, json={"data": [{"id": "oxr"}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RuntimeError, match="exited during startup"):
                await MODULE._wait_for_http(
                    client, "vLLM", "http://test/models", [managed], asyncio.Event(), 1,
                )

    asyncio.run(check())


def test_shutdown_cleans_group_even_when_leader_already_exited(monkeypatch):
    async def check():
        calls = []

        def killpg(pid, signum):
            calls.append((pid, signum))

        async def wait():
            return 7

        monkeypatch.setattr(MODULE.os, "killpg", killpg)
        process = SimpleNamespace(pid=123456, returncode=7, wait=wait)
        await MODULE._shutdown_processes([MODULE.ManagedProcess("vLLM", process)], 0)
        assert calls == [(123456, signal.SIGTERM), (123456, signal.SIGKILL)]

    asyncio.run(check())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux container process groups")
def test_real_process_ignoring_sigterm_is_reaped():
    async def check():
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
            stdout=asyncio.subprocess.PIPE, start_new_session=True,
        )
        try:
            assert await asyncio.wait_for(process.stdout.readline(), 5) == b"ready\n"
            await MODULE._shutdown_processes([MODULE.ManagedProcess("stub", process)], 0.05)
            assert process.returncode == -signal.SIGKILL
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

    asyncio.run(check())


@pytest.mark.parametrize("failure", [
    "startup",
    pytest.param("exit-zero", marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Dead-leader group probes require Linux")),
    pytest.param("exit-seven", marks=pytest.mark.skipif(
        sys.platform != "linux", reason="Dead-leader group probes require Linux")),
    "stop",
])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux container process groups")
def test_supervisor_stops_all_real_owned_processes(monkeypatch, failure):
    async def check():
        owned = []
        stop_event = None

        def install_handlers(event):
            nonlocal stop_event
            stop_event = event

        async def start(name, command, env=None):
            code = 7 if failure == "exit-seven" else 0
            script = (
                "import signal,sys,time; "
                f"signal.signal(signal.SIGUSR1, lambda *_: sys.exit({code})); "
                "print('ready', flush=True); time.sleep(60)"
            )
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-c", script,
                stdout=asyncio.subprocess.PIPE, start_new_session=True,
            )
            managed = MODULE.ManagedProcess(name, process)
            owned.append(managed)
            assert await asyncio.wait_for(process.stdout.readline(), 5) == b"ready\n"
            return managed

        async def ready(client, name, url, processes, stop_requested, timeout, *args):
            if failure == "startup" and name == "vLLM":
                raise TimeoutError("stub startup failure")
            if name == "OXR API":
                if failure == "stop":
                    stop_event.set()
                else:
                    os.kill(owned[1].process.pid, signal.SIGUSR1)

        monkeypatch.setattr(MODULE, "_install_signal_handlers", install_handlers)
        monkeypatch.setattr(MODULE, "_start_process", start)
        monkeypatch.setattr(MODULE, "_wait_for_http", ready)
        async def prepare(*args):
            pass
        monkeypatch.setattr(MODULE, "_prepare_plugin", prepare)
        try:
            result = await asyncio.wait_for(MODULE._run(_config()), 10)
            assert result == {"startup": 1, "exit-zero": 1, "exit-seven": 7, "stop": 0}[failure]
            assert len(owned) == (2 if failure == "startup" else 3)
            assert all(item.process.returncode is not None for item in owned)
            for item in owned:
                with pytest.raises(ProcessLookupError):
                    os.killpg(item.process.pid, 0)
        finally:
            for item in owned:
                try:
                    os.killpg(item.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await asyncio.gather(*(item.process.wait() for item in owned))
            await asyncio.gather(*(item.process.stdout.read() for item in owned))

    asyncio.run(check())


@pytest.mark.parametrize("exit_code, expected", [(0, 1), (7, 7)])
def test_unexpected_exit_propagates_code_and_cleans_siblings(monkeypatch, exit_code, expected):
    async def check():
        owned = []

        class Process:
            def __init__(self, pid):
                self.pid = pid
                self.returncode = None
                self.done = asyncio.Event()

            def exit(self, code):
                self.returncode = code
                self.done.set()

            async def wait(self):
                await self.done.wait()
                return self.returncode

        async def start(name, command, env=None):
            managed = MODULE.ManagedProcess(name, Process(1000 + len(owned)))
            owned.append(managed)
            return managed

        async def ready(client, name, *args):
            if name == "OXR API":
                owned[1].process.exit(exit_code)

        def killpg(pid, signum):
            process = next(item.process for item in owned if item.process.pid == pid)
            if process.returncode is not None:
                raise ProcessLookupError
            if signum != 0:
                process.exit(-signum)

        monkeypatch.setattr(MODULE, "_install_signal_handlers", lambda _: None)
        monkeypatch.setattr(MODULE, "_start_process", start)
        monkeypatch.setattr(MODULE, "_wait_for_http", ready)
        async def prepare(*args):
            pass
        monkeypatch.setattr(MODULE, "_prepare_plugin", prepare)
        monkeypatch.setattr(MODULE.os, "killpg", killpg)
        assert await MODULE._run(_config()) == expected
        assert len(owned) == 3
        assert all(item.process.returncode is not None for item in owned)

    asyncio.run(check())
