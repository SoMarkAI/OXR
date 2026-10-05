#!/usr/bin/env python3
"""Supervise the D-FINE, vLLM, and OXR API processes in one container."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shlex
import shutil
import signal
import sys
import tempfile
from typing import Mapping, Sequence

import httpx


LOGGER = logging.getLogger("oxr.allinone")
APP_ROOT = Path("/app")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero; got {value}")
    return value


def _env_port(env: Mapping[str, str], name: str, default: int) -> int:
    value = _env_int(env, name, default)
    if value > 65535:
        raise ValueError(f"{name} must be at most 65535; got {value}")
    return value


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be set")
    return value


@dataclass(frozen=True)
class AllInOneConfig:
    model_name: str
    served_model_name: str
    dfine_model_path: str
    api_host: str
    api_port: int
    vllm_port: int
    dfine_port: int
    gpu_memory_utilization: float
    max_model_len: int
    startup_timeout: int
    shutdown_timeout: int
    vllm_python: str
    oxr_bin: str
    dfine_python: str
    vllm_extra_args: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AllInOneConfig":
        source = os.environ if env is None else env
        raw_utilization = source.get("VLLM_GPU_MEMORY_UTILIZATION", "0.70")
        try:
            gpu_memory_utilization = float(raw_utilization)
        except ValueError as exc:
            raise ValueError(
                "VLLM_GPU_MEMORY_UTILIZATION must be a number between 0 and 1; "
                f"got {raw_utilization!r}"
            ) from exc
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError(
                "VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and less than 1; "
                f"got {gpu_memory_utilization}"
            )

        ports = {
            "OXR_API_PORT": _env_port(source, "OXR_API_PORT", 8000),
            "VLLM_PORT": _env_port(source, "VLLM_PORT", 8080),
            "DFINE_PORT": _env_port(source, "DFINE_PORT", 8081),
        }
        if len(set(ports.values())) != len(ports):
            raise ValueError(f"OXR_API_PORT, VLLM_PORT, and DFINE_PORT must be distinct: {ports}")

        model_path = _required(source, "MODEL_PATH")
        dfine_model_path = source.get(
            "DFINE_MODEL_PATH", str(Path(model_path) / "layout_analysis" / "last.pth")
        ).strip()
        if not dfine_model_path:
            raise ValueError("DFINE_MODEL_PATH must be set")

        vllm_extra_args = tuple(shlex.split(source.get("VLLM_EXTRA_ARGS", "")))
        managed_vllm_options = {
            "--gpu-memory-utilization",
            "--host",
            "--max-model-len",
            "--port",
            "--served-model-name",
            "--chat-template",
            "--dtype",
            "--model",
            "--max-num-seqs",
            "--attention-config",
            "--attention-backend",
            "-ac",
        }
        conflicts = sorted(
            argument
            for argument in vllm_extra_args
            if argument in managed_vllm_options
            or any(argument.startswith(f"{option}=") for option in managed_vllm_options)
            or argument.startswith("--attention-config.")
            or argument.startswith("-ac.")
        )
        if conflicts:
            raise ValueError(
                "VLLM_EXTRA_ARGS cannot override managed options: " + ", ".join(conflicts)
            )

        return cls(
            model_name=model_path,
            served_model_name=source.get("OXR_SERVED_MODEL_NAME", "oxr").strip() or "oxr",
            dfine_model_path=dfine_model_path,
            api_host=source.get("OXR_API_HOST", "0.0.0.0").strip() or "0.0.0.0",
            api_port=ports["OXR_API_PORT"],
            vllm_port=ports["VLLM_PORT"],
            dfine_port=ports["DFINE_PORT"],
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=_env_int(source, "VLLM_MAX_MODEL_LEN", 16384),
            startup_timeout=_env_int(source, "ALLINONE_STARTUP_TIMEOUT", 900),
            shutdown_timeout=_env_int(source, "ALLINONE_SHUTDOWN_TIMEOUT", 30),
            vllm_python=source.get("VLLM_PYTHON", "/opt/vllm-venv/bin/python"),
            oxr_bin=source.get("OXR_BIN", "/opt/oxr-venv/bin/oxr"),
            dfine_python=source.get("DFINE_PYTHON", "/opt/dfine-venv/bin/python"),
            vllm_extra_args=vllm_extra_args,
        )

    @property
    def dfine_url(self) -> str:
        return f"http://127.0.0.1:{self.dfine_port}"

    @property
    def vllm_url(self) -> str:
        return f"http://127.0.0.1:{self.vllm_port}/v1"

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def dfine_command(self) -> list[str]:
        return [
            self.dfine_python,
            "-m",
            "uvicorn",
            "dfine_la.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.dfine_port),
            "--workers",
            "1",
        ]

    def vllm_command(self) -> list[str]:
        return [
            self.vllm_python,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            self.model_name,
            "--served-model-name",
            self.served_model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.vllm_port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--trust-remote-code",
            "--chat-template",
            str(Path(self.model_name) / "chat_template.jinja"),
            "--dtype",
            "bfloat16",
            "--attention-config",
            '{"backend":"TRITON_ATTN"}',
            "--max-num-seqs",
            "16",
            *self.vllm_extra_args,
        ]

    def api_command(self) -> list[str]:
        return [
            self.oxr_bin,
            "serve",
            "--host",
            self.api_host,
            "--port",
            str(self.api_port),
            "--workers",
            "1",
            "--layout-analyze-model-url",
            self.dfine_url,
            "--oxr-model-url",
            self.vllm_url,
            "--oxr-model-name",
            self.served_model_name,
        ]

    def dfine_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env["MODEL_PATH"] = self.dfine_model_path
        env["MODEL_METADATA_PATH"] = env.get("DFINE_MODEL_METADATA_PATH") or str(
            Path(self.dfine_model_path).with_name("metadata.json")
        )
        env["PORT"] = str(self.dfine_port)
        dfine_root = str(APP_ROOT / "services" / "layout_analyze")
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{dfine_root}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else dfine_root
        )
        return env

    def vllm_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env["VLLM_PLUGINS"] = "oxr"
        return env

    def api_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env["OXR__LAYOUT_ANALYZE_MODEL__URL"] = self.dfine_url
        env["OXR__OXR_MODEL__URL"] = self.vllm_url
        env["OXR__OXR_MODEL__MODEL_NAME"] = self.served_model_name
        return env


@dataclass
class ManagedProcess:
    name: str
    process: asyncio.subprocess.Process


async def _start_process(
    name: str,
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> ManagedProcess:
    LOGGER.info("starting %s", name)
    process = await asyncio.create_subprocess_exec(
        *command,
        env=None if env is None else dict(env),
        start_new_session=True,
    )
    return ManagedProcess(name=name, process=process)


def _assert_processes_running(processes: Sequence[ManagedProcess]) -> None:
    for managed in processes:
        return_code = managed.process.returncode
        if return_code is not None:
            raise RuntimeError(f"{managed.name} exited during startup with code {return_code}")


def _validate_response(
    name: str, response: httpx.Response, served_model_name: str = "oxr",
) -> None:
    """A reachable endpoint is not necessarily the expected ready service."""
    response.raise_for_status()
    if name == "D-FINE":
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("status") != "healthy":
            raise ValueError("D-FINE did not report healthy status")
    elif name == "vLLM":
        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list) or not any(
            isinstance(model, dict) and model.get("id") == served_model_name
            for model in models
        ):
            raise ValueError("vLLM did not advertise the expected model ID")


async def _wait_for_http(
    client: httpx.AsyncClient,
    name: str,
    url: str,
    processes: Sequence[ManagedProcess],
    stop_requested: asyncio.Event,
    timeout: int,
    served_model_name: str = "oxr",
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_error = "no response"
    while asyncio.get_running_loop().time() < deadline:
        _assert_processes_running(processes)
        if stop_requested.is_set():
            raise InterruptedError("shutdown requested during startup")
        try:
            response = await client.get(url)
            _validate_response(name, response, served_model_name)
            _assert_processes_running(processes)
            if stop_requested.is_set():
                raise InterruptedError("shutdown requested during startup")
            LOGGER.info("%s is ready at %s", name, url)
            return
        except (httpx.HTTPError, OSError, ValueError) as exc:
            last_error = str(exc)
        try:
            await asyncio.wait_for(stop_requested.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
    raise TimeoutError(f"{name} did not become ready at {url}: {last_error}")


async def _shutdown_processes(
    processes: Sequence[ManagedProcess],
    timeout: int,
) -> None:
    # A dead service leader can leave workers in its process group (and on GPU).
    # Do not filter groups by the leader's return code.
    for managed in reversed(processes):
        LOGGER.info("stopping %s", managed.name)
        try:
            os.killpg(managed.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = asyncio.get_running_loop().time() + timeout
    remaining = list(processes)
    while remaining and asyncio.get_running_loop().time() < deadline:
        alive = []
        for managed in remaining:
            try:
                os.killpg(managed.process.pid, 0)
            except ProcessLookupError:
                continue
            alive.append(managed)
        remaining = alive
        if remaining:
            await asyncio.sleep(0.05)
    for managed in remaining:
        LOGGER.warning("killing %s process group after shutdown timeout", managed.name)
        try:
            os.killpg(managed.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    await asyncio.gather(*(item.process.wait() for item in processes))


def _install_signal_handlers(stop_requested: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_requested.set)


def _validate_model_repository(config: AllInOneConfig) -> Path:
    root = Path(config.model_name)
    files = [
        root / name for name in (
            "config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja",
            "plugins/vllm/pyproject.toml",
        )
    ]
    files.extend([
        Path(config.dfine_model_path),
        Path(config.dfine_env()["MODEL_METADATA_PATH"]),
    ])
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing model repository file: {path}")
        with path.open("rb") as handle:
            if handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Unresolved Git LFS pointer; run git lfs pull: {path}")
    return root / "plugins" / "vllm"


async def _prepare_plugin(
    config: AllInOneConfig, processes: list[ManagedProcess], stop_requested: asyncio.Event,
) -> None:
    plugin = _validate_model_repository(config)

    async def run_checked(name: str, command: Sequence[str]) -> None:
        managed = await _start_process(name, command, config.vllm_env())
        processes.append(managed)
        task = asyncio.create_task(managed.process.wait())
        stop_task = asyncio.create_task(stop_requested.wait())
        try:
            done, _ = await asyncio.wait(
                [task, stop_task], timeout=config.startup_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done:
                raise InterruptedError("shutdown requested during plugin preparation")
            if task not in done:
                raise TimeoutError(f"{name} timed out")
            if task.result() != 0:
                raise RuntimeError(f"{name} failed with code {task.result()}")
            await _shutdown_processes([managed], 0)
            processes.remove(managed)
        finally:
            for pending in (task, stop_task):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, stop_task, return_exceptions=True)

    # Refuse a different runtime; installing the model plugin must not upgrade or
    # downgrade the benchmark's vLLM/PyTorch/Transformers environment.
    await run_checked("runtime version check", [config.vllm_python, "-c", (
        "from importlib.metadata import version; "
        "expected={'vllm':'0.25.1','torch':'2.11.0+cu130','transformers':'5.13.1'}; "
        "actual={key:version(key) for key in expected}; "
        "\nif actual != expected: raise SystemExit('Unsupported model runtime: '+str(actual))"
    )])
    # Stage only the small plugin package. pip/setuptools must not write into
    # the read-only model repository or copy model weights into the image.
    with tempfile.TemporaryDirectory(prefix="oxr-plugin-") as temporary:
        staged = Path(temporary) / "plugin"
        shutil.copytree(plugin, staged, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", "build", "dist",
        ))
        await run_checked("OXR plugin installation", [
            config.vllm_python, "-m", "pip", "install", "--no-deps",
            "--no-build-isolation", str(staged),
        ])


async def _run(config: AllInOneConfig) -> int:
    stop_requested = asyncio.Event()
    _install_signal_handlers(stop_requested)
    processes: list[ManagedProcess] = []

    try:
        await _prepare_plugin(config, processes, stop_requested)
        async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
            dfine = await _start_process(
                "D-FINE",
                config.dfine_command(),
                config.dfine_env(),
            )
            processes.append(dfine)
            await _wait_for_http(
                client,
                "D-FINE",
                f"{config.dfine_url}/health",
                processes,
                stop_requested,
                config.startup_timeout,
            )

            vllm = await _start_process("vLLM", config.vllm_command(), config.vllm_env())
            processes.append(vllm)
            await _wait_for_http(
                client,
                "vLLM",
                f"{config.vllm_url}/models",
                processes,
                stop_requested,
                config.startup_timeout,
                config.served_model_name,
            )

            api = await _start_process("OXR API", config.api_command(), config.api_env())
            processes.append(api)
            await _wait_for_http(
                client,
                "OXR API",
                f"{config.api_url}/docs",
                processes,
                stop_requested,
                config.startup_timeout,
            )

        LOGGER.info(
            "all services are ready; OXR API is listening on %s:%s",
            config.api_host,
            config.api_port,
        )
        wait_tasks = {
            asyncio.create_task(item.process.wait()): item
            for item in processes
        }
        stop_task = asyncio.create_task(stop_requested.wait())
        done, pending = await asyncio.wait(
            [*wait_tasks, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if stop_task in done:
            return 0

        exited_task = next(task for task in done if task in wait_tasks)
        exited = wait_tasks[exited_task]
        return_code = exited_task.result()
        LOGGER.error("%s exited unexpectedly with code %s", exited.name, return_code)
        return return_code if return_code != 0 else 1
    except (InterruptedError, KeyboardInterrupt):
        return 0
    except Exception:
        LOGGER.exception("all-in-one startup failed")
        return 1
    finally:
        await _shutdown_processes(processes, config.shutdown_timeout)


async def _healthcheck(config: AllInOneConfig) -> int:
    checks = (
        ("D-FINE", f"{config.dfine_url}/health"),
        ("vLLM", f"{config.vllm_url}/models"),
        ("OXR API", f"{config.api_url}/docs"),
    )
    async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
        for name, url in checks:
            try:
                response = await client.get(url)
                _validate_response(name, response, config.served_model_name)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                LOGGER.error("%s healthcheck failed at %s: %s", name, url, exc)
                return 1
    return 0


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        config = AllInOneConfig.from_env()
    except ValueError as exc:
        LOGGER.error("invalid all-in-one configuration: %s", exc)
        return 2
    if args.healthcheck:
        return asyncio.run(_healthcheck(config))
    return asyncio.run(_run(config))


if __name__ == "__main__":
    raise SystemExit(main())
