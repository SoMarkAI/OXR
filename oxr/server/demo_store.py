"""Short-lived demo jobs shared by the API's local worker processes."""

from contextlib import contextmanager
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid


JOB_TTL_SECONDS = 60 * 60
MAX_JOBS = 32
MAX_CACHE_BYTES = 1024 * 1024 * 1024
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
ASSET_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*\.(?:png|jpg|jpeg|webp)$")


def default_store_root() -> Path:
    installation = hashlib.sha256(str(Path(__file__).resolve().parent).encode()).hexdigest()[:12]
    user = hashlib.sha256(getpass.getuser().encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"oxr-demo-{user}-{installation}"


@contextmanager
def _admission_lock(path: Path):
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


class DemoStore:
    def __init__(self, root: Path | None = None, ttl: int = JOB_TTL_SECONDS,
                 max_jobs: int = MAX_JOBS, max_bytes: int = MAX_CACHE_BYTES):
        self.root = root if root is not None else default_store_root()
        self.ttl = ttl
        self.max_jobs = max_jobs
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        return self.root / job_id

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, file_name: str) -> dict:
        # Admission is shared by workers as well as concurrent upload threads.
        with _admission_lock(self.root / ".admission.lock"):
            return self._create(file_name)

    def _create(self, file_name: str) -> dict:
        self.cleanup()
        jobs = sum(1 for path in self.root.iterdir() if path.is_dir() and JOB_ID.fullmatch(path.name))
        if jobs >= self.max_jobs:
            raise OSError("The demo has reached its temporary document limit. Try again later.")
        self.ensure_capacity()
        job_id = uuid.uuid4().hex
        directory = self.job_dir(job_id)
        (directory / "images").mkdir(parents=True, mode=0o700)
        (directory / "lease").touch()
        now = time.time()
        manifest = {
            "id": job_id,
            "status": "queued",
            "file_name": file_name,
            "pages": [],
            "metadata": {},
            "error": None,
            "result_url": None,
            "assets": [],
            "expires_at": now + self.ttl,
        }
        try:
            self._write_json(directory / "manifest.json", manifest)
        except OSError:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return manifest

    def ensure_capacity(self, extra_bytes: int = 0) -> None:
        total = extra_bytes
        for path in self.root.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
            if total > self.max_bytes:
                raise OSError("The demo's temporary storage is full. Try again later.")

    def discard_payload(self, job_id: str) -> None:
        directory = self.job_dir(job_id)
        shutil.rmtree(directory / "images", ignore_errors=True)
        (directory / "result.json").unlink(missing_ok=True)

    def get(self, job_id: str) -> dict:
        try:
            manifest = json.loads((self.job_dir(job_id) / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raise KeyError(job_id) from None
        if self._expires_at(self.job_dir(job_id), manifest) <= time.time():
            raise KeyError(job_id)
        return manifest

    def _expires_at(self, directory: Path, manifest: dict) -> float:
        try:
            return max(manifest["expires_at"], (directory / "lease").stat().st_mtime + self.ttl)
        except FileNotFoundError:
            return manifest["expires_at"]

    def keep_alive(self, job_id: str) -> None:
        # A separate lease prevents heartbeat writes racing manifest publication.
        (self.job_dir(job_id) / "lease").touch(exist_ok=True)

    def update(self, job_id: str, **changes) -> dict:
        manifest = self.get(job_id)
        manifest.update(changes, expires_at=time.time() + self.ttl)
        self._write_json(self.job_dir(job_id) / "manifest.json", manifest)
        return manifest

    def publish_result(self, job_id: str, result: dict) -> None:
        directory = self.job_dir(job_id)
        self._write_json(directory / "result.json", result)
        self.ensure_capacity()
        assets = sorted(
            path.name for path in (directory / "images").iterdir()
            if ASSET_ID.fullmatch(path.name) and path.is_file() and not path.is_symlink()
        )
        self.update(
            job_id,
            status="success",
            metadata=result.get("metadata", {}),
            result_url=f"/v1/demo/jobs/{job_id}/result",
            assets=assets,
        )

    def result(self, job_id: str) -> dict:
        manifest = self.get(job_id)
        if manifest["status"] != "success":
            raise ValueError("The document is not ready yet.")
        return json.loads((self.job_dir(job_id) / "result.json").read_text(encoding="utf-8"))

    def asset(self, job_id: str, asset_id: str) -> Path:
        manifest = self.get(job_id)
        if not ASSET_ID.fullmatch(asset_id) or asset_id not in manifest["assets"]:
            raise KeyError(asset_id)
        directory = self.job_dir(job_id) / "images"
        path = directory / asset_id
        if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
            raise KeyError(asset_id)
        return path

    def cleanup(self) -> None:
        now = time.time()
        for directory in self.root.iterdir():
            if not directory.is_dir() or not JOB_ID.fullmatch(directory.name):
                continue
            try:
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
                expired = self._expires_at(directory, manifest) <= now
            except (FileNotFoundError, json.JSONDecodeError):
                # Also reclaim interrupted job creation after the normal TTL.
                try:
                    expired = directory.stat().st_mtime + self.ttl <= now
                except FileNotFoundError:
                    continue
            if expired:
                shutil.rmtree(directory, ignore_errors=True)
