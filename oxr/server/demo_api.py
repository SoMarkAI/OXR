"""The OXR Online Demo's upload, preview, and result endpoints."""

import asyncio
from contextlib import suppress
import logging
import math
from pathlib import Path
import time

import cv2
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from oxr.pipeline import orchestrator
from oxr.pipeline.errors import get_public_error_message
from oxr.server.demo_store import DemoStore


router = APIRouter(tags=["OXR Online Demo"])
LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"code": status, "message": message, "data": None}, status_code=status)


def _response(data: dict) -> dict:
    return {"code": 0, "message": "success", "data": data}


def _decode_and_publish(store: DemoStore, job_id: str, data: bytes, file_name: str):
    images = orchestrator._decode_document_images(data, file_name)
    directory = store.job_dir(job_id) / "images"
    pages = []
    assets = []
    for page_num, image in enumerate(images):
        asset_id = f"page-{page_num:04d}.png"
        if not cv2.imwrite(str(directory / asset_id), image):
            raise OSError("Unable to create the page preview. Temporary storage may be full.")
        store.ensure_capacity()
        height, width = image.shape[:2]
        pages.append({
            "page_num": page_num,
            "page_size": {"w": width, "h": height},
            "image_url": f"/v1/demo/jobs/{job_id}/assets/{asset_id}",
        })
        assets.append(asset_id)
    store.update(job_id, status="processing", pages=pages, assets=assets)
    return images


async def _process_job(
    store: DemoStore, job_id: str, data: bytes, file_name: str,
    model_options: dict[str, float] | None = None,
) -> None:
    decode_task = None
    publish_task = None
    try:
        started_at = time.perf_counter()
        store.update(job_id, status="rendering")
        decode_task = asyncio.create_task(asyncio.to_thread(_decode_and_publish, store, job_id, data, file_name))
        images = await asyncio.shield(decode_task)
        result = await orchestrator.process_decoded_document(
            images,
            file_name,
            ["markdown", "json"],
            model_options=model_options,
            demo=True,
            asset_dir=store.job_dir(job_id) / "images",
            asset_url_prefix=f"/v1/demo/jobs/{job_id}/assets/",
            started_at=started_at,
        )
        publish_task = asyncio.create_task(asyncio.to_thread(store.publish_result, job_id, result))
        await asyncio.shield(publish_task)
    except asyncio.CancelledError:
        # Threads cannot be cancelled: finish publication before writing failed,
        # so a late decoder cannot resurrect a stopped job as processing.
        if decode_task is not None:
            await asyncio.gather(decode_task, return_exceptions=True)
        if publish_task is not None:
            await asyncio.gather(publish_task, return_exceptions=True)
        with suppress(KeyError):
            if store.get(job_id)["status"] != "success":
                store.update(job_id, status="failed", error="The server stopped while processing this document.")
        raise
    except Exception as error:
        LOGGER.warning(
            "Demo processing failed (job_id=%s, error_type=%s)", job_id, type(error).__name__,
        )
        if isinstance(error, OSError):
            with suppress(OSError):
                store.discard_payload(job_id)
        public_message = (
            "Temporary storage is unavailable or full. Try again later."
            if isinstance(error, OSError) else get_public_error_message(error)
        )
        with suppress(KeyError, OSError):
            store.update(
                job_id, status="failed", error=public_message,
                **({"pages": [], "assets": []} if isinstance(error, OSError) else {}),
            )


async def _maintain_jobs(app) -> None:
    while True:
        await asyncio.sleep(30)
        # Keep active work alive even when recognition lasts longer than the TTL.
        for job_id in tuple(app.state.demo_tasks):
            with suppress(KeyError, FileNotFoundError):
                app.state.demo_store.keep_alive(job_id)
        await asyncio.to_thread(app.state.demo_store.cleanup)


async def start_demo(app) -> None:
    app.state.demo_store = DemoStore()
    app.state.demo_tasks = {}
    await asyncio.to_thread(app.state.demo_store.cleanup)
    app.state.demo_maintenance = asyncio.create_task(_maintain_jobs(app))


async def stop_demo(app) -> None:
    app.state.demo_maintenance.cancel()
    tasks = list(app.state.demo_tasks.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(app.state.demo_maintenance, *tasks, return_exceptions=True)


# OXR Online Demo: independent of the existing SoMark-compatible parse API.
@router.post("/demo/jobs")
async def submit_demo_job(
    request: Request,
    file: UploadFile = File(...),
    temperature: str = Form("0"),
    top_p: str = Form("1"),
    repetition_penalty: str = Form("1"),
):
    # Read the raw form as well: FastAPI substitutes defaults for empty fields,
    # but explicitly blank model parameters must be rejected rather than reset.
    form = await request.form()
    model_options = {}
    for name, default, minimum, maximum in (
        ("temperature", temperature, 0.0, 2.0),
        ("top_p", top_p, 0.0, 1.0),
        ("repetition_penalty", repetition_penalty, 0.1, 2.0),
    ):
        try:
            value = float(form.get(name, default))
        except (TypeError, ValueError):
            await file.close()
            return _error(f"{name} must be a number between {minimum:g} and {maximum:g}.", 422)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            await file.close()
            return _error(f"{name} must be a finite number between {minimum:g} and {maximum:g}.", 422)
        model_options[name] = value
    file_name = Path((file.filename or "").replace("\\", "/")).name
    if Path(file_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        await file.close()
        return _error("Upload a PDF or a supported image file.", 415)
    data = bytearray()
    try:
        while chunk := await file.read(1024 * 1024):
            data.extend(chunk)
            if len(data) > MAX_UPLOAD_BYTES:
                return _error("The file exceeds the 100 MiB upload limit.", 413)
    finally:
        await file.close()
    if not data:
        return _error("The uploaded file is empty.", 422)
    store = request.app.state.demo_store
    try:
        manifest = await asyncio.to_thread(store.create, file_name)
    except OSError:
        return _error("The demo's temporary storage is unavailable or full. Try again later.", 503)
    job_id = manifest["id"]
    task = asyncio.create_task(_process_job(store, job_id, bytes(data), file_name, model_options))
    request.app.state.demo_tasks[job_id] = task
    task.add_done_callback(lambda _: request.app.state.demo_tasks.pop(job_id, None))
    return _response({"id": job_id, "status": "queued"})


@router.get("/demo/jobs/{job_id}")
async def demo_job_status(request: Request, job_id: str):
    try:
        manifest = request.app.state.demo_store.get(job_id)
    except KeyError:
        return _error("This document is unavailable or has expired. Upload it again.", 404)
    return _response({key: value for key, value in manifest.items() if key not in {"assets", "expires_at"}})


@router.get("/demo/jobs/{job_id}/result")
async def demo_job_result(request: Request, job_id: str):
    try:
        result = await asyncio.to_thread(request.app.state.demo_store.result, job_id)
    except (KeyError, FileNotFoundError):
        return _error("This document is unavailable or has expired. Upload it again.", 404)
    except ValueError:
        return _error("The document is not ready yet.", 409)
    return _response(result)


@router.get("/demo/jobs/{job_id}/assets/{asset_id}")
async def demo_job_asset(request: Request, job_id: str, asset_id: str):
    try:
        path = request.app.state.demo_store.asset(job_id, asset_id)
    except KeyError:
        return _error("The image is unavailable or has expired.", 404)
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
