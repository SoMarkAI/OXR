import os
from typing import Optional

import typer
import uvicorn


LEGACY_OPTION_MIGRATIONS = {
    "--model-url": "--oxr-model-url",
    "--model-name": "--oxr-model-name",
    "--max-tokens": "--oxr-model-max-tokens",
    "--layout-batch-size": "--layout-analyze-model-batch-size",
    "--max-layout-concurrency": "--layout-analyze-model-max-concurrency",
    "--max-recognition-concurrency": "--oxr-model-max-concurrency",
}


def _ensure_supported_workers(workers: int) -> None:
    if workers != 1:
        raise typer.BadParameter(
            "Only --workers 1 is supported because async task state is stored in process memory.",
            param_hint="--workers",
        )


def _reject_legacy_options(options: dict[str, object]) -> None:
    for legacy_option, value in options.items():
        if value is not None:
            replacement = LEGACY_OPTION_MIGRATIONS[legacy_option]
            raise typer.BadParameter(
                f"{legacy_option} is no longer supported; use {replacement}.",
                param_hint=legacy_option,
            )


def serve(
    config: Optional[str] = typer.Option(None, "--config", help="Path to config.yaml"),
    full_overlap_dedupe: Optional[bool] = typer.Option(
        None,
        "--full-overlap-dedupe/--no-full-overlap-dedupe",
        help="Across all categories, keep outer boxes and suppress any overlap by confidence.",
    ),
    table_image_placeholders: Optional[bool] = typer.Option(
        None, "--table-image-placeholders/--no-table-image-placeholders",
        help="Restore pictures inside tables with numbered TIT placeholders.",
    ),
    table_image_min_score: Optional[float] = typer.Option(
        None, "--table-image-min-score", help="Minimum LA confidence for table pictures.",
    ),
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    layout_analyze_model_url: Optional[str] = typer.Option(None, "--layout-analyze-model-url"),
    layout_analyze_model_batch_size: Optional[int] = typer.Option(None, "--layout-analyze-model-batch-size"),
    layout_analyze_model_max_concurrency: Optional[int] = typer.Option(None, "--layout-analyze-model-max-concurrency"),
    layout_analyze_model_timeout: Optional[int] = typer.Option(None, "--layout-analyze-model-timeout"),
    layout_analyze_model_retry_times: Optional[int] = typer.Option(None, "--layout-analyze-model-retry-times"),
    layout_analyze_model_score_threshold: Optional[float] = typer.Option(
        None,
        "--layout-analyze-model-score-threshold",
    ),
    layout_analyze_model_nms_iou_threshold: Optional[float] = typer.Option(
        None,
        "--layout-analyze-model-nms-iou-threshold",
    ),
    layout_analyze_model_provides_reading_order: Optional[bool] = typer.Option(
        None,
        "--layout-analyze-model-provides-reading-order/--no-layout-analyze-model-provides-reading-order",
    ),
    oxr_model_url: Optional[str] = typer.Option(None, "--oxr-model-url"),
    oxr_model_name: Optional[str] = typer.Option(None, "--oxr-model-name"),
    oxr_model_max_tokens: Optional[int] = typer.Option(None, "--oxr-model-max-tokens"),
    oxr_model_max_concurrency: Optional[int] = typer.Option(None, "--oxr-model-max-concurrency"),
    oxr_model_timeout: Optional[int] = typer.Option(None, "--oxr-model-timeout"),
    oxr_model_retry_times: Optional[int] = typer.Option(None, "--oxr-model-retry-times"),
    legacy_model_url: Optional[str] = typer.Option(None, "--model-url", hidden=True),
    legacy_model_name: Optional[str] = typer.Option(None, "--model-name", hidden=True),
    legacy_max_tokens: Optional[str] = typer.Option(None, "--max-tokens", hidden=True),
    legacy_layout_batch_size: Optional[str] = typer.Option(
        None,
        "--layout-batch-size",
        hidden=True,
    ),
    legacy_max_layout_concurrency: Optional[str] = typer.Option(
        None,
        "--max-layout-concurrency",
        hidden=True,
    ),
    legacy_max_recognition_concurrency: Optional[str] = typer.Option(
        None,
        "--max-recognition-concurrency",
        hidden=True,
    ),
    workers: Optional[int] = typer.Option(None, "--workers"),
):
    _reject_legacy_options(
        {
            "--model-url": legacy_model_url,
            "--model-name": legacy_model_name,
            "--max-tokens": legacy_max_tokens,
            "--layout-batch-size": legacy_layout_batch_size,
            "--max-layout-concurrency": legacy_max_layout_concurrency,
            "--max-recognition-concurrency": legacy_max_recognition_concurrency,
        }
    )

    from oxr.config.settings import Settings, settings

    if config:
        loaded = Settings.load_from_yaml(config)
        settings.server = loaded.server
        settings.layout_analyze_model = loaded.layout_analyze_model
        settings.oxr_model = loaded.oxr_model
        settings.pipeline = loaded.pipeline

    if host is not None:
        settings.server.host = host
    if full_overlap_dedupe is not None:
        settings.pipeline.full_overlap_dedupe = full_overlap_dedupe
    if table_image_placeholders is not None:
        settings.pipeline.table_image_placeholders = table_image_placeholders
    if table_image_min_score is not None:
        settings.pipeline.table_image_min_score = table_image_min_score
    if port is not None:
        settings.server.port = port
    if workers is not None:
        settings.server.workers = workers

    layout = settings.layout_analyze_model
    if layout_analyze_model_url is not None:
        layout.url = layout_analyze_model_url
    if layout_analyze_model_batch_size is not None:
        layout.batch_size = layout_analyze_model_batch_size
    if layout_analyze_model_max_concurrency is not None:
        layout.max_concurrency = layout_analyze_model_max_concurrency
    if layout_analyze_model_timeout is not None:
        layout.timeout = layout_analyze_model_timeout
    if layout_analyze_model_retry_times is not None:
        layout.retry_times = layout_analyze_model_retry_times
    if layout_analyze_model_score_threshold is not None:
        layout.score_threshold = layout_analyze_model_score_threshold
    if layout_analyze_model_nms_iou_threshold is not None:
        layout.nms_iou_threshold = layout_analyze_model_nms_iou_threshold
    if layout_analyze_model_provides_reading_order is not None:
        layout.provides_reading_order = layout_analyze_model_provides_reading_order

    oxr = settings.oxr_model
    if oxr_model_url is not None:
        oxr.url = oxr_model_url
    if oxr_model_name is not None:
        oxr.model_name = oxr_model_name
    if oxr_model_max_tokens is not None:
        oxr.max_tokens = oxr_model_max_tokens
    if oxr_model_max_concurrency is not None:
        oxr.max_concurrency = oxr_model_max_concurrency
    if oxr_model_timeout is not None:
        oxr.timeout = oxr_model_timeout
    if oxr_model_retry_times is not None:
        oxr.retry_times = oxr_model_retry_times

    _ensure_supported_workers(settings.server.workers)
    settings.validate_for_server()

    os.environ["OXR__SERVER__HOST"] = str(settings.server.host)
    os.environ["OXR__PIPELINE__FULL_OVERLAP_DEDUPE"] = str(
        settings.pipeline.full_overlap_dedupe
    ).lower()
    os.environ["OXR__PIPELINE__TABLE_IMAGE_PLACEHOLDERS"] = str(
        settings.pipeline.table_image_placeholders
    ).lower()
    os.environ["OXR__PIPELINE__TABLE_IMAGE_MIN_SCORE"] = str(settings.pipeline.table_image_min_score)
    os.environ["OXR__SERVER__PORT"] = str(settings.server.port)
    os.environ["OXR__SERVER__WORKERS"] = str(settings.server.workers)
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__URL"] = str(layout.url)
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__BATCH_SIZE"] = str(layout.batch_size)
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__MAX_CONCURRENCY"] = str(layout.max_concurrency)
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__TIMEOUT"] = str(layout.timeout)
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__RETRY_TIMES"] = str(layout.retry_times)
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__SCORE_THRESHOLD"] = str(layout.score_threshold)
    if layout.nms_iou_threshold is None:
        os.environ.pop("OXR__LAYOUT_ANALYZE_MODEL__NMS_IOU_THRESHOLD", None)
    else:
        os.environ["OXR__LAYOUT_ANALYZE_MODEL__NMS_IOU_THRESHOLD"] = str(
            layout.nms_iou_threshold
        )
    os.environ["OXR__LAYOUT_ANALYZE_MODEL__PROVIDES_READING_ORDER"] = str(layout.provides_reading_order).lower()
    os.environ["OXR__OXR_MODEL__URL"] = str(oxr.url)
    os.environ["OXR__OXR_MODEL__MODEL_NAME"] = oxr.model_name
    os.environ["OXR__OXR_MODEL__MAX_TOKENS"] = str(oxr.max_tokens)
    os.environ["OXR__OXR_MODEL__MAX_CONCURRENCY"] = str(oxr.max_concurrency)
    os.environ["OXR__OXR_MODEL__TIMEOUT"] = str(oxr.timeout)
    os.environ["OXR__OXR_MODEL__RETRY_TIMES"] = str(oxr.retry_times)

    print(f"Starting OXR server at {settings.server.host}:{settings.server.port}")
    uvicorn.run(
        "oxr.server.app:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=settings.server.workers,
    )
