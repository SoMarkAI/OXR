import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from oxr.pipeline.errors import get_public_error_message
from oxr.server.schemas import (
    SyncParseResponse, 
    SyncParseData, 
    AsyncSubmitResponse, 
    AsyncSubmitData,
    AsyncQueryResponse,
    AsyncQueryData
)
from oxr.pipeline.orchestrator import process_document
from oxr.server.queue.memory_store import MemoryTaskStore

router = APIRouter()
LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_FORMATS = ["markdown", "json"]
SUPPORTED_OUTPUT_FORMATS = {"markdown", "json"}
INVALID_PARAM_CODE = 1133
DEFAULT_ELEMENT_FORMATS = {
    "image": "url",
    "formula": "latex",
    "table": "html",
    "cs": "smiles",
}
SUPPORTED_ELEMENT_FORMATS = {
    "image": {"url", "base64", "none", "file"},
    "formula": {"latex", "mathml", "ascii"},
    "table": {"html", "markdown", "image"},
    "cs": {"smiles", "image"},
}
FUNCTIONAL_ELEMENT_FORMATS = {
    "image": {"url", "base64", "none"},
    "formula": {"latex"},
    "table": {"html", "markdown"},
    "cs": {"smiles", "image"},
}
FEATURE_CONFIG_SUPPORTED_KEYS = {"keep_header_footer"}
MODEL_OPTION_RANGES = {
    "temperature": (0.0, 2.0, True),
    "top_p": (0.0, 1.0, True),
    "repetition_penalty": (0.0, None, False),
}


@dataclass
class ParseRequestOptions:
    output_formats: List[str] = field(default_factory=lambda: DEFAULT_OUTPUT_FORMATS.copy())
    keep_header_footer: bool = False
    model_options: Dict[str, float] = field(default_factory=dict)
    element_formats: Dict[str, str] = field(default_factory=lambda: DEFAULT_ELEMENT_FORMATS.copy())
    warnings: List[str] = field(default_factory=list)


def _api_error(message: str, code: int = INVALID_PARAM_CODE) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": code, "message": message, "warnings": [], "data": None})


def _format_issues(prefix: str, issues: List[str]) -> str:
    return f"{prefix}: " + "；".join(issues)


def _parse_output_formats(output_formats: Optional[List[str]]) -> tuple[List[str], List[str]]:
    errors: List[str] = []
    formats = output_formats or DEFAULT_OUTPUT_FORMATS
    normalized = [item.strip().lower() for item in formats if item and item.strip()]
    if not normalized:
        return DEFAULT_OUTPUT_FORMATS.copy(), errors

    invalid = [item for item in normalized if item not in SUPPORTED_OUTPUT_FORMATS]
    for item in invalid:
        if item == "zip":
            errors.append("output_formats=zip is not supported by open-source OXR; zip packaging is not implemented.")
        else:
            errors.append(
                f"output_formats={item} is invalid; supported values are markdown and json."
            )
    return [item for item in normalized if item in SUPPORTED_OUTPUT_FORMATS], errors


def _parse_model_option(config: Dict[str, Any], key: str) -> Optional[float]:
    if key not in config:
        return None

    value = config[key]
    if isinstance(value, bool):
        raise ValueError(f"extra.{key} must be a number.")

    try:
        parsed_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extra.{key} must be a number.") from exc

    minimum, maximum, include_minimum = MODEL_OPTION_RANGES[key]
    if include_minimum:
        if parsed_value < minimum:
            raise ValueError(f"extra.{key} must be between {minimum} and {maximum}.")
    elif parsed_value <= minimum:
        raise ValueError(f"extra.{key} must be greater than {minimum}.")

    if maximum is not None and parsed_value > maximum:
        raise ValueError(f"extra.{key} must be between {minimum} and {maximum}.")

    return parsed_value


def _parse_feature_config(feature_config: Optional[str]) -> tuple[bool, List[str], List[str]]:
    warnings: List[str] = []
    errors: List[str] = []
    if not feature_config:
        return False, warnings, errors
    try:
        config = json.loads(feature_config)
    except json.JSONDecodeError as exc:
        errors.append("feature_config must be a JSON object.")
        return False, warnings, errors

    if not isinstance(config, dict):
        errors.append("feature_config must be a JSON object.")
        return False, warnings, errors

    keep_header_footer = config.get("keep_header_footer", False)
    if not isinstance(keep_header_footer, bool):
        errors.append("feature_config.keep_header_footer must be a boolean.")
        keep_header_footer = False

    for key, value in sorted(config.items()):
        if key in FEATURE_CONFIG_SUPPORTED_KEYS:
            continue
        if not isinstance(value, bool):
            errors.append(f"feature_config.{key} must be a boolean.")
        elif value:
            warnings.append(
                f"feature_config.{key}=true is accepted but ignored; open-source OXR only implements feature_config.keep_header_footer."
            )
    return keep_header_footer, warnings, errors


def _parse_extra(extra: Optional[str]) -> tuple[Dict[str, float], List[str]]:
    errors: List[str] = []
    if not extra:
        return {}, errors
    try:
        config = json.loads(extra)
    except json.JSONDecodeError as exc:
        errors.append("extra must be a JSON object.")
        return {}, errors

    if not isinstance(config, dict):
        errors.append("extra must be a JSON object.")
        return {}, errors

    allowed_keys = set(MODEL_OPTION_RANGES)
    extra_keys = set(config) - allowed_keys
    for key in sorted(extra_keys):
        supported_keys = ", ".join(sorted(allowed_keys))
        errors.append(f"extra.{key} is unsupported; supported extra keys are {supported_keys}.")

    model_options = {}
    for key in MODEL_OPTION_RANGES:
        try:
            value = _parse_model_option(config, key)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if value is not None:
            model_options[key] = value
    return model_options, errors


def _parse_element_formats(element_formats: Optional[str]) -> tuple[Dict[str, str], List[str]]:
    errors: List[str] = []
    parsed = DEFAULT_ELEMENT_FORMATS.copy()
    if not element_formats:
        return parsed, errors

    try:
        config = json.loads(element_formats)
    except json.JSONDecodeError:
        errors.append("element_formats must be a JSON object.")
        return parsed, errors

    if not isinstance(config, dict):
        errors.append("element_formats must be a JSON object.")
        return parsed, errors

    for key, value in sorted(config.items()):
        if key not in SUPPORTED_ELEMENT_FORMATS:
            supported_keys = ", ".join(sorted(SUPPORTED_ELEMENT_FORMATS))
            errors.append(f"element_formats.{key} is invalid; supported keys are {supported_keys}.")
            continue
        if not isinstance(value, str):
            errors.append(f"element_formats.{key} must be a string.")
            continue

        normalized = value.strip().lower()
        if normalized not in SUPPORTED_ELEMENT_FORMATS[key]:
            supported_values = ", ".join(sorted(SUPPORTED_ELEMENT_FORMATS[key]))
            errors.append(
                f"element_formats.{key}={normalized} is invalid; supported values are {supported_values}."
            )
            continue

        if normalized not in FUNCTIONAL_ELEMENT_FORMATS[key]:
            if key == "image" and normalized == "file":
                errors.append(
                    "element_formats.image=file is not supported by open-source OXR because output_formats=zip is not implemented."
                )
            elif key == "formula":
                errors.append(
                    f"element_formats.formula={normalized} is not supported by open-source OXR; use element_formats.formula=latex."
                )
            elif key == "table" and normalized == "image":
                errors.append(
                    "element_formats.table=image is not supported by open-source OXR; use element_formats.table=html or markdown."
                )
            continue

        parsed[key] = normalized

    return parsed, errors


def _parse_request_options(
    output_formats: Optional[List[str]],
    feature_config: Optional[str],
    extra: Optional[str],
    element_formats: Optional[str],
) -> tuple[Optional[ParseRequestOptions], List[str]]:
    errors: List[str] = []
    parsed_output_formats, output_errors = _parse_output_formats(output_formats)
    element_format_options, element_errors = _parse_element_formats(element_formats)
    keep_header_footer, feature_warnings, feature_errors = _parse_feature_config(feature_config)
    model_options, extra_errors = _parse_extra(extra)
    errors.extend(output_errors)
    errors.extend(element_errors)
    errors.extend(feature_errors)
    errors.extend(extra_errors)
    if errors:
        return None, errors
    return ParseRequestOptions(
        output_formats=parsed_output_formats,
        keep_header_footer=keep_header_footer,
        model_options=model_options,
        element_formats=element_format_options,
        warnings=feature_warnings,
    ), errors


def _build_result_payload(result: dict) -> dict:
    return {
        "file_name": result["file_name"],
        "outputs": result["outputs"],
    }

async def process_task_background(
    task_store: MemoryTaskStore,
    task_id: str,
    file_bytes: bytes,
    file_name: str,
    output_formats: List[str],
    keep_header_footer: bool,
    model_options: Dict[str, float],
    element_formats: Dict[str, str],
):
    await task_store.set_processing(task_id)
    try:
        result = await process_document(
            file_bytes,
            file_name,
            output_formats,
            keep_header_footer=keep_header_footer,
            model_options=model_options,
            element_formats=element_formats,
        )
        store_result = {
            "file_name": result["file_name"],
            "result": _build_result_payload(result),
            "metadata": result["metadata"]
        }
        await task_store.set_success(task_id, store_result)
    except Exception as e:
        LOGGER.warning(
            "Document processing failed (task_id=%s, error_type=%s)", task_id, type(e).__name__,
        )
        await task_store.set_failed(task_id, get_public_error_message(e))

@router.post("/parse/sync", response_model=SyncParseResponse)
async def parse_sync(
    file: UploadFile = File(...),
    api_key: str = Form(...),
    output_formats: Optional[List[str]] = Form(None),
    feature_config: Optional[str] = Form(None),
    extra: Optional[str] = Form(None),
    element_formats: Optional[str] = Form(None),
):
    options, errors = _parse_request_options(output_formats, feature_config, extra, element_formats)
    if errors:
        return _api_error(_format_issues("参数错误", errors))
    assert options is not None

    task_id = str(uuid.uuid4())
    file_bytes = await file.read()
    
    try:
        result = await process_document(
            file_bytes,
            file.filename,
            options.output_formats,
            keep_header_footer=options.keep_header_footer,
            model_options=options.model_options,
            element_formats=options.element_formats,
        )
        return SyncParseResponse(
            message="任务成功",
            warnings=options.warnings,
            data=SyncParseData(
                task_id=task_id,
                error=None,
                result=_build_result_payload(result),
                metadata=result["metadata"]
            )
        )
    except Exception as e:
        LOGGER.warning(
            "Document processing failed (task_id=%s, error_type=%s)", task_id, type(e).__name__,
        )
        return _api_error(get_public_error_message(e), code=1137)

@router.post("/parse/async", response_model=AsyncSubmitResponse)
async def parse_async(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Form(...),
    output_formats: Optional[List[str]] = Form(None),
    feature_config: Optional[str] = Form(None),
    extra: Optional[str] = Form(None),
    element_formats: Optional[str] = Form(None),
):
    options, errors = _parse_request_options(output_formats, feature_config, extra, element_formats)
    if errors:
        return _api_error(_format_issues("参数错误", errors))
    assert options is not None

    task_store: MemoryTaskStore = request.app.state.task_store
    task_id = str(uuid.uuid4())
    file_bytes = await file.read()
    
    await task_store.create(task_id, file.filename)
    
    background_tasks.add_task(
        process_task_background,
        task_store,
        task_id,
        file_bytes,
        file.filename,
        options.output_formats,
        options.keep_header_footer,
        options.model_options,
        options.element_formats,
    )
    
    return AsyncSubmitResponse(
        message="任务已提交",
        warnings=options.warnings,
        data=AsyncSubmitData(
            task_id=task_id,
            status="queuing"
        )
    )

@router.post("/parse/async_check", response_model=AsyncQueryResponse)
async def query_async(
    request: Request,
    task_id: str = Form(...),
    api_key: str = Form(...),
):
    task_store: MemoryTaskStore = request.app.state.task_store
    task = await task_store.get(task_id)
    if not task:
        return _api_error(f"Task not found: {task_id}")
        
    return AsyncQueryResponse(
        message="查询成功",
        data=AsyncQueryData(
            record_id=task.get("record_id"),
            task_id=task["task_id"],
            status=task["status"],
            file_name=task.get("file_name"),
            result=task.get("result") if task["status"] == "success" else None,
            error=task.get("error"),
            metadata=task.get("metadata", {})
        )
    )
