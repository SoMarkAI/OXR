import asyncio
import inspect

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from oxr.cli.main import app as cli_app
from oxr.cli.cmd_serve import _ensure_supported_workers, serve
from oxr.config.settings import Settings
from oxr.server.queue.memory_store import MemoryTaskStore


def test_memory_task_store_transitions():
    async def scenario():
        store = MemoryTaskStore()
        await store.create("task-1")
        await store.set_processing("task-1")
        await store.set_success(
            "task-1",
            {
                "result": {"file_name": "demo.pdf"},
                "metadata": {"pages": 1},
            },
        )
        return await store.get("task-1")

    task = asyncio.run(scenario())

    assert task == {
        "record_id": None,
        "task_id": "task-1",
        "status": "success",
        "file_name": None,
        "result": {"file_name": "demo.pdf"},
        "error": None,
        "metadata": {"pages": 1},
    }


def test_settings_load_from_yaml_without_queue_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "server:",
                "  host: 127.0.0.1",
                "layout_analyze_model:",
                "  url: http://localhost:8081",
                "  batch_size: 4",
                "oxr_model:",
                "  url: http://localhost:8080/v1",
                "  max_concurrency: 8",
                "pipeline:",
                "  image_output_dir: parsed",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings.load_from_yaml(str(config_file))

    assert settings.server.host == "127.0.0.1"
    assert settings.layout_analyze_model.url == "http://localhost:8081"
    assert settings.layout_analyze_model.batch_size == 4
    assert settings.layout_analyze_model.nms_iou_threshold == 0.6
    assert settings.oxr_model.url == "http://localhost:8080/v1"
    assert settings.oxr_model.max_concurrency == 8
    assert settings.pipeline.image_output_dir == "parsed"
    assert settings.server.workers == 1


def test_settings_loads_layout_score_threshold_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "layout_analyze_model:",
                "  url: http://localhost:8081",
                "  score_threshold: 0.65",
                "oxr_model:",
                "  url: http://localhost:8080/v1",
            ]
        ),
        encoding="utf-8",
    )

    loaded = Settings.load_from_yaml(str(config_file))

    assert loaded.layout_analyze_model.score_threshold == 0.65


def test_settings_reads_layout_score_threshold_from_environment(monkeypatch):
    monkeypatch.setenv("OXR__LAYOUT_ANALYZE_MODEL__SCORE_THRESHOLD", "0.7")

    loaded = Settings()

    assert loaded.layout_analyze_model.score_threshold == 0.7


def test_settings_reads_layout_nms_threshold_from_environment(monkeypatch):
    monkeypatch.setenv("OXR__LAYOUT_ANALYZE_MODEL__NMS_IOU_THRESHOLD", "0.55")

    loaded = Settings()

    assert loaded.layout_analyze_model.nms_iou_threshold == 0.55


def test_pipeline_full_overlap_dedupe_defaults_to_false():
    loaded = Settings()

    assert loaded.pipeline.model_dump() == {
        "image_output_dir": "outputs",
        "full_overlap_dedupe": False,
        "table_image_placeholders": False,
        "table_image_min_score": 0.5,
    }


@pytest.mark.parametrize("enabled", [True, False])
def test_full_overlap_dedupe_loads_from_yaml_and_environment(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("OXR__PIPELINE__FULL_OVERLAP_DEDUPE", str(enabled).lower())
    assert Settings().pipeline.full_overlap_dedupe is enabled
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"pipeline:\n  full_overlap_dedupe: {str(not enabled).lower()}\n", encoding="utf-8",
    )
    assert Settings.load_from_yaml(str(config_file)).pipeline.full_overlap_dedupe is not enabled


def test_full_overlap_dedupe_rejects_invalid_boolean():
    with pytest.raises(ValidationError, match="full_overlap_dedupe"):
        Settings(pipeline={"full_overlap_dedupe": "typo"})


@pytest.mark.parametrize("enabled", [True, False])
def test_table_image_setting_loads_yaml_and_environment(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("OXR__PIPELINE__TABLE_IMAGE_PLACEHOLDERS", str(enabled).lower())
    monkeypatch.setenv("OXR__PIPELINE__TABLE_IMAGE_MIN_SCORE", "0.75")
    assert Settings().pipeline.table_image_placeholders is enabled
    assert Settings().pipeline.table_image_min_score == 0.75
    config = tmp_path / "config.yaml"
    config.write_text(
        f"pipeline:\n  table_image_placeholders: {str(not enabled).lower()}\n"
        "  table_image_min_score: 0.6\n", encoding="utf-8",
    )
    loaded = Settings.load_from_yaml(str(config))
    assert loaded.pipeline.table_image_placeholders is not enabled
    assert loaded.pipeline.table_image_min_score == 0.6


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_table_image_score_rejects_invalid_values(score):
    with pytest.raises(ValidationError, match="table_image_min_score"):
        Settings(pipeline={"table_image_min_score": score})


@pytest.mark.parametrize("enabled", [True, False])
def test_table_image_cli_overrides_yaml_and_exports_runtime_environment(tmp_path, monkeypatch, enabled):
    from oxr.config.settings import settings as runtime_settings

    config = tmp_path / "config.yaml"
    config.write_text(
        "layout_analyze_model:\n  url: http://layout:8081\noxr_model:\n  url: http://oxr:8080/v1\n"
        f"pipeline:\n  table_image_placeholders: {str(not enabled).lower()}\n  table_image_min_score: 0.6\n",
        encoding="utf-8",
    )
    for field in ("server", "layout_analyze_model", "oxr_model", "pipeline"):
        monkeypatch.setattr(runtime_settings, field, getattr(runtime_settings, field))
    captured = {}
    monkeypatch.setattr("oxr.cli.cmd_serve.os.environ", captured)
    monkeypatch.setattr("oxr.cli.cmd_serve.uvicorn.run", lambda *a, **k: None)
    flag = "--table-image-placeholders" if enabled else "--no-table-image-placeholders"
    result = CliRunner().invoke(cli_app, ["serve", "--config", str(config), flag, "--table-image-min-score", "0.75"])
    assert result.exit_code == 0, result.output
    assert runtime_settings.pipeline.table_image_placeholders is enabled
    assert captured["OXR__PIPELINE__TABLE_IMAGE_PLACEHOLDERS"] == str(enabled).lower()
    assert captured["OXR__PIPELINE__TABLE_IMAGE_MIN_SCORE"] == "0.75"


@pytest.mark.parametrize("enabled", [True, False])
def test_serve_full_overlap_flag_overrides_yaml_and_propagates_environment(tmp_path, monkeypatch, enabled):
    from oxr.config.settings import settings as runtime_settings

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "layout_analyze_model:\n  url: http://layout:8081\n"
        "oxr_model:\n  url: http://oxr:8080/v1\n"
        f"pipeline:\n  full_overlap_dedupe: {str(not enabled).lower()}\n",
        encoding="utf-8",
    )
    # Restore the global configuration replaced by serve after the test.
    for field in ("server", "layout_analyze_model", "oxr_model", "pipeline"):
        monkeypatch.setattr(runtime_settings, field, getattr(runtime_settings, field))
    captured = {}
    monkeypatch.setattr("oxr.cli.cmd_serve.os.environ", captured)
    monkeypatch.setattr("oxr.cli.cmd_serve.uvicorn.run", lambda *args, **kwargs: None)
    flag = "--full-overlap-dedupe" if enabled else "--no-full-overlap-dedupe"
    result = CliRunner().invoke(cli_app, ["serve", "--config", str(config_file), flag])
    assert result.exit_code == 0, result.output
    assert runtime_settings.pipeline.full_overlap_dedupe is enabled
    assert captured["OXR__PIPELINE__FULL_OVERLAP_DEDUPE"] == str(enabled).lower()


@pytest.mark.parametrize(
    "field",
    ["systematic_cross_category_dedupe", "visual_parent_dedupe", "complete_text_dedupe"],
)
def test_settings_reject_removed_postprocessing_environment(monkeypatch, field):
    monkeypatch.setenv(f"OXR__PIPELINE__{field.upper()}", "false")

    with pytest.raises(ValidationError, match=field):
        Settings()


@pytest.mark.parametrize(
    "field",
    ["systematic_cross_category_dedupe", "visual_parent_dedupe", "complete_text_dedupe"],
)
def test_settings_reject_removed_postprocessing_yaml(tmp_path, field):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"pipeline:\n  {field}: false\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=field):
        Settings.load_from_yaml(str(config_file))


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_layout_score_threshold_must_be_between_zero_and_one(value):
    with pytest.raises(ValidationError):
        Settings(layout_analyze_model={"score_threshold": value})


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_layout_nms_threshold_must_be_between_zero_and_one(value):
    with pytest.raises(ValidationError):
        Settings(layout_analyze_model={"nms_iou_threshold": value})


@pytest.mark.parametrize(
    "pipeline_key",
    [
        "layout_batch_size",
        "max_layout_concurrency",
        "layout_timeout",
        "max_recognition_concurrency",
        "recognition_timeout",
        "retry_times",
    ],
)
def test_settings_rejects_removed_pipeline_model_bridge_fields(pipeline_key):
    with pytest.raises(ValidationError, match=pipeline_key):
        Settings(pipeline={pipeline_key: 1})


def test_settings_reads_double_underscore_model_environment(monkeypatch):
    monkeypatch.setenv("OXR__LAYOUT_ANALYZE_MODEL__URL", "http://layout:8081")
    monkeypatch.setenv("OXR__OXR_MODEL__URL", "http://oxr:8080/v1")

    settings = Settings()

    assert settings.layout_analyze_model.url == "http://layout:8081"
    assert settings.oxr_model.url == "http://oxr:8080/v1"


def test_settings_use_8k_default_oxr_output_limit():
    assert Settings().oxr_model.max_tokens == 8192


@pytest.mark.parametrize(
    "config",
    [
        {"layout_analyze_model": {"url": "http://layout", "batch_size": 0}},
        {"oxr_model": {"url": "http://oxr/v1", "max_tokens": 0}},
        {"layout_analyze_model": {"url": "not-a-url"}},
    ],
)
def test_model_settings_require_positive_values_and_http_urls(config):
    with pytest.raises(ValidationError):
        Settings(**config)


@pytest.mark.parametrize("legacy_name", ["OXR_MODEL__URL", "OXR__MODEL__URL"])
def test_settings_reject_legacy_model_environment(monkeypatch, legacy_name):
    monkeypatch.setenv(legacy_name, "http://legacy:8080/v1")

    with pytest.raises(ValueError, match="legacy model configuration"):
        Settings()


def test_settings_reject_legacy_model_yaml(tmp_path):
    config_file = tmp_path / "legacy.yaml"
    config_file.write_text("model:\n  url: http://legacy:8080/v1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="layout_analyze_model.*oxr_model"):
        Settings.load_from_yaml(str(config_file))


def test_server_validation_requires_both_model_urls():
    with pytest.raises(ValueError, match="layout_analyze_model.url.*oxr_model.url"):
        Settings().validate_for_server()


def test_serve_uses_namespaced_model_flags():
    options = inspect.signature(serve).parameters
    option_declarations = {
        declaration
        for parameter in options.values()
        for declaration in parameter.default.param_decls
    }

    assert "--layout-analyze-model-url" in option_declarations
    assert "--layout-analyze-model-score-threshold" in option_declarations
    assert "--layout-analyze-model-nms-iou-threshold" in option_declarations
    assert "--oxr-model-url" in option_declarations


def test_serve_propagates_layout_score_threshold(monkeypatch):
    from oxr.config.settings import settings as runtime_settings

    captured_environment = {}
    monkeypatch.setattr(runtime_settings.layout_analyze_model, "url", None)
    monkeypatch.setattr(runtime_settings.layout_analyze_model, "score_threshold", 0.5)
    monkeypatch.setattr(runtime_settings.oxr_model, "url", None)
    monkeypatch.setattr("oxr.cli.cmd_serve.os.environ", captured_environment)
    monkeypatch.setattr("oxr.cli.cmd_serve.uvicorn.run", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        cli_app,
        [
            "serve",
            "--layout-analyze-model-url",
            "http://layout:8081",
            "--layout-analyze-model-score-threshold",
            "0.65",
            "--oxr-model-url",
            "http://oxr:8080/v1",
        ],
    )

    assert result.exit_code == 0
    assert captured_environment["OXR__LAYOUT_ANALYZE_MODEL__SCORE_THRESHOLD"] == "0.65"


def test_serve_propagates_layout_nms_threshold(monkeypatch):
    from oxr.config.settings import settings as runtime_settings

    captured_environment = {}
    monkeypatch.setattr(runtime_settings.layout_analyze_model, "url", None)
    monkeypatch.setattr(runtime_settings.oxr_model, "url", None)
    monkeypatch.setattr("oxr.cli.cmd_serve.os.environ", captured_environment)
    monkeypatch.setattr("oxr.cli.cmd_serve.uvicorn.run", lambda *args, **kwargs: None)

    result = CliRunner().invoke(
        cli_app,
        [
            "serve",
            "--layout-analyze-model-url",
            "http://layout:8081",
            "--layout-analyze-model-nms-iou-threshold",
            "0.55",
            "--oxr-model-url",
            "http://oxr:8080/v1",
        ],
    )

    assert result.exit_code == 0
    assert captured_environment["OXR__LAYOUT_ANALYZE_MODEL__NMS_IOU_THRESHOLD"] == "0.55"


@pytest.mark.parametrize(
    ("legacy_flag", "value", "replacement"),
    [
        ("--model-url", "http://legacy:8080/v1", "--oxr-model-url"),
        ("--model-name", "legacy", "--oxr-model-name"),
        ("--max-tokens", "1024", "--oxr-model-max-tokens"),
        ("--layout-batch-size", "4", "--layout-analyze-model-batch-size"),
        ("--max-layout-concurrency", "2", "--layout-analyze-model-max-concurrency"),
        ("--max-recognition-concurrency", "8", "--oxr-model-max-concurrency"),
    ],
)
def test_serve_legacy_flags_report_explicit_migration_mapping(
    legacy_flag,
    value,
    replacement,
):
    result = CliRunner().invoke(cli_app, ["serve", legacy_flag, value])

    assert result.exit_code == 2
    normalized_output = " ".join(result.output.replace("│", "").split())
    assert f"{legacy_flag} is no longer supported; use" in normalized_output
    assert replacement in normalized_output


def test_serve_help_hides_legacy_flags():
    result = CliRunner().invoke(cli_app, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--model-url" not in result.output
    assert "--model-name" not in result.output
    assert "--max-tokens" not in result.output
    assert "--layout-batch-size" not in result.output
    assert "--max-layout-concurrency" not in result.output
    assert "--max-recognition-concurrency" not in result.output


def test_settings_reads_debug_from_plain_environment(monkeypatch):
    monkeypatch.setenv("DEBUG", "yes")
    assert Settings().debug is True

    monkeypatch.setenv("DEBUG", "false")
    assert Settings().debug is False

    monkeypatch.delenv("DEBUG")
    assert Settings().debug is False


def test_workers_must_be_single_process():
    with pytest.raises(typer.BadParameter):
        _ensure_supported_workers(2)
