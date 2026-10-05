# Model directory

Use `layout_analysis/last.pth` and `layout_analysis/metadata.json` from the
unified `oxr-model` repository. Both files are required for startup.

The checkpoint is intentionally not included in this source distribution.
Set `OXR_MODEL_DIR` to the absolute model repository root before starting
Compose. It mounts the repository read-only; no checkpoint copy is needed.
Run `git lfs pull` in the model checkout first. The container does not download
weights automatically.

The service reads the model version, supported configuration and checkpoint
size/SHA-256 from `metadata.json` and verifies them before loading weights.
An explicit `MODEL_METADATA_PATH` can override the metadata path.
