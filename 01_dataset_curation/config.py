"""Configuration loading and per-stage settings.

The package is fully self-contained: config.yaml is bundled inside the
package (``video_pipeline/config.yaml``) and all relative paths in it are
resolved against the folder containing the config file actually used
(the "workspace root"). No assumptions are made about any parent project.

Config resolution order:
  1. ``--config <path>`` CLI argument / ``VIDEO_PIPELINE_CONFIG`` env var
  2. ``video_pipeline/config.yaml`` (bundled default)

The active config is recorded (``set_active_config``) so modules that load
``.env`` or resolve relative paths without an explicit path always use the
correct workspace. API keys live in .env files: the workspace ``.env`` is
loaded first, then the package ``.env`` (workspace values win).
"""

import os
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent

# The config file most recently selected (via build_settings / CLI / env).
# Used so that .env and relative paths resolve to the right workspace even
# when called from other modules without an explicit path.
_active_config = None


def find_config(config_path=None):
    """Resolve the config file path (CLI/env override, then bundled default)."""
    if config_path:
        p = Path(config_path)
        return str(p if p.is_absolute() else Path.cwd() / p)
    env_config = os.getenv("VIDEO_PIPELINE_CONFIG")
    if env_config:
        return env_config
    return str(PACKAGE_ROOT / "config.yaml")


def set_active_config(config_path=None):
    """Record the active config file (used by load_env / resolve_path)."""
    global _active_config
    _active_config = find_config(config_path)


def get_active_config():
    return _active_config or find_config(None)


def load_config(config_path=None):
    """Load the YAML config. Path resolution: explicit > env > bundled."""
    config_path = find_config(config_path)
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_workspace_root(config_path=None):
    """The folder containing the active config file — the base for all
    relative paths in the pipeline. Returns a str."""
    return str(Path(get_active_config() if config_path is None else find_config(config_path)).resolve().parent)


def load_env(config_path=None):
    """Load .env files: workspace .env first, then the bundled package .env.
    Workspace values take precedence."""
    load_dotenv(os.path.join(get_workspace_root(config_path), ".env"))
    load_dotenv(PACKAGE_ROOT / ".env")


def resolve_path(raw_path, config_path=None):
    """Resolve a config path relative to the workspace root if not absolute."""
    p = Path(raw_path)
    return str(p if p.is_absolute() else get_workspace_root(config_path) / p)


def load_video_links(file_path, config_path=None):
    file_path = resolve_path(file_path, config_path)
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Video links file not found at {file_path}")
        return []
    with open(file_path, 'r') as f:
        links = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"ℹ️ Loaded {len(links)} URL(s) from {file_path}")
    return links


@dataclass
class DownloadSettings:
    max_duration_seconds: int = 1500


@dataclass
class WhisperSettings:
    model_size: str = "large"
    device: str = "cuda"


@dataclass
class DownsampleSettings:
    target_fps: int = 10
    input_dir: str = ""
    output_dir: str = ""


@dataclass
class LLMSettings:
    """A text-only LLM stage: OpenAI-compatible base_url + model + env key."""
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class VlmSettings:
    gatekeeper: LLMSettings = field(default_factory=LLMSettings)
    analyst_model: str = "gemini-2.5-pro"
    analyst_api_key_env: str = "GEMINI_API_KEY"
    aggregate_file: str = "vlm_dataset_all.jsonl"
    log_file: str = "process_log.csv"


@dataclass
class PhaseSftSettings:
    """Settings for grounded SFT generation from the phase dataset."""
    source_dir: str
    manifest_file: str
    output_dir: str
    log_file: str
    teacher: LLMSettings
    critic: LLMSettings
    max_attempts: int = 2
    max_retries: int = 3
    retry_delay_seconds: float = 3.0
    request_delay_seconds: float = 0.0
    max_video_bytes: int = 0
    fallback_max_frames: int = 32
    frame_quality: int = 90
    materialization: str = "hardlink"
    hardlink_fallback_copy: bool = True
    video_part_type: str = "video_url"


@dataclass
class PhaseGrpoSettings:
    """Settings for deterministic GRPO generation from accepted phase SFT clips."""
    input_dir: str
    log_file: str


@dataclass
class DatasetSettings:
    """Runtime view of config with resolved absolute paths."""
    videos_dir: str
    videos_down_dir: str
    audio_dir: str
    transcripts_dir: str
    refined_dir: str
    vlm_dir: str
    adverse_events_dir: str
    clips_dir: str
    dataset_dir: str
    dataset_sft_dir: str
    dataset_grpo_dir: str
    splitter_videos_dir: str
    metadata_file: str
    cookies_file: str
    refinement_log: str
    links_file: str
    summary_file: str
    download: DownloadSettings
    whisper: WhisperSettings
    downsample: DownsampleSettings
    refiner: LLMSettings
    vlm: VlmSettings
    adverse_event: LLMSettings
    splitter_log: str
    dataset_formatter: LLMSettings
    refine_instrument_options: LLMSettings
    generate_missing_grpo_sft: LLMSettings
    formatter_output_dir: str
    formatter_split_file: str
    formatter_log: str
    formatter_max_retries: int
    phase_sft: PhaseSftSettings
    phase_grpo: PhaseGrpoSettings
    config_path: str = ""

    def llm_for(self, name):
        return {"refiner": self.refiner,
                "adverse_event": self.adverse_event,
                "dataset_formatter": self.dataset_formatter,
                "refine_instrument_options": self.refine_instrument_options,
                "generate_missing_grpo_sft": self.generate_missing_grpo_sft}[name]


def _parse_llm_section(section, api_key_env, defaults):
    section = section or {}
    base_url = section.get("base_url") or defaults.get("base_url") or ""
    model = section.get("model") or defaults.get("model") or ""
    env = section.get("api_key_env") or api_key_env
    extra = {k: v for k, v in section.items() if k not in ("base_url", "model", "api_key_env")}
    return LLMSettings(base_url=base_url, model=model, api_key_env=env, extra=extra)


def build_settings(config=None, config_path=None):
    config = config or load_config(config_path)
    set_active_config(config_path)
    dirs = config.get("directories", {})

    videos_dir = resolve_path(dirs.get("videos", "./videos"), config_path)
    videos_down_dir = resolve_path(dirs.get("videos_down", "./videos_down"), config_path)
    clips_dir = resolve_path(dirs.get("clips", "./clip_dataset"), config_path)
    dataset_dir = resolve_path(dirs.get("dataset", "./dataset"), config_path)
    # Separated SFT/GRPO datasets (mirror dataset with videos in both; defaults next to dataset)
    dataset_sft_dir = resolve_path(dirs.get("dataset_sft", "./dataset_sft"), config_path)
    dataset_grpo_dir = resolve_path(dirs.get("dataset_grpo", "./dataset_grpo"), config_path)
    splitter_videos_dir = resolve_path(config.get("splitter", {}).get(
        "videos_dir", videos_down_dir), config_path)

    s = DatasetSettings(
        config_path=find_config(config_path),
        videos_dir=videos_dir,
        videos_down_dir=videos_down_dir,
        audio_dir=resolve_path(dirs.get("audio", "./audio"), config_path),
        transcripts_dir=resolve_path(dirs.get("transcripts", "./transcripts"), config_path),
        refined_dir=resolve_path(dirs.get("refined_transcripts", "./refined_transcripts"), config_path),
        vlm_dir=resolve_path(dirs.get("vlm_dataset", "./vlm_dataset"), config_path),
        adverse_events_dir=resolve_path(dirs.get("adverse_events", "./adverse_events"), config_path),
        clips_dir=clips_dir,
        dataset_dir=dataset_dir,
        dataset_sft_dir=dataset_sft_dir,
        dataset_grpo_dir=dataset_grpo_dir,
        splitter_videos_dir=splitter_videos_dir,
        metadata_file=resolve_path(os.path.join(videos_dir,
                                                config.get("files", {}).get("metadata_csv", "video_metadata.csv")),
                                   config_path),
        cookies_file=resolve_path(config.get("files", {}).get("cookies", "cookie.txt"), config_path),
        refinement_log=resolve_path(os.path.join(
            dirs.get("refined_transcripts", "./refined_transcripts"),
            config.get("files", {}).get("refinement_log", "refinement_log.csv")), config_path),
        links_file=config.get("files", {}).get("video_links", "youtube_links.txt"),
        summary_file=resolve_path(config.get("files", {}).get("dataset_summary", "dataset_info.csv"), config_path),
        download=DownloadSettings(**config.get("download", {})),
        whisper=WhisperSettings(**config.get("whisper", {})),
        downsample=DownsampleSettings(
            target_fps=config.get("downsample", {}).get("target_fps", 10),
            input_dir=resolve_path(config.get("downsample", {}).get("input_dir", videos_dir), config_path),
            output_dir=resolve_path(config.get("downsample", {}).get("output_dir", videos_down_dir), config_path),
        ),
        refiner=_parse_llm_section(config.get("refiner", {}), "REFINER_API_KEY",
                                   {"base_url": "https://api.cerebras.ai/v1"}),
        vlm=_parse_vlm(config.get("vlm", {})),
        adverse_event=_parse_llm_section(config.get("adverse_event", {}), "ADVERSE_EVENT_API_KEY", {}),
        splitter_log=resolve_path(os.path.join(
            clips_dir,
            config.get("splitter", {}).get("log_file", "split_log.csv")), config_path),
        dataset_formatter=_parse_llm_section(config.get("dataset_formatter", {}), "FORMATTER_API_KEY", {}),
        refine_instrument_options=_parse_llm_section(
            config.get("refine_instrument_options", {}), "FORMATTER_API_KEY",
            {"base_url": "https://api.gapgpt.app/v1", "model": "deepseek-v4-flash"}),
        generate_missing_grpo_sft=_parse_llm_section(
            config.get("generate_missing_grpo_sft", {}), "FORMATTER_API_KEY",
            {"base_url": "https://api.gapgpt.app/v1", "model": "gemini-3-flash-preview"}),
        formatter_output_dir=resolve_path(config.get("dataset_formatter", {}).get("output_dir", "./training_datasets"),
                                          config_path),
        formatter_split_file=config.get("dataset_formatter", {}).get("split_file", "dataset_split.json"),
        formatter_log=config.get("dataset_formatter", {}).get("log_file", "dataset_formatter_log.csv"),
        formatter_max_retries=config.get("dataset_formatter", {}).get("max_retries", 3),
        phase_sft=_parse_phase_sft(config.get("phase_sft", {}), config_path),
        phase_grpo=_parse_phase_grpo(config.get("phase_grpo", {}), config_path),
    )
    return s


def _parse_vlm(vlm_section):
    vlm_section = vlm_section or {}
    gatekeeper = _parse_llm_section(vlm_section.get("gatekeeper", {}), "VLM_GATEKEEPER_API_KEY", {})
    analyst = vlm_section.get("analyst", {}) or {}
    return VlmSettings(
        gatekeeper=gatekeeper,
        analyst_model=analyst.get("model", "gemini-2.5-pro"),
        analyst_api_key_env=analyst.get("api_key_env", "GEMINI_API_KEY"),
        aggregate_file=vlm_section.get("aggregate_file", "vlm_dataset_all.jsonl"),
        log_file=vlm_section.get("log_file", "process_log.csv"),
    )


def _parse_phase_sft(section, config_path=None):
    """Parse and resolve the phase-dataset SFT generation settings."""
    section = section or {}
    teacher = _parse_llm_section(
        section.get("teacher", {}),
        "PHASE_SFT_TEACHER_API_KEY",
        {"base_url": "http://localhost:8000/v1", "model": "qwen3.8-max"},
    )
    critic = _parse_llm_section(
        section.get("critic", {}),
        "PHASE_SFT_CRITIC_API_KEY",
        {"base_url": "http://localhost:8000/v1", "model": "qwen3.8-max"},
    )
    output_dir = resolve_path(section.get("output_dir", "./phase_vlm_dataset"), config_path)
    log_file = section.get("log_file", "phase_sft_log.csv")
    return PhaseSftSettings(
        source_dir=resolve_path(section.get("source_dir", "./phase_dataset"), config_path),
        manifest_file=resolve_path(
            section.get("manifest_file", "./phase_dataset/manifest.csv"), config_path),
        output_dir=output_dir,
        log_file=resolve_path(os.path.join(output_dir, log_file), config_path),
        teacher=teacher,
        critic=critic,
        max_attempts=int(section.get("max_attempts", 2)),
        max_retries=int(section.get("max_retries", 3)),
        retry_delay_seconds=float(section.get("retry_delay_seconds", 3.0)),
        request_delay_seconds=float(section.get("request_delay_seconds", 0.0)),
        max_video_bytes=int(section.get("max_video_bytes", 0)),
        fallback_max_frames=int(section.get("fallback_max_frames", 32)),
        frame_quality=int(section.get("frame_quality", 90)),
        materialization=section.get("materialization", "hardlink"),
        hardlink_fallback_copy=bool(section.get("hardlink_fallback_copy", True)),
        video_part_type=section.get("video_part_type", "video_url"),
    )


def _parse_phase_grpo(section, config_path=None):
    """Parse the deterministic phase-GRPO generation settings."""
    section = section or {}
    input_dir = resolve_path(
        section.get("input_dir", "./phase_vlm_dataset"), config_path)
    log_file = section.get("log_file", "phase_grpo_log.csv")
    return PhaseGrpoSettings(
        input_dir=input_dir,
        log_file=resolve_path(os.path.join(input_dir, log_file), config_path),
    )
