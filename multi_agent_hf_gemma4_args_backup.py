from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    AutoModelForCausalLM = None
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    TRANSFORMERS_IMPORT_ERROR = exc


# -----------------------------
# Utilities
# -----------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ns() -> int:
    return time.perf_counter_ns()


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, default=str)


def serialized_bytes(obj: Any) -> int:
    return len(safe_json_dumps(obj).encode("utf-8"))


def serialized_bits(obj: Any) -> int:
    return serialized_bytes(obj) * 8


def preview(obj: Any, max_len: int = 240) -> str:
    text = safe_json_dumps(obj) if not isinstance(obj, str) else obj
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def sha256_text(obj: Any) -> str:
    return hashlib.sha256(safe_json_dumps(obj).encode("utf-8")).hexdigest()


def coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return safe_json_dumps(content)


def make_chat_prompt(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user").upper()
        content = coerce_text_content(m.get("content", ""))
        lines.append(f"[{role}]\n{content}")
    lines.append("[ASSISTANT]\n")
    return "\n\n".join(lines)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def copy_hf_model_config(cfg: "HFModelConfig") -> "HFModelConfig":
    return HFModelConfig(**asdict(cfg))


def copy_runtime_config(cfg: "RuntimeConfig") -> "RuntimeConfig":
    return RuntimeConfig(**asdict(cfg))


def deep_update_dataclass(default_obj: Any, raw: Dict[str, Any]) -> Any:
    merged = asdict(default_obj)
    merged.update(raw)
    return type(default_obj)(**merged)


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# -----------------------------
# Trace schema
# -----------------------------


@dataclass
class MessageRecord:
    message_id: str
    run_id: str
    created_at: str
    sender: str
    receiver: str
    message_type: str
    content: Any
    parent_message_id: Optional[str] = None
    caused_by_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_preview: str = ""
    content_sha256: str = ""
    serialized_bytes: int = 0
    serialized_bits: int = 0


@dataclass
class LLMCallRecord:
    call_id: str
    run_id: str
    step_id: str
    agent_name: str
    provider_name: str
    model: str
    request_started_at: str
    request_finished_at: str
    request_started_ns: int
    request_finished_ns: int
    duration_ms: float
    success: bool
    http_status: Optional[int]
    error: Optional[str]
    request_payload: Dict[str, Any]
    response_payload: Optional[Dict[str, Any]]
    request_payload_bytes: int
    request_payload_bits: int
    response_payload_bytes: Optional[int]
    response_payload_bits: Optional[int]
    response_created_unix: Optional[int]
    output_content: Any
    output_preview: str
    output_bytes: int
    output_bits: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    output_tokens_per_sec: Optional[float]


@dataclass
class AgentStepRecord:
    step_id: str
    run_id: str
    agent_name: str
    role_name: str
    input_message_id: str
    output_message_id: Optional[str]
    llm_call_id: Optional[str]
    started_at: str
    finished_at: str
    started_ns: int
    finished_ns: int
    duration_ms: float
    input_message_age_ms: float
    success: bool
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_id: str
    task_id: str
    task: Any
    started_at: str
    finished_at: Optional[str] = None
    started_ns: int = 0
    finished_ns: Optional[int] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    total_messages: int = 0
    total_message_bits: int = 0
    total_llm_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_steps: int = 0


class TraceStore:
    def __init__(self, task: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.run_id = uuid.uuid4().hex
        self.config = config or {}
        self.messages: List[MessageRecord] = []
        self.llm_calls: List[LLMCallRecord] = []
        self.agent_steps: List[AgentStepRecord] = []
        self.summary = RunSummary(
            run_id=self.run_id,
            task_id=uuid.uuid4().hex,
            task=task,
            started_at=utc_now_iso(),
            started_ns=monotonic_ns(),
        )

    def log_message(
        self,
        sender: str,
        receiver: str,
        content: Any,
        message_type: str,
        parent_message_id: Optional[str] = None,
        caused_by_call_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        record = MessageRecord(
            message_id=uuid.uuid4().hex,
            run_id=self.run_id,
            created_at=utc_now_iso(),
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            content=content,
            parent_message_id=parent_message_id,
            caused_by_call_id=caused_by_call_id,
            metadata=metadata or {},
            content_preview=preview(content),
            content_sha256=sha256_text(content),
            serialized_bytes=serialized_bytes(content),
            serialized_bits=serialized_bits(content),
        )
        self.messages.append(record)
        self.summary.total_messages += 1
        self.summary.total_message_bits += record.serialized_bits
        return record

    def log_llm_call(self, record: LLMCallRecord) -> None:
        self.llm_calls.append(record)
        self.summary.total_llm_calls += 1
        self.summary.total_prompt_tokens += record.prompt_tokens or 0
        self.summary.total_completion_tokens += record.completion_tokens or 0
        self.summary.total_tokens += record.total_tokens or 0

    def log_agent_step(self, record: AgentStepRecord) -> None:
        self.agent_steps.append(record)
        self.summary.total_steps += 1

    def finish(self, success: bool) -> None:
        self.summary.finished_at = utc_now_iso()
        self.summary.finished_ns = monotonic_ns()
        self.summary.duration_ms = (
            (self.summary.finished_ns - self.summary.started_ns) / 1_000_000.0
        )
        self.summary.success = success

    def export(self, output_dir: str | Path) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        with (output / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {"summary": asdict(self.summary), "config": self.config},
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        self._write_jsonl(output / "messages.jsonl", self.messages)
        self._write_jsonl(output / "llm_calls.jsonl", self.llm_calls)
        self._write_jsonl(output / "agent_steps.jsonl", self.agent_steps)
        self._write_csv(output / "messages.csv", self.messages)
        self._write_csv(output / "llm_calls.csv", self.llm_calls)
        self._write_csv(output / "agent_steps.csv", self.agent_steps)
        return output

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(safe_json_dumps(asdict(record)))
                f.write("\n")

    @staticmethod
    def _write_csv(path: Path, records: Iterable[Any]) -> None:
        rows = []
        for record in records:
            row = {}
            for k, v in asdict(record).items():
                row[k] = safe_json_dumps(v) if isinstance(v, (dict, list)) else v
            rows.append(row)

        if not rows:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["empty"])
            return

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


# -----------------------------
# Config schema
# -----------------------------


@dataclass
class HFModelConfig:
    provider_name: str
    model_id: str
    prompt_model_id: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    do_sample: bool = True
    repetition_penalty: float = 1.0
    use_chat_template: bool = True
    use_auto_processor: bool = False
    enable_thinking: bool = False
    attn_implementation: Optional[str] = None


@dataclass
class RuntimeConfig:
    sequential_model_loading: bool = False
    unload_model_after_call: bool = False
    clear_cuda_cache_after_call: bool = True
    expected_vram_gb: Optional[int] = None
    strict_same_prompt_model_id: bool = False


@dataclass
class PipelineConfig:
    planner: HFModelConfig
    coder: HFModelConfig
    vision: HFModelConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_dir: str = "./trace_output"
    need_code: bool = True
    need_vision: bool = True
    preset_name: Optional[str] = None
    notes: List[str] = field(default_factory=list)


DEFAULT_CONFIG = PipelineConfig(
    planner=HFModelConfig(
        provider_name="hf-planner",
        model_id="Qwen/Qwen2.5-7B-Instruct",
    ),
    coder=HFModelConfig(
        provider_name="hf-coder",
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    vision=HFModelConfig(
        provider_name="hf-vision",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=256,
        temperature=0.1,
    ),
)


def build_preset_config(preset_name: str) -> PipelineConfig:
    name = preset_name.strip().lower()
    if name == "gemma4_24gb_strict_vocab":
        shared_prompt_model = "google/gemma-4-E2B-it"
        base = dict(
            device_map="auto",
            torch_dtype="bfloat16",
            load_in_4bit=True,
            load_in_8bit=False,
            trust_remote_code=False,
            use_chat_template=True,
            use_auto_processor=True,
            enable_thinking=False,
            attn_implementation=None,
        )
        return PipelineConfig(
            planner=HFModelConfig(
                provider_name="hf-planner",
                model_id="google/gemma-4-E4B-it",
                prompt_model_id=shared_prompt_model,
                max_new_tokens=640,
                temperature=0.15,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.02,
                **base,
            ),
            coder=HFModelConfig(
                provider_name="hf-coder",
                model_id="google/gemma-4-E2B-it",
                prompt_model_id=shared_prompt_model,
                max_new_tokens=768,
                temperature=0.10,
                top_p=0.90,
                do_sample=True,
                repetition_penalty=1.02,
                **base,
            ),
            vision=HFModelConfig(
                provider_name="hf-vision",
                model_id="google/gemma-4-E2B-it",
                prompt_model_id=shared_prompt_model,
                max_new_tokens=192,
                temperature=0.05,
                top_p=0.90,
                do_sample=True,
                repetition_penalty=1.0,
                **base,
            ),
            runtime=RuntimeConfig(
                sequential_model_loading=True,
                unload_model_after_call=True,
                clear_cuda_cache_after_call=True,
                expected_vram_gb=24,
                strict_same_prompt_model_id=True,
            ),
            output_dir="./trace_output_gemma4_24gb",
            need_code=True,
            need_vision=True,
            preset_name="gemma4_24gb_strict_vocab",
            notes=[
                "24GB VRAM용 보수적 프리셋.",
                "메인 플래너는 Gemma 4 E4B, 서브에이전트는 Gemma 4 E2B로 맞춤.",
                "Gemma 4 계열 안에서 1B급 모델이 없어 strict same-vocab 조건을 우선해 E2B를 사용.",
                "prompt_model_id를 통일해서 동일한 프롬프트/토크나이저 소스를 사용.",
                "모델은 순차 로딩 + 호출 후 언로드로 peak VRAM을 낮춤.",
            ],
        )

    if name == "gemma4_24gb_near_1b_compromise":
        return PipelineConfig(
            planner=HFModelConfig(
                provider_name="hf-planner",
                model_id="google/gemma-4-E4B-it",
                prompt_model_id="google/gemma-4-E4B-it",
                device_map="auto",
                torch_dtype="bfloat16",
                load_in_4bit=True,
                max_new_tokens=640,
                temperature=0.15,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.02,
                use_chat_template=True,
                use_auto_processor=True,
                enable_thinking=False,
            ),
            coder=HFModelConfig(
                provider_name="hf-coder",
                model_id="google/gemma-3-1b-it",
                prompt_model_id="google/gemma-3-1b-it",
                device_map="auto",
                torch_dtype="bfloat16",
                load_in_4bit=True,
                max_new_tokens=768,
                temperature=0.10,
                top_p=0.90,
                do_sample=True,
                repetition_penalty=1.02,
                use_chat_template=True,
                use_auto_processor=False,
                enable_thinking=False,
            ),
            vision=HFModelConfig(
                provider_name="hf-vision",
                model_id="google/gemma-3-1b-it",
                prompt_model_id="google/gemma-3-1b-it",
                device_map="auto",
                torch_dtype="bfloat16",
                load_in_4bit=True,
                max_new_tokens=192,
                temperature=0.05,
                top_p=0.90,
                do_sample=True,
                repetition_penalty=1.0,
                use_chat_template=True,
                use_auto_processor=False,
                enable_thinking=False,
            ),
            runtime=RuntimeConfig(
                sequential_model_loading=True,
                unload_model_after_call=True,
                clear_cuda_cache_after_call=True,
                expected_vram_gb=24,
                strict_same_prompt_model_id=False,
            ),
            output_dir="./trace_output_gemma4_24gb_compromise",
            need_code=True,
            need_vision=True,
            preset_name="gemma4_24gb_near_1b_compromise",
            notes=[
                "24GB VRAM용 절충 프리셋.",
                "메인 플래너는 Gemma 4 E4B, 서브에이전트는 Gemma 3 1B로 맞춤.",
                "subagent를 1B 수준으로 낮추지만 strict same-vocab 보장은 포기한 구성.",
                "Gemma 3 1B와 Gemma 4 E2B는 둘 다 262144 vocab_size 계열이지만 tokenizer source를 완전히 동일하게 강제하지는 않음.",
            ],
        )

    raise ValueError(f"Unknown preset: {preset_name}")


def load_pipeline_config(
    config_path: Optional[str],
    base_config: Optional[PipelineConfig] = None,
) -> PipelineConfig:
    base = base_config or DEFAULT_CONFIG
    if not config_path:
        return base

    raw = read_json(config_path)
    planner = deep_update_dataclass(base.planner, raw.get("planner", {}))
    coder = deep_update_dataclass(base.coder, raw.get("coder", {}))
    vision = deep_update_dataclass(base.vision, raw.get("vision", {}))
    runtime = deep_update_dataclass(base.runtime, raw.get("runtime", {}))

    notes = raw.get("notes", list(base.notes))

    return PipelineConfig(
        planner=planner,
        coder=coder,
        vision=vision,
        runtime=runtime,
        output_dir=raw.get("output_dir", base.output_dir),
        need_code=raw.get("need_code", base.need_code),
        need_vision=raw.get("need_vision", base.need_vision),
        preset_name=raw.get("preset_name", base.preset_name),
        notes=notes,
    )


def apply_cli_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    planner = copy_hf_model_config(cfg.planner)
    coder = copy_hf_model_config(cfg.coder)
    vision = copy_hf_model_config(cfg.vision)
    runtime = copy_runtime_config(cfg.runtime)

    if args.planner_model:
        planner.model_id = args.planner_model
    if args.coder_model:
        coder.model_id = args.coder_model
    if args.vision_model:
        vision.model_id = args.vision_model

    if args.planner_prompt_model:
        planner.prompt_model_id = args.planner_prompt_model
    if args.coder_prompt_model:
        coder.prompt_model_id = args.coder_prompt_model
    if args.vision_prompt_model:
        vision.prompt_model_id = args.vision_prompt_model

    if args.planner_max_new_tokens is not None:
        planner.max_new_tokens = args.planner_max_new_tokens
    if args.coder_max_new_tokens is not None:
        coder.max_new_tokens = args.coder_max_new_tokens
    if args.vision_max_new_tokens is not None:
        vision.max_new_tokens = args.vision_max_new_tokens

    if args.planner_load_in_4bit is not None:
        planner.load_in_4bit = args.planner_load_in_4bit
    if args.coder_load_in_4bit is not None:
        coder.load_in_4bit = args.coder_load_in_4bit
    if args.vision_load_in_4bit is not None:
        vision.load_in_4bit = args.vision_load_in_4bit

    if args.planner_torch_dtype:
        planner.torch_dtype = args.planner_torch_dtype
    if args.coder_torch_dtype:
        coder.torch_dtype = args.coder_torch_dtype
    if args.vision_torch_dtype:
        vision.torch_dtype = args.vision_torch_dtype

    if args.sequential_model_loading is not None:
        runtime.sequential_model_loading = args.sequential_model_loading
    if args.unload_model_after_call is not None:
        runtime.unload_model_after_call = args.unload_model_after_call
    if args.clear_cuda_cache_after_call is not None:
        runtime.clear_cuda_cache_after_call = args.clear_cuda_cache_after_call

    return PipelineConfig(
        planner=planner,
        coder=coder,
        vision=vision,
        runtime=runtime,
        output_dir=args.output_dir or cfg.output_dir,
        need_code=cfg.need_code if args.no_code is False else False,
        need_vision=cfg.need_vision if args.no_vision is False else False,
        preset_name=cfg.preset_name,
        notes=list(cfg.notes),
    )


def effective_prompt_model_id(cfg: HFModelConfig) -> str:
    return cfg.prompt_model_id or cfg.model_id


def validate_pipeline_config(cfg: PipelineConfig) -> None:
    if cfg.planner.load_in_4bit and cfg.planner.load_in_8bit:
        raise ValueError("planner cannot enable both 4bit and 8bit at the same time")
    if cfg.coder.load_in_4bit and cfg.coder.load_in_8bit:
        raise ValueError("coder cannot enable both 4bit and 8bit at the same time")
    if cfg.vision.load_in_4bit and cfg.vision.load_in_8bit:
        raise ValueError("vision cannot enable both 4bit and 8bit at the same time")

    if cfg.runtime.strict_same_prompt_model_id:
        ids = {effective_prompt_model_id(cfg.planner)}
        if cfg.need_code:
            ids.add(effective_prompt_model_id(cfg.coder))
        if cfg.need_vision:
            ids.add(effective_prompt_model_id(cfg.vision))
        if len(ids) != 1:
            raise ValueError(
                "strict_same_prompt_model_id=True 인데 에이전트별 prompt_model_id가 서로 다릅니다."
            )


def build_example_config_file(path: str | Path, cfg: PipelineConfig) -> Path:
    out = Path(path)
    out.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# -----------------------------
# Shared text asset cache
# -----------------------------


_TEXT_ASSET_CACHE: Dict[Tuple[str, bool, bool], Any] = {}


def get_text_asset(model_id: str, use_auto_processor: bool, trust_remote_code: bool) -> Any:
    if TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "transformers is required for model loading. Install with: pip install -U transformers accelerate bitsandbytes"
        ) from TRANSFORMERS_IMPORT_ERROR
    key = (model_id, use_auto_processor, trust_remote_code)
    if key in _TEXT_ASSET_CACHE:
        return _TEXT_ASSET_CACHE[key]

    asset = None
    if use_auto_processor:
        try:
            asset = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            asset = None

    if asset is None:
        asset = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

    tokenizer = asset.tokenizer if hasattr(asset, "tokenizer") and getattr(asset, "tokenizer") is not None else asset
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    _TEXT_ASSET_CACHE[key] = asset
    return asset


# -----------------------------
# Hugging Face local client
# -----------------------------


class HuggingFaceLocalClient:
    def __init__(self, cfg: HFModelConfig, runtime: RuntimeConfig) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self._model = None
        self._text_asset = None
        if not runtime.sequential_model_loading:
            self._ensure_loaded()

    @property
    def model(self) -> Any:
        if self._model is None:
            self._ensure_loaded()
        return self._model

    @property
    def text_asset(self) -> Any:
        if self._text_asset is None:
            source_model_id = effective_prompt_model_id(self.cfg)
            self._text_asset = get_text_asset(
                model_id=source_model_id,
                use_auto_processor=self.cfg.use_auto_processor,
                trust_remote_code=self.cfg.trust_remote_code,
            )
        return self._text_asset

    @property
    def tokenizer(self) -> Any:
        asset = self.text_asset
        if hasattr(asset, "tokenizer") and getattr(asset, "tokenizer") is not None:
            return asset.tokenizer
        return asset

    def _ensure_loaded(self) -> None:
        _ = self.text_asset
        if self._model is not None:
            return

        quantization_config = None
        if self.cfg.load_in_4bit or self.cfg.load_in_8bit:
            quantization_kwargs: Dict[str, Any] = {
                "load_in_4bit": self.cfg.load_in_4bit,
                "load_in_8bit": self.cfg.load_in_8bit,
            }
            if self.cfg.load_in_4bit:
                quantization_kwargs["bnb_4bit_quant_type"] = "nf4"
                quantization_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
            quantization_config = BitsAndBytesConfig(**quantization_kwargs)

        if self.cfg.torch_dtype == "auto":
            torch_dtype: Any = "auto"
        elif hasattr(torch, self.cfg.torch_dtype):
            torch_dtype = getattr(torch, self.cfg.torch_dtype)
        else:
            raise ValueError(f"Unsupported torch dtype: {self.cfg.torch_dtype}")

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.cfg.trust_remote_code,
            "device_map": self.cfg.device_map,
            "torch_dtype": torch_dtype,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if self.cfg.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, **model_kwargs)

    def release_model(self) -> None:
        if self._model is not None:
            model = self._model
            self._model = None
            del model
            if self.runtime.clear_cuda_cache_after_call:
                cuda_cleanup()

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for m in messages:
            normalized.append(
                {
                    "role": str(m.get("role", "user")),
                    "content": coerce_text_content(m.get("content", "")),
                }
            )
        return normalized

    def _render_prompt(self, messages: List[Dict[str, Any]]) -> str:
        normalized_messages = self._normalize_messages(messages)
        asset = self.text_asset
        if self.cfg.use_chat_template and hasattr(asset, "apply_chat_template"):
            try:
                return asset.apply_chat_template(
                    normalized_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.cfg.enable_thinking,
                )
            except TypeError:
                try:
                    return asset.apply_chat_template(
                        normalized_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            except Exception:
                pass
        return make_chat_prompt(normalized_messages)

    def _encode_prompt(self, prompt_text: str) -> Dict[str, Any]:
        asset = self.text_asset
        try:
            model_inputs = asset(text=prompt_text, return_tensors="pt")
        except TypeError:
            model_inputs = asset(prompt_text, return_tensors="pt")
        return dict(model_inputs)

    def _decode(self, ids: Any) -> str:
        asset = self.text_asset
        if hasattr(asset, "decode"):
            try:
                return asset.decode(ids, skip_special_tokens=True)
            except TypeError:
                return asset.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _infer_model_device(self) -> torch.device:
        try:
            return self.model.device
        except Exception:
            pass

        try:
            param = next(self.model.parameters())
            return param.device
        except Exception:
            pass

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for value in device_map.values():
                if isinstance(value, str) and value not in {"cpu", "disk"}:
                    return torch.device(value)
                if isinstance(value, int):
                    return torch.device(f"cuda:{value}")
        return torch.device("cpu")

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        request_payload: Dict[str, Any] = {
            "model": self.cfg.model_id,
            "prompt_model_id": effective_prompt_model_id(self.cfg),
            "messages": self._normalize_messages(messages),
            "generation_config": {
                "max_new_tokens": self.cfg.max_new_tokens,
                "temperature": self.cfg.temperature,
                "top_p": self.cfg.top_p,
                "do_sample": self.cfg.do_sample,
                "repetition_penalty": self.cfg.repetition_penalty,
            },
        }

        try:
            if self.runtime.sequential_model_loading:
                self._ensure_loaded()

            prompt_text = self._render_prompt(messages)
            model_inputs = self._encode_prompt(prompt_text)
            device = self._infer_model_device()
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            prompt_len = int(model_inputs["input_ids"].shape[-1])

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    do_sample=self.cfg.do_sample,
                    repetition_penalty=self.cfg.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][prompt_len:]
            output_text = self._decode(generated_ids)
            completion_tokens = int(generated_ids.shape[-1])
            total_tokens = prompt_len + completion_tokens

            response_json = {
                "id": f"hf-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.cfg.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            return {
                "ok": True,
                "status": 200,
                "request_payload": request_payload,
                "response_json": response_json,
            }
        except Exception as e:
            return {
                "ok": False,
                "status": None,
                "request_payload": request_payload,
                "error": repr(e),
            }
        finally:
            if self.runtime.unload_model_after_call:
                self.release_model()


# -----------------------------
# Agent
# -----------------------------


@dataclass
class AgentSpec:
    name: str
    role_name: str
    system_prompt: str
    client: HuggingFaceLocalClient


class Agent:
    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    def act(
        self,
        trace: TraceStore,
        input_message: MessageRecord,
        reply_to: str,
        extra_messages: Optional[List[MessageRecord]] = None,
        output_message_type: str = "agent_reply",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MessageRecord:
        step_id = uuid.uuid4().hex
        started_at = utc_now_iso()
        started_ns = monotonic_ns()

        input_created_dt = datetime.fromisoformat(input_message.created_at)
        now_dt = datetime.fromisoformat(started_at)
        input_age_ms = (now_dt - input_created_dt).total_seconds() * 1000.0

        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": self.spec.system_prompt}
        ]
        if extra_messages:
            for msg in extra_messages:
                conversation.append(
                    {
                        "role": "user",
                        "content": {
                            "from": msg.sender,
                            "to": msg.receiver,
                            "type": msg.message_type,
                            "content": msg.content,
                        },
                    }
                )
        conversation.append(
            {
                "role": "user",
                "content": {
                    "from": input_message.sender,
                    "to": input_message.receiver,
                    "type": input_message.message_type,
                    "content": input_message.content,
                },
            }
        )

        req_started_at = utc_now_iso()
        req_started_ns = monotonic_ns()
        result = self.spec.client.chat(conversation)
        req_finished_at = utc_now_iso()
        req_finished_ns = monotonic_ns()
        llm_duration_ms = (req_finished_ns - req_started_ns) / 1_000_000.0

        llm_call_id = uuid.uuid4().hex
        response_json = result.get("response_json")
        usage = (response_json or {}).get("usage", {}) if response_json else {}

        output_content: Any = ""
        if response_json:
            choices = response_json.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                output_content = message.get("content", "")

        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        tokens_per_sec = None
        if completion_tokens and llm_duration_ms > 0:
            tokens_per_sec = completion_tokens / (llm_duration_ms / 1000.0)

        llm_record = LLMCallRecord(
            call_id=llm_call_id,
            run_id=trace.run_id,
            step_id=step_id,
            agent_name=self.spec.name,
            provider_name=self.spec.client.cfg.provider_name,
            model=self.spec.client.cfg.model_id,
            request_started_at=req_started_at,
            request_finished_at=req_finished_at,
            request_started_ns=req_started_ns,
            request_finished_ns=req_finished_ns,
            duration_ms=llm_duration_ms,
            success=bool(result.get("ok")),
            http_status=result.get("status"),
            error=result.get("error"),
            request_payload=result["request_payload"],
            response_payload=response_json,
            request_payload_bytes=serialized_bytes(result["request_payload"]),
            request_payload_bits=serialized_bits(result["request_payload"]),
            response_payload_bytes=serialized_bytes(response_json) if response_json else None,
            response_payload_bits=serialized_bits(response_json) if response_json else None,
            response_created_unix=(response_json or {}).get("created"),
            output_content=output_content,
            output_preview=preview(output_content),
            output_bytes=serialized_bytes(output_content),
            output_bits=serialized_bits(output_content),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            output_tokens_per_sec=tokens_per_sec,
        )
        trace.log_llm_call(llm_record)

        finished_at = utc_now_iso()
        finished_ns = monotonic_ns()
        success = bool(result.get("ok")) and bool(coerce_text_content(output_content).strip())

        output_message: Optional[MessageRecord] = None
        if success:
            output_message = trace.log_message(
                sender=self.spec.name,
                receiver=reply_to,
                content=output_content,
                message_type=output_message_type,
                parent_message_id=input_message.message_id,
                caused_by_call_id=llm_call_id,
                metadata={
                    "provider": self.spec.client.cfg.provider_name,
                    "model": self.spec.client.cfg.model_id,
                    "prompt_model_id": effective_prompt_model_id(self.spec.client.cfg),
                    **(metadata or {}),
                },
            )

        step_record = AgentStepRecord(
            step_id=step_id,
            run_id=trace.run_id,
            agent_name=self.spec.name,
            role_name=self.spec.role_name,
            input_message_id=input_message.message_id,
            output_message_id=output_message.message_id if output_message else None,
            llm_call_id=llm_call_id,
            started_at=started_at,
            finished_at=finished_at,
            started_ns=started_ns,
            finished_ns=finished_ns,
            duration_ms=(finished_ns - started_ns) / 1_000_000.0,
            input_message_age_ms=input_age_ms,
            success=success,
            error=result.get("error"),
            metadata=metadata or {},
        )
        trace.log_agent_step(step_record)

        if not output_message:
            raise RuntimeError(f"Agent {self.spec.name} failed. error={result.get('error')}")
        return output_message


# -----------------------------
# Minimal orchestrator
# -----------------------------


class SimpleMultiAgentPipeline:
    def __init__(self, planner: Agent, coder: Agent, vision: Agent):
        self.planner = planner
        self.coder = coder
        self.vision = vision

    def run(
        self,
        task: str,
        trace: TraceStore,
        need_code: bool = True,
        need_vision: bool = True,
        success_fn: Optional[Callable[[str, TraceStore], bool]] = None,
    ) -> Dict[str, Any]:
        user_msg = trace.log_message(
            sender="user",
            receiver=self.planner.spec.name,
            content=task,
            message_type="user_task",
        )

        planner_msg = self.planner.act(
            trace=trace,
            input_message=user_msg,
            reply_to="orchestrator",
            output_message_type="plan",
        )

        extra_outputs: List[MessageRecord] = []

        if need_code:
            code_request = trace.log_message(
                sender=self.planner.spec.name,
                receiver=self.coder.spec.name,
                content={
                    "task": task,
                    "plan": planner_msg.content,
                    "instruction": "Write code or pseudocode only if needed. Be concise and explicit.",
                },
                message_type="subtask_code",
                parent_message_id=planner_msg.message_id,
            )
            coder_reply = self.coder.act(
                trace=trace,
                input_message=code_request,
                reply_to=self.planner.spec.name,
                output_message_type="code_result",
                extra_messages=[planner_msg],
            )
            extra_outputs.append(coder_reply)

        if need_vision:
            vision_request = trace.log_message(
                sender=self.planner.spec.name,
                receiver=self.vision.spec.name,
                content={
                    "task": task,
                    "plan": planner_msg.content,
                    "instruction": "If the task is not vision-related, explicitly say so in one line.",
                },
                message_type="subtask_vision",
                parent_message_id=planner_msg.message_id,
            )
            vision_reply = self.vision.act(
                trace=trace,
                input_message=vision_request,
                reply_to=self.planner.spec.name,
                output_message_type="vision_result",
                extra_messages=[planner_msg],
            )
            extra_outputs.append(vision_reply)

        synthesis_request = trace.log_message(
            sender="orchestrator",
            receiver=self.planner.spec.name,
            content={
                "task": task,
                "planner_initial_output": planner_msg.content,
                "sub_results": [msg.content for msg in extra_outputs],
                "instruction": "Synthesize the final answer for the user. Mention assumptions and uncertainties explicitly.",
            },
            message_type="synthesis_request",
            parent_message_id=planner_msg.message_id,
        )
        final_reply = self.planner.act(
            trace=trace,
            input_message=synthesis_request,
            reply_to="user",
            output_message_type="final_answer",
            extra_messages=[planner_msg, *extra_outputs],
        )

        final_text = coerce_text_content(final_reply.content)
        success = success_fn(final_text, trace) if success_fn else bool(final_text.strip())
        trace.finish(success=success)
        return {
            "run_id": trace.run_id,
            "success": success,
            "final_answer": final_reply.content,
            "export_hint": "Call trace.export(output_dir) to persist messages/steps/llm_calls.",
        }


# -----------------------------
# Wiring
# -----------------------------


def make_pipeline(cfg: PipelineConfig) -> tuple[SimpleMultiAgentPipeline, TraceStore]:
    planner = Agent(
        AgentSpec(
            name="planner",
            role_name="plan_agent",
            system_prompt=(
                "You are the planner. Break the task down, decide what code/vision help is needed, "
                "and produce a compact actionable plan."
            ),
            client=HuggingFaceLocalClient(cfg.planner, cfg.runtime),
        )
    )
    coder = Agent(
        AgentSpec(
            name="coder",
            role_name="small_coder",
            system_prompt="You are the coder. Produce code, pseudocode, or structured implementation notes only.",
            client=HuggingFaceLocalClient(cfg.coder, cfg.runtime),
        )
    )
    vision = Agent(
        AgentSpec(
            name="vision",
            role_name="vision_agent",
            system_prompt="You are the vision specialist. If no image is given, explain briefly what visual evidence would be needed.",
            client=HuggingFaceLocalClient(cfg.vision, cfg.runtime),
        )
    )

    pipeline = SimpleMultiAgentPipeline(planner=planner, coder=coder, vision=vision)
    trace = TraceStore(
        task="",
        config={
            "planner": asdict(cfg.planner),
            "coder": asdict(cfg.coder),
            "vision": asdict(cfg.vision),
            "runtime": asdict(cfg.runtime),
            "output_dir": cfg.output_dir,
            "need_code": cfg.need_code,
            "need_vision": cfg.need_vision,
            "preset_name": cfg.preset_name,
            "notes": cfg.notes,
        },
    )
    return pipeline, trace


# -----------------------------
# CLI
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal multi-agent skeleton with detailed tracing (Hugging Face local models + presets)."
    )
    parser.add_argument("task", nargs="?", type=str, help="User task")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["gemma4_24gb_strict_vocab", "gemma4_24gb_near_1b_compromise"],
        help="Start from a built-in preset before applying config/CLI overrides",
    )
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument(
        "--write-config",
        type=str,
        help="Write the effective config JSON to this path and exit",
    )
    parser.add_argument("--output-dir", type=str, help="Where to export trace files")
    parser.add_argument("--no-code", action="store_true", help="Skip coder agent")
    parser.add_argument("--no-vision", action="store_true", help="Skip vision agent")

    parser.add_argument("--planner-model", type=str)
    parser.add_argument("--coder-model", type=str)
    parser.add_argument("--vision-model", type=str)
    parser.add_argument("--planner-prompt-model", type=str)
    parser.add_argument("--coder-prompt-model", type=str)
    parser.add_argument("--vision-prompt-model", type=str)
    parser.add_argument("--planner-max-new-tokens", type=int)
    parser.add_argument("--coder-max-new-tokens", type=int)
    parser.add_argument("--vision-max-new-tokens", type=int)
    parser.add_argument("--planner-torch-dtype", type=str)
    parser.add_argument("--coder-torch-dtype", type=str)
    parser.add_argument("--vision-torch-dtype", type=str)
    parser.add_argument("--planner-load-in-4bit", type=parse_optional_bool)
    parser.add_argument("--coder-load-in-4bit", type=parse_optional_bool)
    parser.add_argument("--vision-load-in-4bit", type=parse_optional_bool)
    parser.add_argument("--sequential-model-loading", type=parse_optional_bool)
    parser.add_argument("--unload-model-after-call", type=parse_optional_bool)
    parser.add_argument("--clear-cuda-cache-after-call", type=parse_optional_bool)

    args = parser.parse_args()

    base_cfg = build_preset_config(args.preset) if args.preset else DEFAULT_CONFIG
    cfg = load_pipeline_config(args.config, base_cfg)
    cfg = apply_cli_overrides(cfg, args)
    validate_pipeline_config(cfg)

    if args.write_config:
        out = build_example_config_file(args.write_config, cfg)
        print(f"Wrote config to: {out.resolve()}")
        return

    if not args.task:
        parser.error("task is required unless --write-config is used")

    pipeline, trace = make_pipeline(cfg)
    trace.summary.task = args.task

    result = pipeline.run(
        task=args.task,
        trace=trace,
        need_code=cfg.need_code,
        need_vision=cfg.need_vision,
    )
    output_dir = trace.export(cfg.output_dir)

    print("=" * 80)
    print("RUN RESULT")
    print("=" * 80)
    print(safe_json_dumps(result))
    print(f"Exported trace files to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
