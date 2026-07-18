"""Run small, reproducible CPU benchmarks for Ultra LLM Engine.

These numbers are meant as a smoke benchmark for the reference implementation,
not a claim of production serving performance. The default configuration keeps
the model intentionally tiny so the benchmark can run quickly on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from engine.config import EngineConfig
from engine.kernels import FusedKernels
from engine.model import QuantizedTransformer
from engine.scheduler import Scheduler
from engine.tokenizer import BPETokenizer


PROMPTS = [
    "Speculative decoding speeds up generation when draft tokens are accepted.",
    "Paged KV cache reuses memory pages across active and finished sequences.",
    "Quantized weights reduce memory bandwidth pressure during inference.",
    "Batched decoding amortizes dequantization work across several prompts.",
]


@dataclass
class BenchmarkResult:
    name: str
    mode: str
    cache: str
    batch_size: int
    generated_tokens: int
    avg_seconds: float
    tokens_per_second: float
    acceptance_rate: Optional[float]
    notes: str


def build_tokenizer() -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.train(PROMPTS, num_merges=80)
    return tokenizer


def make_config(tokenizer: BPETokenizer, *, use_paged_cache: bool) -> EngineConfig:
    return EngineConfig(
        hidden_dim=96,
        n_layers=3,
        n_heads=6,
        vocab_size=tokenizer.vocab_size,
        group_size=16,
        gpu_layers=1,
        prefetch_depth=1,
        quant_bits=4,
        quant_scheme="int",
        use_paged_cache=use_paged_cache,
        page_size=8,
        speculate_k=4,
        target_temperature=1e-8,
        seed=42,
    )


def make_model(cfg: EngineConfig) -> QuantizedTransformer:
    return QuantizedTransformer.from_random(cfg, FusedKernels(cfg, force_numpy=True))


def time_case(fn: Callable[[], Optional[float]], repeats: int) -> tuple[float, Optional[float]]:
    durations: List[float] = []
    acceptance_rates: List[float] = []
    fn()  # warm-up
    for _ in range(repeats):
        start = time.perf_counter()
        acceptance_rate = fn()
        durations.append(time.perf_counter() - start)
        if acceptance_rate is not None:
            acceptance_rates.append(acceptance_rate)
    avg_seconds = statistics.mean(durations)
    avg_acceptance = statistics.mean(acceptance_rates) if acceptance_rates else None
    return avg_seconds, avg_acceptance


def run_single(
    *,
    name: str,
    mode: str,
    tokenizer: BPETokenizer,
    use_paged_cache: bool,
    max_tokens: int,
    repeats: int,
) -> BenchmarkResult:
    cfg = make_config(tokenizer, use_paged_cache=use_paged_cache)
    target = make_model(cfg)
    draft = None
    if mode == "model":
        draft_cfg = EngineConfig(
            hidden_dim=32,
            n_layers=1,
            n_heads=4,
            vocab_size=cfg.vocab_size,
            group_size=16,
            quant_bits=4,
            quant_scheme="int",
            target_temperature=1e-8,
            seed=7,
        )
        draft = make_model(draft_cfg)

    def once() -> Optional[float]:
        with Scheduler(cfg, tokenizer, target, draft_model=draft, draft_mode=mode) as scheduler:
            scheduler.generate(PROMPTS[0], max_tokens=max_tokens, temperature=1e-8)
            if scheduler.spec is not None:
                return scheduler.spec.acceptance_rate
            if scheduler.lightweight_spec is not None:
                return scheduler.lightweight_spec.acceptance_rate
            return None

    avg_seconds, acceptance_rate = time_case(once, repeats)
    generated_tokens = max_tokens
    return BenchmarkResult(
        name=name,
        mode=mode,
        cache="paged" if use_paged_cache else "plain",
        batch_size=1,
        generated_tokens=generated_tokens,
        avg_seconds=avg_seconds,
        tokens_per_second=generated_tokens / avg_seconds,
        acceptance_rate=acceptance_rate,
        notes="single prompt",
    )


def run_batch(
    *,
    tokenizer: BPETokenizer,
    use_paged_cache: bool,
    max_tokens: int,
    repeats: int,
) -> BenchmarkResult:
    cfg = make_config(tokenizer, use_paged_cache=use_paged_cache)
    target = make_model(cfg)
    prompts = PROMPTS

    def once() -> Optional[float]:
        with Scheduler(cfg, tokenizer, target, draft_mode="greedy") as scheduler:
            scheduler.generate_batch(prompts, max_tokens=max_tokens, temperature=1e-8)
        return None

    avg_seconds, _ = time_case(once, repeats)
    generated_tokens = len(prompts) * max_tokens
    return BenchmarkResult(
        name="Greedy batched decode",
        mode="greedy",
        cache="paged" if use_paged_cache else "plain",
        batch_size=len(prompts),
        generated_tokens=generated_tokens,
        avg_seconds=avg_seconds,
        tokens_per_second=generated_tokens / avg_seconds,
        acceptance_rate=None,
        notes="aggregate throughput",
    )


def markdown_table(results: List[BenchmarkResult]) -> str:
    lines = [
        "| Benchmark | Mode | Cache | Batch | Tokens | Avg seconds | Tok/s | Accept rate | Notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        acceptance = "-" if result.acceptance_rate is None else f"{result.acceptance_rate * 100:.1f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    result.name,
                    result.mode,
                    result.cache,
                    str(result.batch_size),
                    str(result.generated_tokens),
                    f"{result.avg_seconds:.3f}",
                    f"{result.tokens_per_second:.1f}",
                    acceptance,
                    result.notes,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def environment_summary() -> dict:
    processor = platform.processor() or "CPU"
    if platform.system() == "Windows" and "Family" in processor:
        try:
            detected = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if detected:
                processor = detected
        except Exception:
            pass

    return {
        "date": date.today().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": processor,
        "numpy": np.__version__,
        "backend": "NumPy CPU fallback",
    }


def render_report(results: List[BenchmarkResult]) -> str:
    env = environment_summary()
    return "\n".join(
        [
            "# Benchmark Results",
            "",
            "Small CPU smoke benchmark for the reference implementation.",
            "",
            f"- Date: {env['date']}",
            f"- Platform: {env['platform']}",
            f"- Processor: {env['processor']}",
            f"- Python: {env['python']}",
            f"- NumPy: {env['numpy']}",
            f"- Backend: {env['backend']}",
            "- Model: random 3-layer decoder, hidden_dim=96, n_heads=6, INT4 weights",
            "- Decode length: 48 generated tokens per prompt",
            "",
            markdown_table(results),
            "",
            "These measurements are best used for regression tracking inside this",
            "repository. They are not production throughput claims.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ultra LLM Engine smoke benchmarks")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats per case")
    parser.add_argument("--max-tokens", type=int, default=48, help="Generated tokens per prompt")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    tokenizer = build_tokenizer()
    results = [
        run_single(
            name="Greedy single decode",
            mode="greedy",
            tokenizer=tokenizer,
            use_paged_cache=False,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        ),
        run_single(
            name="Greedy single decode",
            mode="greedy",
            tokenizer=tokenizer,
            use_paged_cache=True,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        ),
        run_single(
            name="Prompt lookup speculation",
            mode="lookup",
            tokenizer=tokenizer,
            use_paged_cache=True,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        ),
        run_single(
            name="N-gram speculation",
            mode="ngram",
            tokenizer=tokenizer,
            use_paged_cache=True,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        ),
        run_single(
            name="Neural draft speculation",
            mode="model",
            tokenizer=tokenizer,
            use_paged_cache=True,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        ),
        run_batch(
            tokenizer=tokenizer,
            use_paged_cache=False,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        ),
    ]

    report = render_report(results)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"environment": environment_summary(), "results": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
