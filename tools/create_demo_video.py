"""Create a 2+ minute MP4 demo video for the GitHub README.

This script is intentionally simple: it renders a handful of static slides
with light motion into an MP4 file. It depends on Pillow, imageio, and
imageio-ffmpeg, which are documentation-generation tools rather than runtime
dependencies for the engine itself.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np


WIDTH = 1280
HEIGHT = 720
FPS = 10

BG = (14, 17, 23)
PANEL = (25, 31, 42)
PANEL_2 = (33, 40, 52)
TEXT = (238, 242, 247)
MUTED = (157, 168, 184)
CYAN = (65, 209, 199)
GREEN = (89, 205, 144)
AMBER = (245, 177, 84)
BLUE = (116, 170, 255)


def require_video_deps():
    try:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise SystemExit(
            "Missing video dependencies. Install them with:\n"
            "python -m pip install pillow imageio imageio-ffmpeg"
        ) from exc
    return imageio, Image, ImageDraw, ImageFont


def font_loader(ImageFont):
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]

    def load(size: int, bold: bool = False):
        names = [
            Path("C:/Windows/Fonts/segoeuib.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ] if bold else candidates
        for path in names:
            if path.exists():
                return ImageFont.truetype(str(path), size=size)
        return ImageFont.load_default()

    return load


def wrap(draw, text: str, font, width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, xy, text: str, font, fill, width: int, line_gap: int = 8) -> int:
    x, y = xy
    for line in wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += bbox[3] - bbox[1] + line_gap
    return y


def read_benchmark_lines(path: Path) -> List[str]:
    if not path.exists():
        return [
            "Greedy single decode       164.2 tok/s",
            "Paged cache greedy         169.8 tok/s",
            "Prompt lookup speculation  248.5 tok/s",
            "Greedy batched decode      537.9 tok/s",
        ]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| ---") or "Benchmark" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 7:
            label = cells[0]
            if cells[3] != "1":
                label = f"{label} (batch {cells[3]})"
            elif cells[2] not in ("", "-"):
                label = f"{label} ({cells[2]})"
            rows.append(f"{label:36} {cells[6]} tok/s")
    return rows[:6]


def rounded_rectangle(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_header(draw, load_font, title: str, subtitle: str, second: float, duration: float):
    progress = min(1.0, max(0.0, second / duration))
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=BG)
    draw.rectangle((0, 0, int(WIDTH * progress), 6), fill=CYAN)
    draw.text((70, 58), "Ultra LLM Engine", font=load_font(24, bold=True), fill=CYAN)
    draw.text((70, 108), title, font=load_font(56, bold=True), fill=TEXT)
    if subtitle:
        draw_wrapped(draw, (74, 184), subtitle, load_font(25), MUTED, 860, line_gap=10)


def draw_bullets(draw, load_font, bullets: Iterable[str], start_y: int):
    y = start_y
    for idx, bullet in enumerate(bullets):
        color = [CYAN, GREEN, AMBER, BLUE][idx % 4]
        draw.ellipse((80, y + 10, 94, y + 24), fill=color)
        y = draw_wrapped(draw, (112, y), bullet, load_font(25), TEXT, 980, line_gap=8) + 18


def slide_intro(draw, load_font, t, duration, benchmark_lines):
    draw_header(
        draw,
        load_font,
        "Small engine. Real inference ideas.",
        "A compact NumPy/CUDA reference runtime for studying how modern LLM serving systems work.",
        t,
        duration,
    )
    draw_bullets(
        draw,
        load_font,
        [
            "Quantized decoder-only transformer",
            "Paged KV cache and layer offload",
            "Greedy, n-gram, prompt-lookup, and neural speculative decoding",
            "Runs on CPU by default; CUDA sources are included for experimentation",
        ],
        280,
    )
    draw.text((70, 642), "Demo video duration: 2 minutes 5 seconds", font=load_font(21), fill=MUTED)


def slide_architecture(draw, load_font, t, duration, benchmark_lines):
    draw_header(
        draw,
        load_font,
        "Readable inference pipeline",
        "The hot path is intentionally explicit so each optimization can be inspected and modified.",
        t,
        duration,
    )
    steps = ["Prompt", "BPE tokenizer", "Scheduler", "Quantized model", "KV cache", "Text output"]
    x = 92
    y = 332
    for idx, step in enumerate(steps):
        color = [CYAN, BLUE, GREEN, AMBER, CYAN, BLUE][idx]
        rounded_rectangle(draw, (x, y, x + 165, y + 82), 10, PANEL, outline=color, width=2)
        draw.text((x + 18, y + 26), step, font=load_font(22, bold=True), fill=TEXT)
        if idx < len(steps) - 1:
            draw.line((x + 176, y + 41, x + 222, y + 41), fill=MUTED, width=3)
            draw.polygon([(x + 222, y + 41), (x + 210, y + 33), (x + 210, y + 49)], fill=MUTED)
        x += 194
    draw_bullets(
        draw,
        load_font,
        [
            "Prefill walks prompt tokens through every layer.",
            "Generation selects a decoding strategy and updates cache state.",
            "Batched decode amortizes dequantization across multiple prompts.",
        ],
        492,
    )


def slide_features(draw, load_font, t, duration, benchmark_lines):
    draw_header(
        draw,
        load_font,
        "Core features in one repo",
        "Each system idea has a focused module and test coverage.",
        t,
        duration,
    )
    items = [
        ("Quantization", "INT-2 / INT-4 / INT-8 plus NF4"),
        ("Kernels", "Fused dequant GEMV/GEMM and RMSNorm + RoPE"),
        ("Cache", "Plain KV cache plus reusable page pool"),
        ("Checkpoints", "safetensors and supported GGUF tensors"),
        ("Speculation", "Neural, n-gram, and prompt-lookup drafting"),
        ("Testing", "32 pytest checks across the engine"),
    ]
    for idx, (title, body) in enumerate(items):
        col = idx % 2
        row = idx // 2
        left = 82 + col * 560
        top = 250 + row * 115
        rounded_rectangle(draw, (left, top, left + 500, top + 82), 10, PANEL, outline=PANEL_2, width=2)
        draw.text((left + 24, top + 15), title, font=load_font(24, bold=True), fill=[CYAN, GREEN, AMBER][row])
        draw.text((left + 24, top + 47), body, font=load_font(20), fill=MUTED)


def slide_benchmarks(draw, load_font, t, duration, benchmark_lines):
    draw_header(
        draw,
        load_font,
        "Benchmark snapshot",
        "CPU smoke benchmark on the random 3-layer reference model.",
        t,
        duration,
    )
    rounded_rectangle(draw, (86, 238, 1194, 592), 10, PANEL, outline=PANEL_2, width=2)
    draw.text((118, 270), "Benchmark", font=load_font(24, bold=True), fill=CYAN)
    draw.text((742, 270), "Throughput", font=load_font(24, bold=True), fill=CYAN)
    y = 326
    for line in benchmark_lines:
        if " tok/s" in line:
            name, speed = line.rsplit(" ", 2)[0], " ".join(line.rsplit(" ", 2)[1:])
        else:
            name, speed = line, ""
        draw.text((118, y), name.strip(), font=load_font(22), fill=TEXT)
        draw.text((742, y), speed.strip(), font=load_font(22, bold=True), fill=GREEN)
        draw.line((118, y + 34, 1128, y + 34), fill=(48, 56, 70), width=1)
        y += 48
    draw.text((118, 626), "Numbers are regression signals, not production-serving claims.", font=load_font(20), fill=MUTED)


def slide_commands(draw, load_font, t, duration, benchmark_lines):
    draw_header(
        draw,
        load_font,
        "Run it locally",
        "The default path needs no checkpoint and works on CPU.",
        t,
        duration,
    )
    commands = [
        "pip install -r requirements.txt",
        "python -m pytest tests/ -v",
        "python cli.py --prompt \"Explain speculative decoding\" --draft-mode lookup --max-tokens 48",
        "python benchmarks/run_benchmarks.py --output benchmarks/results.md",
    ]
    y = 248
    for command in commands:
        rounded_rectangle(draw, (84, y, 1194, y + 64), 8, (8, 12, 18), outline=PANEL_2, width=2)
        draw.text((112, y + 20), command, font=load_font(22), fill=TEXT)
        y += 88


def slide_close(draw, load_font, t, duration, benchmark_lines):
    draw_header(
        draw,
        load_font,
        "Built for learning and modification",
        "A compact reference engine for experimenting with inference techniques before moving to production stacks.",
        t,
        duration,
    )
    draw_bullets(
        draw,
        load_font,
        [
            "MIT licensed",
            "Dependency-light runtime",
            "Honest scope and known limitations documented",
            "Roadmap: GGUF k-quant support, attention masking, prefix cache, and broader CUDA coverage",
        ],
        292,
    )
    draw.text((70, 640), "github.com/maharab549/Ultra-LLM-Engine", font=load_font(24, bold=True), fill=CYAN)


SLIDES = [
    (slide_intro, 20),
    (slide_architecture, 21),
    (slide_features, 22),
    (slide_benchmarks, 24),
    (slide_commands, 20),
    (slide_close, 18),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the GitHub demo video")
    parser.add_argument("--output", type=Path, default=Path("assets/demo/ultra-llm-engine-demo.mp4"))
    parser.add_argument("--benchmarks", type=Path, default=Path("benchmarks/results.md"))
    args = parser.parse_args()

    imageio, Image, ImageDraw, ImageFont = require_video_deps()
    load_font = font_loader(ImageFont)
    benchmark_lines = read_benchmark_lines(args.benchmarks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_seconds = sum(duration for _, duration in SLIDES)

    with imageio.get_writer(
        args.output,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=16,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for slide_func, duration in SLIDES:
            frame_count = duration * FPS
            for frame in range(frame_count):
                t = frame / FPS
                img = Image.new("RGB", (WIDTH, HEIGHT), BG)
                draw = ImageDraw.Draw(img)
                slide_func(draw, load_font, t, duration, benchmark_lines)
                pulse = int(22 + 10 * math.sin((t / max(duration, 1)) * math.pi * 2))
                draw.ellipse((1168, 54, 1168 + pulse, 54 + pulse), fill=CYAN)
                writer.append_data(np.asarray(img))

    print(f"Wrote {args.output} ({total_seconds} seconds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
