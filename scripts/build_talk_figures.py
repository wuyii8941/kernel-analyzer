#!/usr/bin/env python3
"""Build the three simple, question-driven figures used by the public talk."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
W, H = 1800, 1000
BG = "#F7F8FA"
INK = "#172033"
MUTED = "#687386"
GRID = "#DDE3EB"
BLUE = "#2867AD"
RED = "#D34A59"
GREEN = "#2A9D70"
ORANGE = "#E98B26"
GRAY = "#AAB2BF"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def base(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((80, 50), title, font=f(48, True), fill=INK)
    draw.text((82, 118), subtitle, font=f(25), fill=MUTED)
    return image, draw


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
           size: int, color: str = INK, bold: bool = False) -> None:
    box = draw.textbbox((0, 0), text, font=f(size, bold))
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
              text, font=f(size, bold), fill=color)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          color: str = BLUE) -> None:
    draw.line((start, end), fill=color, width=6)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for offset in (2.55, -2.55):
        point = (end[0] + 20 * math.cos(angle + offset),
                 end[1] + 20 * math.sin(angle + offset))
        draw.line((end, point), fill=color, width=6)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=24, fill="white", outline="#D5DCE6", width=2)
    draw.text((box[0] + 34, box[1] + 24), title, font=f(29, True), fill=INK)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, optimize=True)


def method_overview() -> None:
    image, draw = base(
        "只看输出不够：差异要穿过完整训练路径",
        "同一状态下只替换一个实现，然后观察它是否经过 backward 和 AdamW 持续进入参数",
    )

    boxes = [
        (90, 270, 390, 500, "原实现 vs 修复实现", "同一参数、输入、随机状态\n只改变目标实现"),
        (510, 270, 790, 500, "真实 backward", "使用真实保存量\n和上游 gradient"),
        (910, 270, 1190, 500, "目标 AdamW", "使用相同的\n历史状态"),
        (1310, 270, 1710, 500, "参数 update 差异", "前 16 步筛查\n32 步完整确认"),
    ]
    colors = [BLUE, BLUE, ORANGE, GREEN]
    for (x1, y1, x2, y2, head, body), color in zip(boxes, colors):
        fill = "#EEF4FC" if color == BLUE else "#FFF4E8" if color == ORANGE else "#EAF6F0"
        draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=fill, outline=color, width=3)
        center(draw, ((x1 + x2) / 2, y1 + 65), head, 27, color, True)
        for i, line in enumerate(body.split("\n")):
            center(draw, ((x1 + x2) / 2, y1 + 130 + 36 * i), line, 20, MUTED)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2], 385), (right[0], 385))

    draw.rounded_rectangle((250, 650, 1550, 800), radius=24, fill="white", outline=GRID, width=2)
    center(draw, (900, 700), "我们真正判断的是：这些 update 差异会抵消，还是持续同向？", 32, INK, True)
    center(draw, (900, 756), "证据不足就停止，不从相似案例借结论", 23, RED)
    save(image, "method_overview.png")


def main_results() -> None:
    rms = json.loads((ROOT / "results/property/joint_bias_formation_v1/rms_persistence/rms_persistence.json").read_text())
    rows = rms["formation_population"]["rows"]
    ablation_paths = {
        "Liger": ROOT / "results/property/direct_persistence_v4/optimizer_state/liger_t128_same_state_ablation.json",
        "Phi": ROOT / "results/property/direct_persistence_v4/optimizer_state/phi_seq64_same_state_ablation.json",
        "Qwen": ROOT / "results/property/direct_persistence_v4/optimizer_state/qwen_seq128_same_state_ablation.json",
    }
    ablations = {name: json.loads(path.read_text())["arms"] for name, path in ablation_paths.items()}

    image, draw = base(
        "两个主结果：误差大小不够，Optimizer 也会改变结论",
        "左：32 个真实记录中，RMS 几乎不解释方向；右：同一 gradient 经过 AdamW 后可能持续，也可能抵消",
    )
    panel(draw, (55, 185, 900, 920), "A  误差 RMS 与方向几乎无关")
    panel(draw, (935, 185, 1745, 920), "B  Gradient 有方向，不代表 update 仍有方向")

    # Scatter.
    plot = (130, 315, 835, 800)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=INK, width=2)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=INK, width=2)
    x_min, x_max = -7.0, -1.0
    y_min, y_max = -0.05, 0.12
    model_colors = {"deepseek8b": BLUE, "mamba": GREEN, "phi4": RED, "qwen": ORANGE}
    for exponent in range(-7, 0):
        x = plot[0] + (exponent - x_min) / (x_max - x_min) * (plot[2] - plot[0])
        draw.line((x, plot[1], x, plot[3]), fill=GRID, width=1)
        center(draw, (x, plot[3] + 30), f"1e{exponent}", 15, MUTED)
    for value in (-0.05, 0.0, 0.05, 0.10):
        y = plot[3] - (value - y_min) / (y_max - y_min) * (plot[3] - plot[1])
        draw.line((plot[0], y, plot[2], y), fill=GRID, width=1)
        draw.text((72, y - 10), f"{value:.2f}", font=f(15), fill=MUTED)
    for row in rows:
        x = plot[0] + (math.log10(row["local_rms"]) - x_min) / (x_max - x_min) * (plot[2] - plot[0])
        y = plot[3] - (row["formation_cross_state_ratio"] - y_min) / (y_max - y_min) * (plot[3] - plot[1])
        color = model_colors[row["model"]]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
    center(draw, ((plot[0] + plot[2]) / 2, 865), "局部误差 RMS（对数刻度）", 21)
    draw.text((74, 275), "方向统计", font=f(19), fill=INK)
    draw.rounded_rectangle((480, 340, 810, 440), radius=16, fill="white", outline=GRID)
    center(draw, (645, 375), "相关系数  r = 0.018", 25, BLUE, True)
    center(draw, (645, 415), "基本为零", 21, MUTED)

    # Gradient -> AdamW.
    p2 = (1030, 320, 1665, 790)
    draw.line((p2[0], p2[3], p2[2], p2[3]), fill=INK, width=2)
    draw.line((p2[0], p2[1], p2[0], p2[3]), fill=INK, width=2)
    for value in range(0, 6):
        y = p2[3] - value / 5 * (p2[3] - p2[1])
        draw.line((p2[0], y, p2[2], y), fill=GRID, width=1)
        draw.text((990, y - 10), str(value), font=f(15), fill=MUTED)
    xs = [1130, 1350, 1570]
    colors = [BLUE, RED, GREEN]
    for x, (name, arms), color in zip(xs, ablations.items(), colors):
        grad = arms["gradient_difference"]["A32"]
        adam = arms["captured_adamw_moments"]["A32"]
        for j, (value, label) in enumerate(((grad, "gradient"), (adam, "AdamW"))):
            bx = x - 58 + 72 * j
            y = p2[3] - value / 5 * (p2[3] - p2[1])
            draw.rounded_rectangle((bx, y, bx + 52, p2[3]), radius=8,
                                   fill=color if j == 0 else "#C8D0DC")
            center(draw, (bx + 26, y - 22), f"{value:.2f}", 17,
                   color if j == 0 else MUTED, True)
        center(draw, (x, 840), name, 21, INK, True)
    draw.line((1110, 885, 1150, 885), fill=BLUE, width=9)
    draw.text((1160, 869), "gradient", font=f(17), fill=INK)
    draw.line((1360, 885, 1400, 885), fill="#C8D0DC", width=9)
    draw.text((1410, 869), "AdamW update", font=f(17), fill=INK)
    save(image, "oracle_main_results.png")


def causal_result() -> None:
    phi = json.loads((ROOT / "results/property/direct_persistence_v4/interventions/phi_seq64_adamw_sr32.json").read_text())
    image, draw = base(
        "只改舍入方式：持续方向消失，误差能量没有随之消失",
        "Phi、相同 32 个状态、相同 backward、相同 AdamW；唯一变化是确定性舍入改为随机舍入",
    )
    panel(draw, (55, 185, 935, 915), "A  方向：原实现超过随机范围，随机舍入回到范围内")
    panel(draw, (970, 185, 1745, 915), "B  能量：前三次随机舍入并没有更小")

    names = ["natural", "sham", "sr_0", "sr_1", "sr_2", "sr_3"]
    labels = ["原实现", "空对照", "SR-0", "SR-1", "SR-2", "SR-3"]
    values = [phi["metrics"][name]["coherence_amplification"] - 1 for name in names]
    nulls = [phi["metrics"][name]["sign_flip_null"]["upper_95"] - 1 for name in names]
    plot = (130, 315, 870, 790)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=INK, width=2)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=INK, width=2)
    ymax = 0.032
    slot = (plot[2] - plot[0]) / len(names)
    for value in (0.0, 0.01, 0.02, 0.03):
        y = plot[3] - value / ymax * (plot[3] - plot[1])
        draw.line((plot[0], y, plot[2], y), fill=GRID, width=1)
        draw.text((75, y - 10), f"{value:.2f}", font=f(15), fill=MUTED)
    for i, (label, value, null) in enumerate(zip(labels, values, nulls)):
        x = plot[0] + slot * (i + 0.5)
        y = plot[3] - value / ymax * (plot[3] - plot[1])
        color = RED if i == 0 else ORANGE if i == 1 else GREEN
        draw.rounded_rectangle((x - 34, y, x + 34, plot[3]), radius=8, fill=color)
        yn = plot[3] - null / ymax * (plot[3] - plot[1])
        draw.line((x - 42, yn, x + 42, yn), fill=INK, width=4)
        center(draw, (x, 830), label, 16)
    center(draw, (500, 875), "柱：A−1    黑线：本次执行的随机 95% 上界", 18, MUTED)

    natural_energy = phi["metrics"]["natural"]["path_l2"]
    energies = [1.0] + [phi["metrics"][f"sr_{i}"]["path_l2"] / natural_energy for i in range(4)]
    labels2 = ["原实现", "SR-0", "SR-1", "SR-2", "SR-3"]
    p2 = (1060, 315, 1665, 790)
    draw.line((p2[0], p2[3], p2[2], p2[3]), fill=INK, width=2)
    draw.line((p2[0], p2[1], p2[0], p2[3]), fill=INK, width=2)
    for value in (0.0, 0.5, 1.0):
        y = p2[3] - value / 1.15 * (p2[3] - p2[1])
        draw.line((p2[0], y, p2[2], y), fill=GRID, width=1)
        draw.text((1010, y - 10), f"{value:.1f}", font=f(15), fill=MUTED)
    slot2 = (p2[2] - p2[0]) / len(energies)
    for i, (label, value) in enumerate(zip(labels2, energies)):
        x = p2[0] + slot2 * (i + 0.5)
        y = p2[3] - value / 1.15 * (p2[3] - p2[1])
        draw.rounded_rectangle((x - 34, y, x + 34, p2[3]), radius=8,
                               fill=BLUE if i == 0 else GREEN)
        center(draw, (x, y - 25), f"{value:.3f}×", 18, INK, True)
        center(draw, (x, 830), label, 16)
    draw.rounded_rectangle((1090, 855, 1635, 895), radius=14, fill="#E9F5EF")
    center(draw, (1362, 875), "前三个 SR 的误差能量相当或更高", 21, GREEN, True)
    save(image, "phi_causal_closure.png")


def main() -> None:
    method_overview()
    main_results()
    causal_result()


if __name__ == "__main__":
    main()
