#!/usr/bin/env python3
"""Build the three evidence figures used by the public talk.

The figures are deliberately question-driven:
1. what the method measures;
2. why magnitude is not a useful persistence oracle;
3. whether the Phi mechanism survives a same-protocol causal test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
W, H = 1800, 1000
BG = "#F7F8FA"
INK = "#162033"
MUTED = "#5D687A"
GRID = "#DDE2EA"
BLUE = "#2364AA"
RED = "#D1495B"
GREEN = "#2A9D6F"
ORANGE = "#E58A2B"
PURPLE = "#7756A8"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.text((80, 52), title, font=font(46, True), fill=INK)
    draw.text((82, 116), subtitle, font=font(24), fill=MUTED)
    return image, draw


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=24, fill="white", outline="#D9DEE7", width=2)
    draw.text((box[0] + 30, box[1] + 22), title, font=font(28, True), fill=INK)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
             size: int, fill: str = INK, bold: bool = False) -> None:
    box = draw.textbbox((0, 0), text, font=font(size, bold))
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
              text, font=font(size, bold), fill=fill)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          color: str = BLUE, width: int = 5) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    for offset in (2.55, -2.55):
        point = (end[0] + length * math.cos(angle + offset),
                 end[1] + length * math.sin(angle + offset))
        draw.line((end, point), fill=color, width=width)


def save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, optimize=True)


def method_overview() -> None:
    image, draw = canvas(
        "从实现差异到训练风险判断",
        "先隔离一个真实 F+B 差异，再判断它是抵消、直接持续，还是由训练反馈维持",
    )
    panel(draw, (70, 180, 860, 845), "1  精确隔离：同一状态，只改一个实现边界")
    panel(draw, (940, 180, 1730, 845), "2  训练判定：先分诊，再确认，再归因")

    # Phase 1
    boxes = [
        (135, 280, 360, 390, "共同状态", "weights · input · RNG · moments"),
        (480, 245, 770, 425, "三条执行臂", "candidate\nmatched repair\nno-op sham"),
        (135, 525, 360, 635, "真实反向传播", "使用真实保存量与上游梯度"),
        (480, 525, 770, 635, "目标 optimizer", "同一 AdamW 配置与 pre-state"),
    ]
    for x1, y1, x2, y2, head, body in boxes:
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill="#EEF4FC", outline="#AFC5DF", width=2)
        centered(draw, ((x1+x2)//2, y1+36), head, 25, BLUE, True)
        lines = body.split("\n")
        for index, line in enumerate(lines):
            centered(draw, ((x1+x2)//2, y1+73+index*29), line, 17, MUTED)
    arrow(draw, (360, 335), (480, 335))
    arrow(draw, (625, 425), (250, 525))
    arrow(draw, (360, 580), (480, 580))
    draw.rounded_rectangle((230, 705, 700, 790), radius=20, fill="#E7F5EE", outline="#8CC9AA", width=2)
    centered(draw, (465, 735), "直接有效更新差异  L(t)", 29, GREEN, True)
    centered(draw, (465, 771), "不推进双臂轨迹；先测算子本步作用", 18, MUTED)
    arrow(draw, (625, 635), (625, 705), GREEN)

    # Phase 2
    steps = [
        (1010, 260, 1235, 420, "16 步筛查", "ESCALATE / NO ESCALATION\n证据不全则 ABSTAIN"),
        (1390, 260, 1645, 420, "32 步确认", "与每一行自己的\n随机抵消基线比较"),
        (1010, 535, 1235, 665, "闭环轨迹", "actual drift D\n= direct L + feedback B"),
        (1390, 535, 1645, 665, "机制干预", "关掉预测机制\nsham 不应改变结果"),
    ]
    for x1, y1, x2, y2, head, body in steps:
        fill = "#FFF4E8" if y1 < 500 else "#F2EEFA"
        outline = "#E7B477" if y1 < 500 else "#BBA8D4"
        draw.rounded_rectangle((x1, y1, x2, y2), radius=18, fill=fill, outline=outline, width=2)
        centered(draw, ((x1+x2)//2, y1+38), head, 24, ORANGE if y1 < 500 else PURPLE, True)
        for index, line in enumerate(body.split("\n")):
            centered(draw, ((x1+x2)//2, y1+82+index*25), line, 15, MUTED)
    arrow(draw, (1235, 340), (1390, 340), ORANGE)
    arrow(draw, (1518, 420), (1518, 535), PURPLE)
    arrow(draw, (1390, 600), (1235, 600), PURPLE)
    draw.rounded_rectangle((1050, 730, 1605, 800), radius=18, fill="#E7F5EE", outline="#8CC9AA", width=2)
    centered(draw, (1328, 756), "输出：存在性 · 形成位置 · 后果来源 · 可修复性", 23, GREEN, True)
    centered(draw, (1328, 787), "缺一层证据就收窄结论，不用相邻案例补数", 17, MUTED)
    arrow(draw, (1123, 665), (1180, 730), GREEN)
    arrow(draw, (1518, 665), (1470, 730), GREEN)

    draw.text((86, 900), "设计原则：每个箭头都对应可执行程序和证据文件；不把首轮覆盖误写成 32 步深测。",
              font=font(22), fill=INK)
    save(image, "method_overview.png")


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.line((box[0], box[3], box[2], box[3]), fill=INK, width=2)
    draw.line((box[0], box[1], box[0], box[3]), fill=INK, width=2)


def oracle_results() -> None:
    data = json.loads((ROOT / "results/property/joint_bias_formation_v1/rms_persistence/rms_persistence.json").read_text())
    rows = data["formation_population"]["rows"]
    image, draw = canvas(
        "主结果：误差大小没有回答方向结构",
        "同一批数据中，幅值几乎不解释方向结构；方向分数才提供可用的分诊信号",
    )
    panel(draw, (55, 175, 815, 905), "A  32 个可达记录：RMS 与方向性几乎无关")
    panel(draw, (845, 175, 1265, 905), "B  三阶段定位：方向在哪里形成")
    panel(draw, (1295, 175, 1745, 905), "C  16 步回溯分诊")

    # A: scatter
    plot = (135, 285, 750, 785); axes(draw, plot)
    x_min, x_max = -7.0, -1.0; y_min, y_max = -0.05, 0.12
    model_colors = {"deepseek8b": BLUE, "mamba": GREEN, "phi4": RED, "qwen": ORANGE}
    for exponent in range(-7, 0):
        x = plot[0] + (exponent - x_min) / (x_max - x_min) * (plot[2] - plot[0])
        draw.line((x, plot[1], x, plot[3]), fill=GRID, width=1)
        centered(draw, (int(x), plot[3]+28), f"1e{exponent}", 15, MUTED)
    for value in (-0.05, 0.0, 0.05, 0.10):
        y = plot[3] - (value-y_min)/(y_max-y_min)*(plot[3]-plot[1])
        draw.line((plot[0], y, plot[2], y), fill=GRID, width=1)
        draw.text((75, y-11), f"{value:.2f}", font=font(15), fill=MUTED)
    for row in rows:
        x = plot[0] + (math.log10(row["local_rms"])-x_min)/(x_max-x_min)*(plot[2]-plot[0])
        y = plot[3] - (row["formation_cross_state_ratio"]-y_min)/(y_max-y_min)*(plot[3]-plot[1])
        color = model_colors[row["model"]]
        if row["formation_status"].startswith("UNRESOLVED"):
            draw.ellipse((x-8, y-8, x+8, y+8), fill="white", outline=color, width=4)
        else:
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color)
    centered(draw, ((plot[0]+plot[2])//2, 845), "局部误差 RMS（log scale）", 19, INK)
    draw.text((75, 255), "方向统计", font=font(18), fill=INK)
    draw.rounded_rectangle((420, 315, 730, 395), radius=14, fill="#FFFFFFDD", outline=GRID)
    draw.text((440, 328), "Pearson r = 0.018   p = 0.921", font=font(18, True), fill=INK)
    draw.text((440, 360), "Spearman ρ = 0.243   p = 0.178", font=font(17), fill=MUTED)
    lx = 160
    for label, color in model_colors.items():
        draw.ellipse((lx, 875, lx+12, 887), fill=color)
        draw.text((lx+18, 866), label.replace("deepseek8b", "DeepSeek"), font=font(14), fill=MUTED)
        lx += 140

    # B: three-stage trajectories
    p2 = (910, 300, 1205, 790); axes(draw, p2)
    stages = ["output", "gradient", "SGD update"]
    stage_x = [945, 1058, 1170]
    cases = {
        "Liger": ([2.984, 2.931, 2.931], BLUE),
        "Phi": ([2.074, 4.701, 4.701], RED),
        "Qwen": ([1.008, 1.698, 1.698], GREEN),
    }
    for value in range(0, 6):
        y = p2[3] - value/5*(p2[3]-p2[1])
        draw.line((p2[0], y, p2[2], y), fill=GRID, width=1)
        draw.text((875, y-10), str(value), font=font(14), fill=MUTED)
    for name, (values, color) in cases.items():
        points = []
        for x, value in zip(stage_x, values):
            y = p2[3] - value/5*(p2[3]-p2[1]); points.append((x, y))
            draw.ellipse((x-7, y-7, x+7, y+7), fill=color)
        draw.line(points, fill=color, width=5)
    for x, label in zip(stage_x, stages):
        centered(draw, (x, 825), label, 15, MUTED)
    for index, (name, (_, color)) in enumerate(cases.items()):
        y = 865
        x = 900 + index*105
        draw.line((x, y, x+25, y), fill=color, width=5)
        draw.text((x+32, y-11), name, font=font(14), fill=INK)
    draw.text((935, 245), "同一组阶段定位实验", font=font(17), fill=MUTED)
    draw.text((940, 270), "（第三层为 stateless SGD）", font=font(15), fill=MUTED)

    # C: AUROC bars
    base_y, top_y = 790, 300
    values = [("Directionality", 0.958, BLUE), ("Update RMS", 0.542, "#AAB2BF")]
    for value in (0.0, 0.5, 1.0):
        y = base_y - value*(base_y-top_y)
        draw.line((1360, y, 1695, y), fill=GRID, width=1)
        draw.text((1315, y-10), f"{value:.1f}", font=font(15), fill=MUTED)
    for index, (label, value, color) in enumerate(values):
        x1 = 1385 + index*165; x2 = x1+110
        y = base_y - value*(base_y-top_y)
        draw.rounded_rectangle((x1, y, x2, base_y), radius=12, fill=color)
        centered(draw, ((x1+x2)//2, int(y)-30), f"{value:.3f}", 24, color, True)
        centered(draw, ((x1+x2)//2, 835), label, 16, INK)
    centered(draw, (1525, 260), "AUROC", 20, MUTED, True)
    draw.rounded_rectangle((1340, 860, 1705, 892), radius=12, fill="#FFF4E8")
    centered(draw, (1522, 875), "14 rows; 2 confirmed positives; 0543 excluded", 14, ORANGE)
    save(image, "oracle_main_results.png")


def causal_closure() -> None:
    phi = json.loads((ROOT / "results/property/direct_persistence_v4/interventions/phi_seq64_adamw_sr32.json").read_text())
    qwen = json.loads((ROOT / "results/property/direct_persistence_v4/optimizer_state/qwen_seq128_adamw_response_components.json").read_text())
    phi_parts = json.loads((ROOT / "results/property/direct_persistence_v4/optimizer_state/phi_seq64_adamw_response_components.json").read_text())
    image, draw = canvas(
        "同一 AdamW 设置中的因果闭环",
        "原实现显著、空对照精确复现、随机舍入消除持续方向；AdamW 内部两部分强烈相消",
    )
    panel(draw, (55, 175, 720, 910), "A  是否超过自己的随机抵消范围")
    panel(draw, (750, 175, 1170, 910), "B  不是简单把误差变小")
    panel(draw, (1200, 175, 1745, 910), "C  AdamW 的两部分如何相消")

    # A
    names = ["natural", "sham", "sr_0", "sr_1", "sr_2", "sr_3"]
    labels = ["natural", "sham", "SR-0", "SR-1", "SR-2", "SR-3"]
    vals = [phi["metrics"][name]["coherence_amplification"] - 1.0 for name in names]
    nulls = [phi["metrics"][name]["sign_flip_null"]["upper_95"] - 1.0 for name in names]
    plot = (120, 300, 670, 790); axes(draw, plot)
    ymax = 0.032
    for value in (0.0, 0.01, 0.02, 0.03):
        y = plot[3] - value/ymax*(plot[3]-plot[1])
        draw.line((plot[0], y, plot[2], y), fill=GRID, width=1)
        draw.text((65, y-10), f"{value:.2f}", font=font(14), fill=MUTED)
    slot = (plot[2]-plot[0])/len(names)
    for index, (label, value, null) in enumerate(zip(labels, vals, nulls)):
        x = plot[0] + slot*(index+0.5); half=27
        y = plot[3] - value/ymax*(plot[3]-plot[1])
        color = RED if index == 0 else ORANGE if index == 1 else GREEN
        draw.rounded_rectangle((x-half, y, x+half, plot[3]), radius=7, fill=color)
        yn = plot[3] - null/ymax*(plot[3]-plot[1])
        draw.line((x-half-7, yn, x+half+7, yn), fill=INK, width=4)
        centered(draw, (int(x), 825), label, 14, INK)
    draw.text((115, 255), "柱：A−1    黑线：随机抵消的 95% 上界", font=font(16), fill=MUTED)
    draw.rounded_rectangle((135, 855, 640, 890), radius=12, fill="#E7F5EE")
    centered(draw, (387, 872), "4/4 SR inside own null; sham = natural", 17, GREEN, True)

    # B: path energy ratios
    natural_energy = phi["metrics"]["natural"]["path_l2"]
    energy = [1.0] + [phi["metrics"][f"sr_{i}"]["path_l2"] / natural_energy for i in range(4)]
    labels2 = ["natural", "SR-0", "SR-1", "SR-2", "SR-3"]
    plot2 = (810, 300, 1120, 790); axes(draw, plot2)
    for value in (0.0, 0.5, 1.0):
        y = plot2[3] - value/1.15*(plot2[3]-plot2[1])
        draw.line((plot2[0], y, plot2[2], y), fill=GRID, width=1)
        draw.text((770, y-10), f"{value:.1f}", font=font(14), fill=MUTED)
    slot2=(plot2[2]-plot2[0])/len(energy)
    for i,(label,value) in enumerate(zip(labels2,energy)):
        x=plot2[0]+slot2*(i+0.5); y=plot2[3]-value/1.15*(plot2[3]-plot2[1])
        draw.rounded_rectangle((x-23,y,x+23,plot2[3]),radius=7,fill=BLUE if i==0 else GREEN)
        centered(draw,(int(x),int(y)-24),f"{value:.3f}×",14,INK,True)
        centered(draw,(int(x),825),label,13,INK)
    draw.rounded_rectangle((795, 855, 1140, 895), radius=12, fill="#EEF4FC")
    centered(draw,(967,875),"3 个 SR 能量相当或更高",17,BLUE,True)

    # C: signed shares
    p3=(1270,300,1695,790); axes(draw,p3)
    ymin,ymax=-5.0,6.0
    for value in (-4,-2,0,2,4,6):
        y=p3[3]-(value-ymin)/(ymax-ymin)*(p3[3]-p3[1])
        draw.line((p3[0],y,p3[2],y),fill=INK if value==0 else GRID,width=2 if value==0 else 1)
        draw.text((1220,y-10),str(value),font=font(14),fill=MUTED)
    shares = [
        ("Phi", phi_parts["signed_share_along_total_resultant"]),
        ("Qwen", qwen["signed_share_along_total_resultant"]),
    ]
    for i,(case,share) in enumerate(shares):
        center=1390+i*190
        for j,(key,color,label) in enumerate((("first_moment_numerator",BLUE,"numerator"),("second_moment_denominator",ORANGE,"denominator"))):
            value=share[key]; x1=center-60+j*70; x2=x1+50
            y0=p3[3]-(0-ymin)/(ymax-ymin)*(p3[3]-p3[1])
            yv=p3[3]-(value-ymin)/(ymax-ymin)*(p3[3]-p3[1])
            draw.rounded_rectangle((x1,min(y0,yv),x2,max(y0,yv)),radius=6,fill=color)
            centered(draw,((x1+x2)//2,int(yv-20 if value>0 else yv+20)),f"{value:+.2f}",14,color,True)
        centered(draw,(center-25,830),case,18,INK,True)
    draw.line((1280,870,1320,870),fill=BLUE,width=8); draw.text((1330,857),"first moment numerator",font=font(14),fill=INK)
    draw.line((1490,870,1530,870),fill=ORANGE,width=8); draw.text((1540,857),"second moment denominator",font=font(14),fill=INK)
    save(image, "phi_causal_closure.png")


def main() -> None:
    method_overview()
    oracle_results()
    causal_closure()


if __name__ == "__main__":
    main()
