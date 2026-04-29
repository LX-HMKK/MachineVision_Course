import json
import math
import subprocess
import sys
from pathlib import Path


REQUIRED_FILES = [
    "output_annotated.png",
    "output_gray.png",
    "output_edges.png",
    "output_contours.png",
    "report.md",
    "measure_result.json",
]

TABLE_HEADER = "| 序号 | 圆心坐标 (x, y) px | 半径 r(px) | 直径 d(px) | 是否完整 | 拟合重合率 | 边缘圆度 |"


def main() -> int:
    run = subprocess.run([sys.executable, "circle_measure.py"], check=False)
    if run.returncode != 0:
        print("[FAIL] circle_measure.py 执行失败")
        return 1

    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        print(f"[FAIL] 缺少输出文件: {missing}")
        return 1

    data = json.loads(Path("measure_result.json").read_text(encoding="utf-8"))
    circles = data.get("circles", [])
    f = data.get("filter", {})
    effective_min_radius = float(f.get("effective_min_radius", 185.0))
    min_circle_area = float(f.get("min_circle_area", math.pi * (effective_min_radius ** 2)))

    invalid = [
        c
        for c in circles
        if math.pi * (float(c.get("radius_px", 0)) ** 2) < min_circle_area
    ]
    if invalid:
        print(f"[FAIL] 仍存在未过滤的小圆: {len(invalid)} 个")
        return 1

    report = Path("report.md").read_text(encoding="utf-8")
    report_visual_section = report.split("## 4. 源码（节选）", 1)[0]
    required_refs = ["output_gray.png", "output_edges.png", "output_contours.png", "output_annotated.png"]
    if any(ref not in report_visual_section for ref in required_refs):
        print("[FAIL] report.md 缺少关键图像引用")
        return 1

    forbidden_refs = ["output_blur.png", "output_thresh.png", "output_dilated.png"]
    if any(ref in report_visual_section for ref in forbidden_refs):
        print("[FAIL] report.md 仍包含冗余过程图引用")
        return 1

    if TABLE_HEADER not in report:
        print("[FAIL] report.md 表格表头结构异常")
        return 1

    print(f"[PASS] 输出文件齐全，过滤有效。circle_count={len(circles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
