# 圆形测量报告

---

## 项目概览

- **输入图片**：`作业2图像.png`
- **检测方法**：`OpenCV 轮廓法（Otsu + Canny 双通路）`
- **过滤策略**：`采用单一面积阈值过滤（等效半径 >= 185.0）`
- **有效圆数量**：`10`

>已过滤小圆噪点，仅保留满足尺寸阈值的有效圆

## 1. 最终标注结果

![annotated](output_annotated.png)

## 2. 关键过程图

### 2.1 灰度图
![gray](output_gray.png)

### 2.2 边缘图
![edges](output_edges.png)

### 2.3 候选轮廓图
![contours](output_contours.png)

## 3. 测量数据详细列表

| 序号 | 圆心坐标 (x, y) px | 半径 r(px) | 直径 d(px) | 是否完整 | 拟合重合率 | 边缘圆度 |
|:---:|:---:|---:|---:|:---:|---:|---:|
| 1 | (2529, 2597) | 2575 | 5150 | 否 (被截断) | 0.97 | 0.74 |
| 2 | (2573, 2594) | 2117 | 4234 | 是 | 1.00 | 0.90 |
| 3 | (2560, 2586) | 1475 | 2950 | 是 | 1.00 | 0.90 |
| 4 | (2559, 2585) | 1141 | 2282 | 是 | 0.99 | 0.50 |
| 5 | (2567, 2591) | 956 | 1912 | 是 | 0.99 | 0.87 |
| 6 | (2564, 2588) | 807 | 1614 | 是 | 1.00 | 0.90 |
| 7 | (2905, 2765) | 189 | 378 | 是 | 0.99 | 0.89 |
| 8 | (2225, 2411) | 189 | 378 | 是 | 0.98 | 0.90 |
| 9 | (2743, 2248) | 189 | 378 | 是 | 0.99 | 0.89 |
| 10 | (2387, 2929) | 188 | 376 | 是 | 0.99 | 0.89 |

## 4. 源码（节选）

```python
import json
import math
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 小圆噪点过滤阈值
MIN_RADIUS = 180
MIN_DIAMETER = 370
# 单一面积阈值：由尺寸阈值统一推导，避免多条件叠加造成边界过严
# 以更严格的等效半径作为面积阈值来源：max(180, 370/2)=185
EFFECTIVE_MIN_RADIUS = max(float(MIN_RADIUS), float(MIN_DIAMETER) / 2.0)
MIN_CIRCLE_AREA = math.pi * (EFFECTIVE_MIN_RADIUS ** 2)
ESSENTIAL_PROCESS_IMAGES = ["output_gray.png", "output_edges.png", "output_contours.png"]
OBSOLETE_PROCESS_IMAGES = ["output_blur.png", "output_thresh.png", "output_dilated.png", "output_test.png"]


def find_input_image() -> Path:
    candidates = [
        p
        for p in Path(".").iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTS
        and not p.name.startswith("output_")
        and p.name not in {"report.md", "measure_result.json"}
    ]
    if not candidates:
        raise FileNotFoundError("未找到输入图片")
    return sorted(candidates)[0]


def read_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cleanup_obsolete_outputs():
    for name in OBSOLETE_PROCESS_IMAGES:
        p = Path(name)
        if p.exists():
            p.unlink()


def detect_circles_robust(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # 1) Otsu 二值化分支
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts_th, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2) Canny 边缘分支
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    cnts_ed, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 仅保留关键过程图（A）
    cv2.imwrite("output_gray.png", gray)
    cv2.imwrite("output_edges.png", edges)

    cnts = list(cnts_th) + list(cnts_ed)
    circles = []
    h, w = gray.shape[:2]
    contour_img = img.copy()

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            continue

        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 8 or r > max(h, w) * 0.9:
            continue

        circle_area = math.pi * r * r
        area_ratio = area / circle_area if circle_area > 0 else 0.0
        perim = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0.0
        is_clipped = bx <= 10 or by <= 10 or bx + bw >= w - 10 or by + bh >= h - 10

        is_circle = False
        if circ > 0.7:
            is_circle = True
        elif aspect > 0.85 and (circ > 0.4 or area_ratio > 0.7):
            is_circle = True
        elif is_clipped and aspect > 0.65 and area_ratio > 0.5:
            is_circle = True

        if not is_circle:
            continue

        # 用户要求的小圆过滤：采用单一面积阈值（常用做法）
        if circle_area < MIN_CIRCLE_AREA:
            continue

        cv2.drawContours(contour_img, [c], -1, (0, 0, 255), 2)
        circles.append((int(round(x)), int(round(y)), int(round(r)), float(area_ratio), float(circ)))

    cv2.imwrite("output_contours.png", contour_img)
    circles.sort(key=lambda x: x[2], reverse=True)

    # 去重
    final_circles = []
    for c in circles:
        x, y, r = c[:3]
        keep = True
        for fc in final_circles:
            fx, fy, fr = fc[:3]
            dist = math.hypot(x - fx, y - fy)
            if dist < min(r, fr) * 0.3 and abs(r - fr) < max(r, fr) * 0.15:
                keep = False
                break
        if keep:
            final_circles.append(c)

    return final_circles


def check_overlap(box1, box2):
    return not (
        box1[2] < box2[0]
        or box1[0] > box2[2]
        or box1[3] < box2[1]
        or box1[1] > box2[3]
    )


def draw_annotations(img, circles):
    out = img.copy()
    h, w = out.shape[:2]

    base_scale = max(0.5, w / 2000.0)
    font_scale = base_scale * 1.0
    line_thick = max(1, int(base_scale * 1.0))
    outline_thick = line_thick * 3
    circ_thick = max(2, int(base_scale * 1.5))
    dot_rad = max(3, int(base_scale * 2.0))

    palette = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 165, 255),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 255),
        (0, 128, 255),
        (255, 128, 0),
    ]

    drawn_boxes = []

    for i, c in enumerate(circles, start=1):
        x, y, r = c[:3]
        color = palette[(i - 1) % len(palette)]

        cv2.circle(out, (x, y), r, color, circ_thick)
        cv2.circle(out, (x, y), dot_rad, (0, 0, 255), -1)

        label = f"C{i}: r={r} d={2 * r}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thick
        )

        placed = False
        angles_to_try = [math.radians(deg) for deg in range(0, 360, 15)]
        best_tx, best_ty = x, y
        best_box = None

        for angle in angles_to_try:
            offset = r + int(base_scale * 10)
            tx = int(x + offset * math.cos(angle))
            ty = int(y + offset * math.sin(angle))

            if math.cos(angle) < 0:
                tx -= tw
            if math.sin(angle) < 0:
                ty -= th

            tx = max(10, min(w - tw - 10, tx))
            ty = max(th + 10, min(h - 10, ty))

            box = (tx - 5, ty - th - 5, tx + tw + 5, ty + baseline + 5)
            overlap = any(check_overlap(box, db) for db in drawn_boxes)

            if not overlap:
                best_tx, best_ty = tx, ty
                best_box = box
                placed = True
                break

        if not placed:
            angle = math.radians(45 + (i * 10))
            offset = r + int(base_scale * 20)
            tx = int(x + offset * math.cos(angle))
            ty = int(y + offset * math.sin(angle))
            tx = max(10, min(w - tw - 10, tx))
            ty = max(th + 10, min(h - 10, ty))
            best_tx, best_ty = tx, ty
            best_box = (tx - 5, ty - th - 5, tx + tw + 5, ty + baseline + 5)

        drawn_boxes.append(best_box)

        line_start_x = int(x + r * math.cos(math.atan2(best_ty - y, best_tx - x)))
        line_start_y = int(y + r * math.sin(math.atan2(best_ty - y, best_tx - x)))
        line_end_x = best_tx + tw // 2
        line_end_y = best_ty - th // 2

        cv2.line(
            out,
            (line_start_x, line_start_y),
            (line_end_x, line_end_y),
            color,
            max(1, int(base_scale * 0.8)),
        )
        cv2.putText(
            out,
            label,
            (best_tx, best_ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            outline_thick,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (best_tx, best_ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_thick,
            cv2.LINE_AA,
        )

    return out


def build_report_markdown(img_name: str, circles: list[dict]) -> str:
    md = [
        "# 圆形测量报告",
        "",
        "---",
        "",
        "## 项目概览",
        "",
        f"- **输入图片**：`{img_name}`",
        "- **检测方法**：`OpenCV 轮廓法（Otsu + Canny 双通路）`",
        f"- **过滤策略**：`采用单一面积阈值过滤（等效半径 >= {EFFECTIVE_MIN_RADIUS:.1f}）`",
        f"- **有效圆数量**：`{len(circles)}`",
        "",
        "> 说明：本次结果已过滤小圆噪点，仅保留满足尺寸阈值的有效圆，并保持表格结构不变。",
        "",
        "## 1. 最终标注结果",
        "",
        "![annotated](output_annotated.png)",
        "",
        "## 2. 关键过程图（精简版）",
        "",
        "### 2.1 灰度图",
        "![gray](output_gray.png)",
        "",
        "### 2.2 边缘图",
        "![edges](output_edges.png)",
        "",
        "### 2.3 候选轮廓图",
        "![contours](output_contours.png)",
        "",
        "## 3. 测量数据详细列表",
        "",
        "| 序号 | 圆心坐标 (x, y) px | 半径 r(px) | 直径 d(px) | 是否完整 | 拟合重合率 | 边缘圆度 |",
        "|:---:|:---:|---:|---:|:---:|---:|---:|",
    ]

    for c in circles:
        is_approx = "否 (被截断)" if c["approximate"] else "是"
        md.append(
            f"| {c['index']} | ({c['center_px']['x']}, {c['center_px']['y']}) | {c['radius_px']} | {c['diameter_px']} | {is_approx} | {c['fill_ratio']:.2f} | {c['circularity']:.2f} |"
        )

    md += [
        "",
        "## 4. 源码（节选）",
        "",
        "```python",
        Path(__file__).read_text(encoding="utf-8"),
        "```",
    ]

    return "\n".join(md)


def main():
    cleanup_obsolete_outputs()
    img_path = find_input_image()
    img = read_image_unicode(img_path)
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")

    circles = detect_circles_robust(img)
    annotated = draw_annotations(img, circles)
    cv2.imwrite("output_annotated.png", annotated)

    result = {
        "image": img_path.name,
        "method": "contours_robust_filtered",
        "filter": {
            "min_radius": MIN_RADIUS,
            "min_diameter": MIN_DIAMETER,
            "effective_min_radius": round(EFFECTIVE_MIN_RADIUS, 2),
            "min_circle_area": round(MIN_CIRCLE_AREA, 2),
        },
        "circle_count": len(circles),
        "circles": [
            {
                "index": i + 1,
                "center_px": {"x": c[0], "y": c[1]},
                "radius_px": c[2],
                "diameter_px": 2 * c[2],
                "circularity": round(c[4], 3),
                "fill_ratio": round(c[3], 3),
                "approximate": (
                    c[0] - c[2] < 10
                    or c[1] - c[2] < 10
                    or c[0] + c[2] > img.shape[1] - 10
                    or c[1] + c[2] > img.shape[0] - 10
                ),
            }
            for i, c in enumerate(circles)
        ],
    }

    Path("measure_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_text = build_report_markdown(img_path.name, result["circles"])
    Path("report.md").write_text(report_text, encoding="utf-8")

    print(f"DONE! Found {len(circles)} circles.")


if __name__ == "__main__":
    main()

```