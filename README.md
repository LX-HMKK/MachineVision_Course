# 机器人学项目总览

本仓库整理了《机器人学》课程相关的三个项目/作业。根目录 README 只做总览和导航，具体实验说明、结果图和实现细节放在各自子目录中。

## 目录结构

- `poject_1/`：作业 1，`无监督学习` 相关报告，包含 Markdown 和 PDF 版本。
- `project_2/`：圆形测量项目，基于 OpenCV 对输入图像中的圆进行检测、筛选和标注。
- `project_3/`：场景分类项目，使用 `SIFT + KMeans + BoVW/SPM + SVM` 完成传统机器视觉分类流程。

## 各项目说明

### `poject_1`

- 文件：`HW1_王少卿_2300090326_无监督学习.md`
- 文件：`HW1_王少卿_2300090326_无监督学习.pdf`

这是第一份作业的报告材料，建议直接查看 PDF 用于提交效果，Markdown 版本适合快速浏览和修改。

### `project_2`

- 主要脚本：`circle_measure.py`
- 辅助脚本：`smoke_check.py`
- 说明文档：`report.md`
- 输入图片：`作业2图像.png`
- 输出结果：`output_*.png`、`measure_result.json`

这个项目的目标是对图像中的圆进行自动检测、测量和可视化标注。`report.md` 中包含了结果图、统计数据和实现说明。

### `project_3`

- 主程序：`main.py`
- 数据目录：`data/`
- 输出目录：`result/`
- 日志目录：`logs/`
- 项目说明：`AGANT.MD`

这个项目实现的是传统场景分类流程，核心步骤是提取 SIFT 特征、聚类生成视觉词典、构建多层 SPM 特征，再用 SVM 完成分类。

## 运行环境

如果你要直接运行 `project_3`，建议使用独立 Conda 环境：

```powershell
conda activate mv-course
python .\project_3\main.py --data-dir .\project_3\data --output-dir .\project_3\result
```

`project_2` 依赖 OpenCV、NumPy 等常见科学计算库；如果你已经能正常运行 `circle_measure.py`，通常不需要额外改动。

## 中文编码说明

仓库里的脚本和报告包含中文文件名、中文路径和中文内容。为了减少 Windows 下的编码问题，建议使用 PowerShell 7，并确保终端代码页和 PowerShell 输出编码为 UTF-8。

如果你已经按我之前配置过 PowerShell 7 profile，这些问题一般会明显少很多。

## 建议的工作方式

1. 先看根目录 README，确认当前项目属于哪个子目录。
2. 再进入对应子项目查看各自的说明文档。
3. 运行脚本前先确认依赖环境和输入文件是否齐全。

## 备注

- `poject_1` 是当前目录中的实际名字，我没有改动它，避免影响现有文件引用。
- `project_2` 和 `project_3` 都已经有自己的说明材料，根目录 README 只负责统一入口。
