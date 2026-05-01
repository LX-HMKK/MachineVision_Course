# Project 3 项目说明

## 项目目标
- 完成 15 类场景图像分类任务。
- 按课程 PDF 要求使用传统机器视觉流程：SIFT + KMeans 视觉词典 + BoVW/SPM 特征 + SVM 分类器。
- 在测试集上输出分类报告和混淆矩阵。

## PDF 核心要求对应关系
- 数据集为 `scene_categories` 风格的 15 个类别目录。
- 每类图片按文件名数字排序后，前 150 张作为训练集，其余作为测试集。
- 训练集总数应为 2250 张，测试集总数应为 2235 张。
- 训练阶段：
  1. 提取训练图片 SIFT 特征。
  2. 对 SIFT 描述子做归一化。
  3. 使用 KMeans 聚类生成视觉词典。
  4. 使用 `1x1`、`2x2`、`4x4` 三层 SPM 网格生成图像特征。
  5. 使用 SVM 训练分类器。
- 测试阶段：
  1. 对测试图片执行同样的 SIFT、视觉词典映射和 SPM 特征构建流程。
  2. 使用训练好的 SVM 预测类别。
  3. 生成 `classification_report.txt`、`confusion_matrix.csv`、`confusion_matrix.png`。

## 当前实现状态
- 主程序文件：`main.py`。
- 当前实现保持单文件形态，符合“代码部分提交一个 `.py` 文件”的要求。
- 默认数据目录：`project_3/data`。
- 默认输出目录：`project_3/result`。
- 默认日志目录：`project_3/logs`。
- 当前默认关键参数：
  - `--device cuda`
  - `--train-per-class 150`
  - `--num-clusters 1000`
  - `--max-descriptors-per-image 800`
  - `--sift-normalization rootsift`
  - `--descriptor-selection random`
  - `--power-normalize` 默认开启
  - `--sift-sampling hybrid`
  - `--dense-step 6`
  - `--dense-sizes 16`
  - `--svm-c 5.0`
  - `--svm-kernel rbf`
  - `--chi2-gamma 1.0`
  - `--random-state 42`

## 可调参数说明
- `--num-clusters`：视觉词典大小。增大通常能提升表达能力，但会显著增加 KMeans、SPM 特征维度和训练耗时。
- `--log-dir`：训练日志目录。每次运行保存一份详细 JSON，并向 `runs.csv` 追加一行汇总指标，便于对比不同参数组合。
- `--max-descriptors-per-image`：每张图最多保留的 SIFT 描述子数量。过小会丢失局部信息，过大会变慢。
- `--sift-normalization`：SIFT 描述子归一化方式。`l2` 保持原始实现，`rootsift` 通常更适合 BoVW 场景分类实验。
- `--descriptor-selection`：描述子截断策略。`random` 保持当前默认流程，`response` 按关键点响应强度选择。
- `--power-normalize`：对最终 SPM 直方图做平方根压缩，可能降低高频视觉词对分类器的支配。
- `--sift-sampling`：SIFT 采样方式。`hybrid` 同时保留检测关键点和 dense 网格关键点。
- `--dense-step`：dense/hybrid 模式下网格关键点间隔。当前日志里最好的一组是 `6`。
- `--dense-sizes`：dense SIFT 的尺度列表。单尺度写 `16`，多尺度写 `12,16,24,32`。
- `--svm-c`：SVM 正则强度。当前日志里最好的一组是 `5.0`。
- `--svm-kernel`：SVM 核函数。`linear`、`rbf`、`chi2` 都保留在 SVM 分类框架内；`chi2` 会使用预计算卡方核，更适合 SPM 直方图。
- `--chi2-gamma`：卡方核的 gamma 参数，仅在 `--svm-kernel chi2` 时生效。
- `--random-state`：控制随机采样和聚类初始化，保证结果可复现。
- `--device`：设备选择参数。当前代码会检测 CUDA，但主流程仍主要依赖 OpenCV SIFT 和 scikit-learn，实际 GPU 加速能力取决于环境中的 GPU 版本 OpenCV 或 cuML 等额外库。

## 关于方案 B 与 PDF 要求的差距
- 方案 B 指使用 PyTorch CNN、ResNet 或其他深度学习图像分类模型。
- 方案 B 可能显著提高准确率，但与 PDF 要求差距较大：
  - PDF 明确要求使用 SIFT 特征，方案 B 通常不使用 SIFT。
  - PDF 要求 KMeans 生成视觉词典，方案 B 通常不需要视觉词典。
  - PDF 要求 BoVW/SPM 特征，方案 B 通常使用神经网络特征。
  - PDF 要求 SVM 分类器，方案 B 通常使用神经网络分类头。
- 因此方案 B 更像是“性能对照实验”或“扩展方案”，不适合作为本课程作业主提交代码。
- 如果目标是严格满足 PDF，应优先保留当前传统机器视觉流程；如果目标是追求更高准确率，可以另做方案 B 的实验结果，但需要在设计文档中明确说明它不是主流程。

## 精度判断
- 当前传统 BoVW + SPM + SVM 方法属于经典基线，不使用深度特征，准确率低于现代 CNN 是正常现象。
- 提升当前方案精度的优先方向：
  1. 增大 `--num-clusters`。
  2. 增大 `--max-descriptors-per-image`。
  3. 保持当前最优默认配置：`--dense-step 6`、`--svm-c 5.0`、`--svm-kernel rbf`
  4. 如需实验，再试 `--dense-sizes 12,16,24,32`
  5. 如需实验，再试 `--svm-kernel chi2` 和 `--chi2-gamma`
  6. 比较 `random` 与 `response` 两种描述子选择策略。
- 不建议为了精度直接把主流程替换成 CNN，因为这会偏离 PDF 的算法要求。

## 环境说明
- Conda 环境名：`mv-course`。
- Python 版本：`3.10`。
- 主要依赖：
  - `opencv`
  - `numpy`
  - `scipy`
  - `scikit-learn`
  - `matplotlib`
  - `pillow`
  - `tqdm`

## 运行方式
```powershell
conda activate mv-course
python .\project_3\main.py --data-dir .\project_3\data --output-dir .\project_3\result
```

## 提交注意事项
- 代码提交建议只保留 `main.py`。
- 数据集不要打包提交。
- `project_3/result/` 和 `project_3/logs/` 是运行产物目录，不建议提交。
- 设计文档 PDF 需要单独整理并随作业要求提交。
