# Project 3 场景图像分类说明

## 1. 项目目标

本项目完成 15 类场景图像分类任务。实现路线严格保留课程要求中的传统机器视觉流程：

1. 使用 SIFT 提取局部图像描述子。
2. 使用 KMeans 构建视觉词典。
3. 使用 BoVW 和 Spatial Pyramid Matching 生成图像级特征。
4. 使用 SVM 完成分类。
5. 输出分类报告、混淆矩阵和运行参数记录。

当前代码没有使用 CNN、ResNet 或其他深度学习特征。这样做的原因是课程 PDF 的核心考察点是传统特征、视觉词典、空间金字塔和 SVM 分类器，而不是端到端神经网络。

## 2. 文件结构

```text
project_3/
  main.py          主程序，包含训练、测试、日志和结果保存逻辑
  test_main.py     单元测试，覆盖参数解析、特征构建和日志逻辑
  project_3.md     项目说明和实验记录
  AGANT.MD         开发备忘和运行提示
  data/            数据集目录，不提交到 Git
  result/          默认结果目录，不提交到 Git
  logs/            默认日志目录，不提交到 Git
```

`result/` 和 `logs/` 都是运行产物目录，已经在 `.gitignore` 中忽略。正式提交时建议只提交代码和说明文档，数据集、调参临时目录、日志和输出图片不进入 Git。

## 3. 数据划分

数据集采用 15 个类别子目录组织，每个子目录中存放该类别的图片。程序会按文件名中的数字排序，然后对每个类别执行固定划分：

- 前 150 张作为训练集。
- 剩余图片作为测试集。
- 训练集总数为 2250 张。
- 测试集总数为 2235 张。

`--train-per-class` 参数保留为命令行选项，主要用于检查或扩展实验。正式结果不建议修改该参数，否则会偏离课程要求的固定划分。

## 4. 算法流程

### 4.1 SIFT 特征提取

程序支持三种 SIFT 采样方式：

- `keypoint`：使用 OpenCV SIFT 自带关键点检测。
- `dense`：在规则网格上生成 dense SIFT 关键点。
- `hybrid`：合并自带关键点和 dense 网格关键点。

当前默认使用 `hybrid`。原因是场景分类中有些图片纹理分散，单纯依赖关键点检测可能覆盖不足；dense 采样可以补充规则区域，hybrid 则兼顾显著结构和整体纹理覆盖。

### 4.2 RootSIFT 归一化

默认使用 `rootsift`。它会先对每个 SIFT 描述子做 L1 风格归一化，再进行平方根压缩。相比普通 L2 归一化，RootSIFT 通常更适合 BoVW 这类基于视觉词频分布的表示。

### 4.3 KMeans 视觉词典

训练集中的 SIFT 描述子会输入 `MiniBatchKMeans`，聚类中心即视觉词典。每个局部描述子会映射到距离最近的视觉词。

当前默认词典大小为 `1500`。词典越大，表达能力通常越强，但 KMeans、SPM 特征构建和 SVM 训练都会更慢；词典过大也可能带来收益递减。

### 4.4 SPM 空间金字塔

程序使用三层空间金字塔：

- `1x1`
- `2x2`
- `4x4`

每个区域统计一份视觉词直方图，最终拼接成图像特征。三层总区域数为 `1 + 4 + 16 = 21`，因此当前默认特征维度为：

```text
21 * 1500 = 31500
```

SPM 比普通 BoVW 多保留了粗略空间布局信息，更适合区分具有不同结构布局的场景类别。

### 4.5 Power Normalization

默认开启 `--power-normalize`，对最终 SPM 直方图做平方根压缩。它可以削弱高频视觉词对分类器的支配，让特征分布更平滑。

### 4.6 SVM 分类器

当前代码支持三种 SVM 核函数：

- `linear`：默认配置，速度快，当前日志中表现稳定。
- `rbf`：非线性边界，适合作为对照实验。
- `chi2`：预计算卡方核，更贴合直方图特征，但计算成本更高。

当前默认使用 `linear`，`--svm-c` 默认值为 `1.6`。这与当前 `logs/runs.csv` 中较优结果保持一致。

## 5. 当前默认参数

当前 `main.py` 默认参数如下：

```text
--data-dir project_3/data
--output-dir project_3/result
--log-dir project_3/logs
--device cuda
--train-per-class 150
--num-clusters 1500
--max-descriptors-per-image 800
--sift-normalization rootsift
--descriptor-selection random
--power-normalize
--sift-sampling hybrid
--dense-step 6
--dense-sizes 16
--svm-c 1.6
--svm-kernel linear
--chi2-gamma 1.0
--random-state 42
```

`--device cuda` 是请求值。当前环境下 OpenCV 没有实际启用 CUDA SIFT，因此程序会自动回退到 CPU，并在输出和 `run_meta.json` 中记录原因。

## 6. 实验结果

当前 `result/run_meta.json` 中记录的正式结果为：

```text
accuracy: 0.7463087248322148
train_images: 2250
test_images: 2235
feature_dim: 31500
num_clusters: 1500
max_descriptors_per_image: 800
sift_sampling: hybrid
dense_step: 6
dense_sizes: [16]
svm_kernel: linear
svm_c: 1.6
actual_device: cpu
```

对应输出文件包括：

- `result/classification_report.txt`
- `result/confusion_matrix.csv`
- `result/confusion_matrix.png`
- `result/run_meta.json`

`logs/runs.csv` 会按准确率排序保存历史运行记录，便于比较不同参数组合。当前多次较优运行都集中在：

```text
num_clusters=1500
max_descriptors_per_image=800
sift_sampling=hybrid
dense_step=6
dense_sizes=16
svm_kernel=linear
svm_c=1.58 到 1.6 附近
```

这说明当前结果不是单次偶然运行，而是相邻参数区间内比较稳定的结果。

## 7. 关键优化点

### 7.1 从稀疏关键点扩展到 hybrid SIFT

只使用关键点检测时，纹理弱或结构分散的区域可能缺少描述子。加入 dense 采样后，图像区域覆盖更均匀；使用 hybrid 模式后，显著结构和规则区域都能进入视觉词典统计。

### 7.2 使用 RootSIFT 和 Power Normalization

RootSIFT 作用于局部描述子，Power Normalization 作用于最终直方图。两者都在减弱少数高响应或高频视觉词的支配，能让 SPM 特征更稳健。

### 7.3 增大视觉词典和描述子上限

默认 `num_clusters=1500`，`max_descriptors_per_image=800`。相比更小词典和更少描述子，这组参数提供了更强表达能力，同时计算成本仍可接受。

### 7.4 保留多种 SVM 核函数用于对照

`linear` 是当前默认和当前较优结果来源；`rbf` 和 `chi2` 保留为实验项。这样既能保持正式流程稳定，也方便在需要时做横向比较。

## 8. 参数调优建议

优先调整对结果影响较大的参数：

1. `--num-clusters`
   - 建议比较 `1000 / 1500 / 2000`。
   - 过小容易欠表达，过大计算成本明显上升。

2. `--svm-c`
   - 当前较优范围在 `1.58` 到 `1.6` 附近。
   - 可以小范围比较 `1.55 / 1.58 / 1.6 / 1.62 / 1.65`。

3. `--sift-sampling`
   - 建议以 `hybrid` 为主。
   - `keypoint` 可作为快速基线，`dense` 可作为覆盖性对照。

4. `--dense-step`
   - 当前默认 `6`。
   - 数值越小覆盖越密，但会显著增加计算量。

5. `--dense-sizes`
   - 当前默认 `16`。
   - 多尺度可试 `12,16,24,32`，但运行成本更高。

6. `--svm-kernel`
   - `linear` 用于稳定默认结果。
   - `rbf` 和 `chi2` 用于扩展实验。

不建议为了提高准确率修改 `--train-per-class`。该参数改变的是训练测试划分，不是算法优化。

## 9. 运行方式

在项目根目录执行：

```powershell
conda activate mv-course
python .\project_3\main.py --data-dir .\project_3\data --output-dir .\project_3\result --log-dir .\project_3\logs
```

如果只想显式复现当前默认配置，可以写成：

```powershell
python .\project_3\main.py `
  --data-dir .\project_3\data `
  --output-dir .\project_3\result `
  --log-dir .\project_3\logs `
  --num-clusters 1500 `
  --max-descriptors-per-image 800 `
  --sift-normalization rootsift `
  --descriptor-selection random `
  --power-normalize `
  --sift-sampling hybrid `
  --dense-step 6 `
  --dense-sizes 16 `
  --svm-c 1.6 `
  --svm-kernel linear `
  --random-state 42
```

## 10. 输出说明

每次运行会覆盖 `result/` 中的正式结果文件：

- `classification_report.txt`：各类别 precision、recall、f1-score 和总体准确率。
- `confusion_matrix.csv`：混淆矩阵数值。
- `confusion_matrix.png`：混淆矩阵可视化图片。
- `run_meta.json`：本次运行参数、数据规模、特征维度、设备信息和准确率。

同时会在 `logs/` 中写入：

- `run_YYYYMMDD_HHMMSS.json`：单次完整运行日志。
- `runs.csv`：历史运行汇总，按准确率排序。

## 11. 与深度学习方案的边界

使用 CNN 或 ResNet 可能获得更高准确率，但它会偏离本作业的主线要求：

- CNN 通常不使用 SIFT。
- CNN 不需要 KMeans 视觉词典。
- CNN 不使用 BoVW/SPM 直方图。
- CNN 通常使用神经网络分类头，而不是 SVM。

因此深度学习方案可以作为扩展对照，但不适合作为主提交方案。当前 `main.py` 保持单文件传统机器视觉实现，更符合课程 PDF 对代码形态和算法流程的要求。

## 12. 提交注意事项

- 建议提交 `main.py`、`test_main.py`、`project_3.md` 和必要说明文件。
- 不提交 `data/`、`result/`、`logs/` 和临时调参目录。
- 不提交 PDF、运行缓存和 `__pycache__`。
- 如果需要在报告中展示结果，可以引用 `result/classification_report.txt` 和 `result/confusion_matrix.png` 的内容，但不建议把大量运行产物放进 Git。
