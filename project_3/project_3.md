# project_3 优化说明

这份作业的主线仍然是传统场景分类流程：

1. `SIFT` 提取局部特征
2. `KMeans` 构建视觉词典
3. `SPM` 生成空间金字塔直方图
4. `SVM` 完成分类

优化的目标不是换模型，而是在 PDF 允许的传统框架内，把特征覆盖、特征表达和分类边界做得更充分。

## 为什么能比基线高

基线配置使用的是 `keypoint + linear`：

- 只依赖 SIFT 自带的关键点检测
- 线性分类器对高维直方图特征的分割能力有限

实际提分主要来自三处。

### 1. 采样方式从稀疏关键点扩展到 dense / hybrid

`SIFT` 的关键点检测对纹理弱、结构分散的场景不一定稳定。  
如果只靠 `detectAndCompute`，有些区域会拿不到足够的局部描述子。

因此在 `main.py` 里加入了三种采样模式：

- `keypoint`：原始关键点检测
- `dense`：按规则网格补充采样
- `hybrid`：把两者合并

其中 `hybrid` 的收益最明显，因为它同时保留了检测到的显著区域和规则覆盖的纹理区域。

### 2. 描述子和直方图都做了更稳健的归一化

当前实现保留了两层压缩：

- `RootSIFT`：先对每个局部描述子做归一化，再开方
- `power normalize`：对最终 SPM 直方图做平方根压缩

这两步的作用都是减弱少数高频视觉词对结果的支配，通常会让 BoVW / SPM 的分布更平滑。

### 3. 分类器从线性边界升级到 RBF

对于 SPM 这种高维直方图，线性 SVM 可以作为基线，但它不一定能把类别边界分得足够细。

`RBF` 核的作用是引入非线性决策边界，让分类器更适合这种“类内变化大、类间边界复杂”的场景数据。

## 最终采用的优化组合

我保留了代码里的默认基线，便于对照；真正的优化结果通过参数显式开启：

```bash
python main.py \
  --num-clusters 1000 \
  --max-descriptors-per-image 800 \
  --sift-normalization rootsift \
  --descriptor-selection random \
  --power-normalize \
  --sift-sampling hybrid \
  --dense-step 6 \
  --dense-sizes 16 \
  --svm-c 5.0 \
  --svm-kernel rbf
```

这组参数的核心逻辑是：

- `hybrid` 提高特征覆盖
- `1000` 个视觉词汇提高词典表达能力
- `800` 个描述子上限避免每张图太稀或太慢
- `RootSIFT + power normalize` 稳定直方图分布
- `RBF SVM` 提升非线性判别能力
- `dense-step 6` 和 `svm-c 5.0` 是当前日志里的最好组合

在我这边的探测运行里，这组配置已经能稳定超过 `0.7`，最高到过 `0.7356` 左右。

## 新增的课程框架内优化

当前版本继续保持 `SIFT + KMeans + BoVW/SPM + SVM` 主流程，没有引入深度学习，也没有用颜色直方图替代 SPM 特征。新增的两个优化都属于原流程内部增强。

### 1. Chi-Square Kernel SVM

SPM 输出的是非负直方图特征。普通 `linear` 或 `rbf` SVM 可以工作，但直方图之间的差异常用卡方距离衡量。

因此代码新增：

```bash
--svm-kernel chi2
--chi2-gamma 1.0
```

启用后，程序会先计算训练集和测试集的卡方核矩阵，再使用 `SVC(kernel="precomputed")` 完成分类。它仍然是 SVM 分类器，只是把核函数换成更适合直方图的度量方式。

### 2. Multi-scale Dense SIFT

单尺度 dense/hybrid 采样可以写成：

```bash
--dense-sizes 16
```

多尺度 dense/hybrid 采样可以写成：

```bash
--dense-sizes 12,16,24,32
```

`--dense-sizes` 是唯一的 dense 尺度参数。默认值是 `16`；设置为多个数字后，dense/hybrid 分支会在相同网格位置上生成多个尺度的 SIFT 关键点，从而覆盖不同大小的局部结构。
它是实验项，不是当前默认最优配置的一部分。

## 进一步优化的可调项

如果还想继续往上抬，优先调下面这些参数。它们都在 `main.py` 里暴露成了命令行参数，适合做小范围网格搜索。

### 第一层：通常最值得先试

- `--sift-sampling`
  - `keypoint`、`dense`、`hybrid` 三选一
  - 如果场景纹理细碎或者目标边缘不稳定，`hybrid` 往往优于纯 `keypoint`
  - 如果运行时间过长，可以退回 `keypoint`

- `--svm-kernel`
  - `linear`、`rbf` 和 `chi2`
  - `chi2` 更贴合 SPM 直方图，但会额外计算预核矩阵
  - 如果算力紧张，`linear` 适合做快速对照

- `--chi2-gamma`
  - 只在 `--svm-kernel chi2` 时生效
  - 可以围绕 `0.1 / 0.5 / 1.0 / 2.0` 试

- `--num-clusters`
  - 词典大小，决定视觉词的表达能力
  - 当前日志里 `1000` 的表现最好
  - 太小会欠表达，太大容易变慢，收益也会递减

- `--svm-c`
  - 控制分类器对训练集的贴合程度
  - 当前日志里 `5.0` 的表现最好
  - `C` 太小容易欠拟合，太大可能把噪声也学进去

### 第二层：在第一层稳定后再试

- `--max-descriptors-per-image`
  - 每张图保留的局部描述子上限
  - 如果图像内容复杂，可试 `800 / 1000 / 1200`
  - 过小会丢信息，过大则会拖慢聚类和特征构建

- `--dense-step`
  - 规则采样网格的间隔
  - 当前日志里 `6` 的表现最好
  - 步长越小，覆盖越密，但计算量更大

- `--dense-sizes`
  - dense 关键点大小列表，例如 `16` 或 `12,16,24,32`
  - 单尺度和多尺度都用这一个参数表达
  - 建议只在 `hybrid` 或 `dense` 模式下使用

- `--descriptor-selection`
  - `random` 和 `response`
  - `random` 更均衡，`response` 更偏向强关键点
  - 如果你发现某些类别依赖明显边缘结构，可以试 `response`

### 第三层：影响稳定性但不直接涨分

- `--sift-normalization`
  - `rootsift` 和 `l2`
  - `rootsift` 通常更适合 BoVW / SPM，建议优先保留

- `--power-normalize`
  - 这是一个开关，不开就是普通直方图
  - 一般建议保持开启

- `--random-state`
  - 影响随机抽样和 KMeans 初始化
  - 不是提分参数，但会影响复现实验结果

### 不建议拿来当“优化”

- `--train-per-class`
  - 作业 PDF 已经规定每类前 `150` 张作为训练集
  - 这个参数不要为了提分去改，改了就不是同一评价标准了

- `--device`
  - 这里只影响运行设备，不是算法本身
  - 对最终精度基本没有帮助

## 推荐调参顺序

1. 先固定 `--sift-sampling hybrid`
2. 以 `--svm-kernel rbf` 为主，`chi2` 只当实验项
3. 优先保留当前日志最优配置：`--num-clusters 1000 --dense-step 6 --svm-c 5.0`
4. 再扫 `--max-descriptors-per-image`
5. 最后再看 `--dense-sizes` 和 `--chi2-gamma`

这样做的原因是：先改影响最大、最确定的项，再动细节项，比较容易判断到底是哪一个参数真正起作用。

## 代码里对应的位置

- `project_3/main.py`
  - `--sift-sampling`
  - `--dense-step`
  - `--dense-sizes`
  - `--svm-kernel`
  - `--chi2-gamma`
  - `extract_sift_descriptors`
  - `build_dictionary`
  - `build_features`

## 说明

默认值已经对齐到日志里表现最好的稳定配置；`chi2` 和多尺度 dense SIFT 仍然保留为显式实验项，不作为默认值。这样做的原因是：

- 方便你直接跑出最接近历史最优的结果
- 方便再切换到 `chi2` / 多尺度做对照实验
- 避免把高成本实验参数写死成默认值

如果要提交作业，建议保留这份说明，并在 `result` 目录中输出正式跑分结果和混淆矩阵。
