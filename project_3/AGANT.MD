# Project 3 开发备忘

这份文件只保留开发和运行提示。正式项目说明、算法解释和实验结果见 `project_3.md`。

## 任务范围

- 主流程保持为传统机器视觉方法：SIFT + KMeans + BoVW/SPM + SVM。
- 不把 CNN、ResNet 或深度特征作为主提交方案。
- 代码保持单文件主程序形态，主文件为 `main.py`。

## 默认目录

- 数据目录：`project_3/data`
- 输出目录：`project_3/result`
- 日志目录：`project_3/logs`

`result/` 和 `logs/` 是运行产物目录，不建议提交到 Git。

## 当前默认参数

当前 `main.py` 默认参数与较优日志结果对齐：

```text
train_per_class = 150
num_clusters = 1500
max_descriptors_per_image = 800
sift_normalization = rootsift
descriptor_selection = random
power_normalize = true
sift_sampling = hybrid
dense_step = 6
dense_sizes = 16
svm_c = 1.6
svm_kernel = linear
chi2_gamma = 1.0
random_state = 42
```

当前正式结果记录在 `result/run_meta.json`，准确率约为 `0.7463`。

## 运行命令

在仓库根目录执行：

```powershell
conda activate mv-course
python .\project_3\main.py --data-dir .\project_3\data --output-dir .\project_3\result --log-dir .\project_3\logs
```

运行后检查：

- `project_3/result/classification_report.txt`
- `project_3/result/confusion_matrix.csv`
- `project_3/result/confusion_matrix.png`
- `project_3/result/run_meta.json`
- `project_3/logs/runs.csv`

## 调参建议

优先小范围调整：

1. `--svm-c`
2. `--num-clusters`
3. `--dense-step`
4. `--dense-sizes`
5. `--svm-kernel`

不要通过修改 `--train-per-class` 来追求更高准确率。这个参数改变训练测试划分，会偏离课程要求。

## Git 注意事项

不提交以下内容：

- `project_3/data/`
- `project_3/result/`
- `project_3/logs/`
- `project_3/tmp*/`
- `__pycache__/`
- PDF 文件

如果后续重新运行实验，只需要保留最终报告中要引用的关键指标，不需要把调参中间产物放进 Git。
