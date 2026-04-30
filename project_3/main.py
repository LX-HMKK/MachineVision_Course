import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent


# 可调参数说明：
# 本程序实现的是传统 SIFT + BoVW + SPM + SVM 基线方法，不使用深度学习特征。
# 因此 40% 左右的准确率并不异常；主要精度瓶颈在视觉词典大小、每张图保留的
# SIFT 描述子数量，以及 SVM 正则参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scene classification with BoVW + SPM + SVM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="数据目录，里面应包含 15 个类别子目录；改它只影响读取位置，不直接提升精度。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "result",
        help="输出目录，用于保存报告、混淆矩阵和运行参数；改它不影响精度。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=SCRIPT_DIR / "logs",
        help="训练日志目录；每次运行保存一份详细 JSON，并追加汇总 CSV，方便后续比较调参结果。",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cuda",
        help="设备选择开关；当前 SIFT、KMeans、SVM 主流程仍主要走 CPU，改它通常不提升精度。",
    )
    parser.add_argument(
        "--train-per-class",
        type=int,
        default=150,
        help="每类前多少张作为训练集；PDF 要求为 150，改动会偏离作业划分规则，不能作为正式结果。",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=1000,
        help="视觉词典大小，即 KMeans 聚类中心数；适当增大可提升表达能力，但会变慢并可能过拟合。",
    )
    parser.add_argument(
        "--max-descriptors-per-image",
        type=int,
        default=800,
        help="每张图片最多保留的 SIFT 描述子数量；增大通常保留更多局部信息，可能提升精度但会变慢。",
    )
    parser.add_argument(
        "--sift-normalization",
        choices=["l2", "rootsift"],
        default="rootsift",
        help="SIFT 描述子归一化方式；rootsift 更适合 BoVW 距离度量",
    )
    parser.add_argument(
        "--descriptor-selection",
        choices=["random", "response"],
        default="random",
        help="描述子超出上限时的保留策略；random 更均衡，response 偏向强关键点，需实测比较。",
    )
    parser.add_argument(
        "--power-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="对最终 SPM 直方图做平方根压缩；默认开启，可削弱高频视觉词支配，常与 rootsift 一起提升精度。",
    )
    parser.add_argument(
        "--sift-sampling",
        choices=["keypoint", "dense", "hybrid"],
        default="hybrid",
        help="SIFT 采样方式；keypoint 保持原检测器，dense 使用规则网格，hybrid 合并二者以增强场景纹理覆盖。",
    )
    parser.add_argument(
        "--dense-step",
        type=int,
        default=4,
        help="dense/hybrid 模式下网格关键点间隔；数值越小覆盖越密、计算越慢。",
    )
    parser.add_argument(
        "--dense-size",
        type=int,
        default=16,
        help="dense/hybrid 模式下每个规则关键点的 SIFT 尺度大小。",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="SVM 正则强度；增大 C 会更贴合训练集，可能提升精度，也可能过拟合。",
    )
    parser.add_argument(
        "--svm-kernel",
        choices=["linear", "rbf"],
        default="rbf",
        help="SVM 核函数；linear 保持原默认 LinearSVC，rbf 使用非线性 SVC，可能更适合直方图特征但会更慢。",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，控制描述子随机采样和 KMeans 初始化；用于复现实验，不直接提升精度。",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> Tuple[str, str]:
    if device_arg == "cpu":
        return "cpu", "Using CPU. CUDA is disabled by command line option."

    if not hasattr(cv2, "cuda"):
        note = "OpenCV was built without cv2.cuda; falling back to CPU."
        return "cpu", note

    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_count <= 0:
        note = "No CUDA-enabled device is visible to OpenCV; falling back to CPU."
        return "cpu", note

    # 这里仍然可能回退：当前 conda-forge OpenCV 通常没有 CUDA SIFT 接口。
    # 真正 GPU 加速还需要带 CUDA 模块的 OpenCV 或 cuML 等额外依赖，不适合作为单文件作业默认依赖。
    if not hasattr(cv2.cuda, "SIFT_create"):
        note = (
            f"OpenCV sees {cuda_count} CUDA device(s), but cv2.cuda.SIFT_create is unavailable; "
            "SIFT, MiniBatchKMeans and LinearSVC will run on CPU."
        )
        return "cpu", note

    note = (
        f"OpenCV sees {cuda_count} CUDA device(s), but this implementation keeps the assignment "
        "pipeline on CPU to avoid extra GPU-only dependencies."
    )
    return "cpu", note


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


# 对应 PDF 要求：每个类别按图片编号排序，前 150 张作为训练集，其余作为测试集。
def split_dataset(data_dir: Path, train_per_class: int) -> Tuple[List[Path], List[int], List[Path], List[int], List[str]]:
    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    label_names = [p.name for p in class_dirs]
    train_paths, train_labels, test_paths, test_labels = [], [], [], []

    for label_idx, class_dir in enumerate(class_dirs):
        images = sorted(class_dir.glob("*.jpg"), key=numeric_stem)
        train_imgs = images[:train_per_class]
        test_imgs = images[train_per_class:]
        train_paths.extend(train_imgs)
        train_labels.extend([label_idx] * len(train_imgs))
        test_paths.extend(test_imgs)
        test_labels.extend([label_idx] * len(test_imgs))

    return train_paths, train_labels, test_paths, test_labels, label_names


def create_sift():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create()
    raise RuntimeError("Current OpenCV build does not provide SIFT_create")


def read_gray(path: Path) -> np.ndarray:
    # OpenCV 在 Windows 中文路径下可能读图失败，使用 imdecode 更稳。
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def make_dense_keypoints(img_shape: Tuple[int, int], step: int, size: int) -> List[cv2.KeyPoint]:
    if step <= 0:
        raise ValueError("--dense-step must be positive")
    if size <= 0:
        raise ValueError("--dense-size must be positive")

    h, w = img_shape
    offset = step / 2.0
    xs = np.arange(offset, w, step, dtype=np.float32)
    ys = np.arange(offset, h, step, dtype=np.float32)
    return [cv2.KeyPoint(float(x), float(y), float(size)) for y in ys for x in xs]


def extract_sift_descriptors(
    img: np.ndarray,
    sift,
    max_descriptors: int,
    sift_normalization: str,
    descriptor_selection: str,
    sift_sampling: str = "keypoint",
    dense_step: int = 12,
    dense_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    # 提取 SIFT 后做归一化；RootSIFT 往往更适合 BoVW，但会改变实验结果。
    keypoints = []
    descriptors_list = []

    if sift_sampling in ("keypoint", "hybrid"):
        detected_keypoints, detected_descriptors = sift.detectAndCompute(img, None)
        if detected_descriptors is not None and len(detected_keypoints) > 0:
            keypoints.extend(detected_keypoints)
            descriptors_list.append(detected_descriptors)

    if sift_sampling in ("dense", "hybrid"):
        dense_keypoints = make_dense_keypoints(img.shape, dense_step, dense_size)
        dense_keypoints, dense_descriptors = sift.compute(img, dense_keypoints)
        if dense_descriptors is not None and len(dense_keypoints) > 0:
            keypoints.extend(dense_keypoints)
            descriptors_list.append(dense_descriptors)

    if sift_sampling not in ("keypoint", "dense", "hybrid"):
        raise ValueError(f"Unsupported SIFT sampling mode: {sift_sampling}")

    if descriptors_list:
        descriptors = np.vstack(descriptors_list)
    else:
        descriptors = None

    if descriptors is None or len(keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)
    desc = descriptors.astype(np.float32)
    if sift_normalization == "rootsift":
        norms = desc.sum(axis=1, keepdims=True) + 1e-12
        desc = np.sqrt(desc / norms)
    else:
        norms = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
        desc = desc / norms

    if len(desc) > max_descriptors:
        if descriptor_selection == "response":
            idx = np.argsort(responses)[-max_descriptors:]
        else:
            idx = np.random.choice(len(desc), size=max_descriptors, replace=False)
        pts = pts[idx]
        desc = desc[idx]

    return pts, desc.astype(np.float32)


def spm_histogram(
    pts: np.ndarray,
    desc: np.ndarray,
    img_shape: Tuple[int, int],
    kmeans: MiniBatchKMeans,
    power_normalize: bool,
) -> np.ndarray:
    # SPM 使用 1x1、2x2、4x4 三种尺度，共 1 + 4 + 16 = 21 个区域。
    # 每个区域统计 n_clusters 维词袋直方图，最终特征维度为 21 * n_clusters。
    n_clusters = kmeans.n_clusters
    if len(desc) == 0:
        return np.zeros(21 * n_clusters, dtype=np.float32)

    words = kmeans.predict(desc)
    h, w = img_shape
    x = np.clip(pts[:, 0], 0, w - 1)
    y = np.clip(pts[:, 1], 0, h - 1)

    levels = [1, 2, 4]
    level_weights = [0.25, 0.25, 0.5]
    feats = []

    for grid, weight in zip(levels, level_weights):
        cell_h = h / grid
        cell_w = w / grid
        for gy in range(grid):
            for gx in range(grid):
                x0, x1 = gx * cell_w, (gx + 1) * cell_w
                y0, y1 = gy * cell_h, (gy + 1) * cell_h
                mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                hist = np.bincount(words[mask], minlength=n_clusters).astype(np.float32)
                if hist.sum() > 0:
                    hist /= hist.sum()
                feats.append(hist * weight)

    feature = np.concatenate(feats, axis=0).astype(np.float32)
    if power_normalize:
        feature = np.sqrt(np.maximum(feature, 0.0)).astype(np.float32)
    norm = np.linalg.norm(feature) + 1e-12
    feature /= norm
    return feature


def build_dictionary(
    train_paths: List[Path],
    sift,
    max_descriptors_per_image: int,
    sift_normalization: str,
    descriptor_selection: str,
    sift_sampling: str,
    dense_step: int,
    dense_size: int,
    n_clusters: int,
    random_state: int,
):
    # 用训练集 SIFT 描述子聚类生成视觉词典。MiniBatchKMeans 是 KMeans 的加速版本。
    all_desc = []
    for path in tqdm(train_paths, desc="Extracting SIFT for dictionary"):
        img = read_gray(path)
        _, desc = extract_sift_descriptors(
            img,
            sift,
            max_descriptors_per_image,
            sift_normalization,
            descriptor_selection,
            sift_sampling,
            dense_step,
            dense_size,
        )
        if len(desc) > 0:
            all_desc.append(desc)

    if not all_desc:
        raise RuntimeError("No descriptors found in training set.")

    all_desc = np.vstack(all_desc)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        verbose=0,
        n_init=3,
    )
    kmeans.fit(all_desc)
    return kmeans


def build_features(
    paths: List[Path],
    sift,
    kmeans: MiniBatchKMeans,
    max_descriptors_per_image: int,
    sift_normalization: str,
    descriptor_selection: str,
    sift_sampling: str,
    dense_step: int,
    dense_size: int,
    power_normalize: bool,
) -> np.ndarray:
    # 对训练集和测试集使用同一个视觉词典，生成一致维度的 SPM 特征。
    features = []
    for path in tqdm(paths, desc="Building SPM features"):
        img = read_gray(path)
        pts, desc = extract_sift_descriptors(
            img,
            sift,
            max_descriptors_per_image,
            sift_normalization,
            descriptor_selection,
            sift_sampling,
            dense_step,
            dense_size,
        )
        feat = spm_histogram(pts, desc, img.shape, kmeans, power_normalize)
        features.append(feat)
    return np.vstack(features).astype(np.float32)


def save_confusion_matrix(cm: np.ndarray, labels: List[str], output_png: Path) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def save_training_log(
    log_dir: Path,
    args: argparse.Namespace,
    meta: dict,
    report_dict: dict,
    label_names: List[str],
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{run_id}.json"

    log_data = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "log_dir": str(args.log_dir),
            "device": args.device,
            "train_per_class": args.train_per_class,
            "num_clusters": args.num_clusters,
            "max_descriptors_per_image": args.max_descriptors_per_image,
            "sift_normalization": args.sift_normalization,
            "descriptor_selection": args.descriptor_selection,
            "power_normalize": args.power_normalize,
            "sift_sampling": args.sift_sampling,
            "dense_step": args.dense_step,
            "dense_size": args.dense_size,
            "svm_c": args.svm_c,
            "svm_kernel": args.svm_kernel,
            "random_state": args.random_state,
        },
        "dataset": {
            "train_images": meta["train_images"],
            "test_images": meta["test_images"],
            "label_names": label_names,
        },
        "device": {
            "requested_device": meta["requested_device"],
            "actual_device": meta["actual_device"],
            "device_note": meta["device_note"],
        },
        "metrics": {
            "accuracy": meta["accuracy"],
            "macro_avg": report_dict["macro avg"],
            "weighted_avg": report_dict["weighted avg"],
            "per_class": {name: report_dict[name] for name in label_names},
        },
        "feature_dim": meta["feature_dim"],
    }
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = log_dir / "runs.csv"
    csv_fields = [
        "run_id",
        "created_at",
        "num_clusters",
        "max_descriptors_per_image",
        "sift_normalization",
        "descriptor_selection",
        "power_normalize",
        "sift_sampling",
        "dense_step",
        "dense_size",
        "svm_c",
        "svm_kernel",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "feature_dim",
    ]
    row = {
        "run_id": run_id,
        "created_at": log_data["created_at"],
        "num_clusters": args.num_clusters,
        "max_descriptors_per_image": args.max_descriptors_per_image,
        "sift_normalization": args.sift_normalization,
        "descriptor_selection": args.descriptor_selection,
        "power_normalize": args.power_normalize,
        "sift_sampling": args.sift_sampling,
        "dense_step": args.dense_step,
        "dense_size": args.dense_size,
        "svm_c": args.svm_c,
        "svm_kernel": args.svm_kernel,
        "accuracy": meta["accuracy"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "feature_dim": meta["feature_dim"],
    }
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = [
                {field: old_row.get(field, "") for field in csv_fields}
                for old_row in reader
            ]
            existing_fields = reader.fieldnames or []
        if existing_fields != csv_fields:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()
                writer.writerows(existing_rows)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return log_path


def main():
    args = parse_args()
    actual_device, device_note = resolve_device(args.device)
    np.random.seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Requested device: {args.device}")
    print(f"Actual device: {actual_device}")
    print(f"Device note: {device_note}")

    # 1. 数据划分：按 PDF 要求得到 2250 张训练图和 2235 张测试图。
    train_paths, train_labels, test_paths, test_labels, label_names = split_dataset(
        args.data_dir, args.train_per_class
    )

    print(f"Classes: {len(label_names)}")
    print(f"Train images: {len(train_paths)}")
    print(f"Test images: {len(test_paths)}")

    # 2. 特征提取与词典生成：训练集 SIFT -> KMeans 视觉词典。
    sift = create_sift()
    kmeans = build_dictionary(
        train_paths,
        sift,
        args.max_descriptors_per_image,
        args.sift_normalization,
        args.descriptor_selection,
        args.sift_sampling,
        args.dense_step,
        args.dense_size,
        args.num_clusters,
        args.random_state,
    )

    # 3. 图片表示：训练集和测试集都转为 BoVW + SPM 特征向量。
    x_train = build_features(
        train_paths,
        sift,
        kmeans,
        args.max_descriptors_per_image,
        args.sift_normalization,
        args.descriptor_selection,
        args.sift_sampling,
        args.dense_step,
        args.dense_size,
        args.power_normalize,
    )
    x_test = build_features(
        test_paths,
        sift,
        kmeans,
        args.max_descriptors_per_image,
        args.sift_normalization,
        args.descriptor_selection,
        args.sift_sampling,
        args.dense_step,
        args.dense_size,
        args.power_normalize,
    )

    # 4. 执行分类：用 SVM 学习训练集特征，并预测测试集类别。
    if args.svm_kernel == "linear":
        clf = LinearSVC(C=args.svm_c, random_state=args.random_state)
    else:
        clf = SVC(C=args.svm_c, kernel=args.svm_kernel, gamma="scale")
    clf.fit(x_train, np.array(train_labels))
    y_pred = clf.predict(x_test)

    acc = accuracy_score(test_labels, y_pred)
    report_txt = classification_report(
        test_labels, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    report_dict = classification_report(
        test_labels, y_pred, target_names=label_names, digits=4, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(test_labels, y_pred, labels=list(range(len(label_names))))

    print(f"Accuracy: {acc:.4f}")
    print(report_txt)

    # 5. 结果输出：分类报告、混淆矩阵和本次运行参数，便于写设计文档。
    (args.output_dir / "classification_report.txt").write_text(
        f"Accuracy: {acc:.4f}\n\n{report_txt}", encoding="utf-8"
    )

    with (args.output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + label_names)
        for idx, row in enumerate(cm):
            writer.writerow([label_names[idx]] + row.tolist())

    save_confusion_matrix(cm, label_names, args.output_dir / "confusion_matrix.png")

    meta = {
        "data_dir": str(args.data_dir),
        "train_images": len(train_paths),
        "test_images": len(test_paths),
        "num_clusters": args.num_clusters,
        "max_descriptors_per_image": args.max_descriptors_per_image,
        "sift_normalization": args.sift_normalization,
        "descriptor_selection": args.descriptor_selection,
        "power_normalize": args.power_normalize,
        "sift_sampling": args.sift_sampling,
        "dense_step": args.dense_step,
        "dense_size": args.dense_size,
        "svm_c": args.svm_c,
        "svm_kernel": args.svm_kernel,
        "requested_device": args.device,
        "actual_device": actual_device,
        "device_note": device_note,
        "feature_dim": int(x_train.shape[1]),
        "accuracy": float(acc),
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log_path = save_training_log(args.log_dir, args, meta, report_dict, label_names)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Saved training log to: {log_path}")


if __name__ == "__main__":
    main()
