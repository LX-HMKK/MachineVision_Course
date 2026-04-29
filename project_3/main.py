import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import LinearSVC
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scene classification with BoVW + SPM + SVM")
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "result")
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--num-clusters", type=int, default=200)
    parser.add_argument("--max-descriptors-per-image", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 10**9


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
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def extract_sift_descriptors(img: np.ndarray, sift, max_descriptors: int) -> Tuple[np.ndarray, np.ndarray]:
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None or len(keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 128), dtype=np.float32)

    pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    desc = descriptors.astype(np.float32)
    norms = np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
    desc = desc / norms

    if len(desc) > max_descriptors:
        idx = np.random.choice(len(desc), size=max_descriptors, replace=False)
        pts = pts[idx]
        desc = desc[idx]

    return pts, desc


def spm_histogram(pts: np.ndarray, desc: np.ndarray, img_shape: Tuple[int, int], kmeans: MiniBatchKMeans) -> np.ndarray:
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
    norm = np.linalg.norm(feature) + 1e-12
    feature /= norm
    return feature


def build_dictionary(train_paths: List[Path], sift, max_descriptors_per_image: int, n_clusters: int, random_state: int):
    all_desc = []
    for path in tqdm(train_paths, desc="Extracting SIFT for dictionary"):
        img = read_gray(path)
        _, desc = extract_sift_descriptors(img, sift, max_descriptors_per_image)
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


def build_features(paths: List[Path], sift, kmeans: MiniBatchKMeans, max_descriptors_per_image: int) -> np.ndarray:
    features = []
    for path in tqdm(paths, desc="Building SPM features"):
        img = read_gray(path)
        pts, desc = extract_sift_descriptors(img, sift, max_descriptors_per_image)
        feat = spm_histogram(pts, desc, img.shape, kmeans)
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


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_paths, train_labels, test_paths, test_labels, label_names = split_dataset(
        args.data_dir, args.train_per_class
    )

    print(f"Classes: {len(label_names)}")
    print(f"Train images: {len(train_paths)}")
    print(f"Test images: {len(test_paths)}")

    sift = create_sift()
    kmeans = build_dictionary(
        train_paths,
        sift,
        args.max_descriptors_per_image,
        args.num_clusters,
        args.random_state,
    )

    x_train = build_features(train_paths, sift, kmeans, args.max_descriptors_per_image)
    x_test = build_features(test_paths, sift, kmeans, args.max_descriptors_per_image)

    clf = LinearSVC(C=1.0, random_state=args.random_state)
    clf.fit(x_train, np.array(train_labels))
    y_pred = clf.predict(x_test)

    acc = accuracy_score(test_labels, y_pred)
    report_txt = classification_report(
        test_labels, y_pred, target_names=label_names, digits=4, zero_division=0
    )
    cm = confusion_matrix(test_labels, y_pred, labels=list(range(len(label_names))))

    print(f"Accuracy: {acc:.4f}")
    print(report_txt)

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
        "feature_dim": int(x_train.shape[1]),
        "accuracy": float(acc),
    }
    (args.output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
