import argparse
import json
import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import torch
import torch.nn as nn
from ultralytics import YOLO

# Безопасное определение базовой директории для запуска из консоли и Jupyter
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DEFAULT_MODEL = BASE_DIR / "best.pt"


class TabmedPipeline:
    """
    Сквозной пайплайн Tabmed:
    1. Детекция рамок через единую модель YOLO (best.pt).
    2. Нарезка кропов в память с уменьшенным отступом (padding 4%).
    3. Нейтрализация фона в углах рамки (эллиптическая маска).
    4. Извлечение признаков бэкбоном YOLO (слои 0-9) и AdaptiveAvgPool2d.
    5. Центрирование признаков и L2-нормализация (устранение анизотропии).
    6. Расчет матрицы косинусных расстояний с разделением разнородных классов.
    7. Агломеративная иерархическая кластеризация без учителя.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Файл весов не найден: {model_file.resolve()}")

        # Загрузка обученного детектора
        self.yolo = YOLO(str(model_file))
        self.model = self.yolo.model.to(self.device)
        self.model.eval()

        # Адаптивный пулинг для карт признаков бэкбона -> 1D-вектор
        self.pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)

    def crop_detections(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
        padding_ratio: float = 0.04,
    ) -> list[dict]:
        """Вырезает обнаруженные объекты в оперативную память."""
        h_img, w_img = img_bgr.shape[:2]
        crops_data = []

        for idx, (b, c, cf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = b
            bw = x2 - x1
            bh = y2 - y1

            pw = int(bw * padding_ratio)
            ph = int(bh * padding_ratio)

            cx1 = max(0, int(x1 - pw))
            cy1 = max(0, int(y1 - ph))
            cx2 = min(w_img, int(x2 + pw))
            cy2 = min(h_img, int(y2 + ph))

            crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
            if crop_bgr.size == 0:
                continue

            # Стандарт классов проекта: 0: capsule, 1: tablet
            class_id = int(c)
            class_name = "capsule" if class_id == 0 else "tablet"

            crops_data.append(
                {
                    "crop_id": idx,
                    "box": [round(float(v), 2) for v in b],
                    "class_id": class_id,
                    "class_name": class_name,
                    "conf": round(float(cf), 4),
                    "crop_bgr": crop_bgr,
                }
            )

        return crops_data

    def extract_features(
        self, crops_data: list[dict], img_size: int = 128
    ) -> np.ndarray:
        if not crops_data:
            return np.empty((0, 128))

        tensor_list = []
        for item in crops_data:
            bgr = item["crop_bgr"]
            h, w = bgr.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes_len = (int(w * 0.48), int(h * 0.48))
            cv2.ellipse(mask, center, axes_len, 0, 0, 360, 255, -1)

            masked_bgr = cv2.bitwise_and(bgr, bgr, mask=mask)

            resized = cv2.resize(
                masked_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor_list.append(tensor)

        batch_tensor = torch.stack(tensor_list).to(self.device)

        with torch.no_grad():
            x = batch_tensor
            for layer in self.model.model[:10]:
                x = layer(x)

            pooled = self.pool(x).flatten(1)
            feats = pooled.cpu().numpy()

            # Центрирование безопасно ТОЛЬКО при большой выборке (N >= 8).
            # Для 2-7 объектов центрирование выворачивает векторы на 180 градусов друг от друга.
            if feats.shape[0] >= 8:
                feats = feats - feats.mean(axis=0, keepdims=True)

            # L2-нормализация
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            feats_norm = feats / norms

        return feats_norm

    def cluster_crops(
        self,
        crops_data: list[dict],
        features: np.ndarray,
        dist_thresh: float = 0.70,
    ) -> list[int]:
        """Кластеризует кропы на основе матрицы косинусных расстояний."""
        n_items = len(crops_data)
        if n_items == 0:
            return []
        if n_items == 1:
            return [0]

        # Расчет матрицы косинусных расстояний
        dist_matrix = cosine_distances(features)

        # Жесткий фильтр: разнородные классы детектора не объединяются в один кластер
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if crops_data[i]["class_id"] != crops_data[j]["class_id"]:
                    dist_matrix[i, j] = 2.0
                    dist_matrix[j, i] = 2.0

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            metric="precomputed",
            linkage="average",
        )
        labels = clusterer.fit_predict(dist_matrix).tolist()
        return labels

    def process(
        self,
        image_path: Path | str,
        conf_thresh: float = 0.35,
        dist_thresh: float = 0.70,
        padding_ratio: float = 0.04,
    ) -> tuple[dict, np.ndarray, list[dict]]:
        """Полный сквозной цикл инференса."""
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось прочитать файл: {image_path}")

        # 1. Детекция объектов через обученный детектор
        results = self.yolo.predict(source=img_bgr, conf=conf_thresh, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            return (
                {
                    "source_image": Path(image_path).name,
                    "total_detected": 0,
                    "total_unique_drugs": 0,
                    "drug_groups": [],
                },
                img_bgr,
                [],
            )

        # 2. Нарезка кропов объектов
        crops_data = self.crop_detections(
            img_bgr, boxes, classes, confs, padding_ratio=padding_ratio
        )

        # 3. Извлечение признаков бэкбоном с отсечением углов
        embs = self.extract_features(crops_data)

        # 4. Кластеризация
        labels = self.cluster_crops(crops_data, embs, dist_thresh=dist_thresh)

        for it, lbl in zip(crops_data, labels):
            it["cluster_id"] = int(lbl)

        # 5. Сборка итогового JSON-отчета
        unique_groups = sorted(list(set(labels)))
        groups_report = []
        for gid in unique_groups:
            group_items = [it for it in crops_data if it["cluster_id"] == gid]
            classes_in_group = [it["class_name"] for it in group_items]
            dominant_cls = max(set(classes_in_group), key=classes_in_group.count)

            groups_report.append(
                {
                    "group_id": gid,
                    "dominant_form": dominant_cls,
                    "count": len(group_items),
                    "items": [
                        {
                            "crop_id": it["crop_id"],
                            "bbox": it["box"],
                            "confidence": it["conf"],
                        }
                        for it in group_items
                    ],
                }
            )

        report = {
            "source_image": Path(image_path).name,
            "total_detected": len(crops_data),
            "total_unique_drugs": len(unique_groups),
            "drug_groups": groups_report,
        }
        return report, img_bgr, crops_data


def save_visualization(
    img_bgr: np.ndarray, crops_data: list[dict], report: dict, out_path: Path
):
    """Отрисовывает рамки с цветовой кодировкой кластеров и сохраняет на диск."""
    viz = img_bgr.copy()
    cmap = plt.get_cmap("tab10")

    for item in crops_data:
        x1, y1, x2, y2 = map(int, item["box"])
        cid = item.get("cluster_id", 0)
        rgba = cmap(cid % 10)
        color = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
        label = f"G{cid}: {item['class_name']} ({item['conf']:.2f})"
        cv2.putText(
            viz,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), viz)


def main():
    parser = argparse.ArgumentParser(description="Tabmed End-to-End Pipeline")
    parser.add_argument(
        "image", type=str, nargs="?", default="test.jpg", help="Путь к изображению"
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL), help="Путь к весам best.pt"
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Порог детектора")
    parser.add_argument("--dist", type=float, default=0.70, help="Порог кластеризации")
    parser.add_argument(
        "--padding", type=float, default=0.04, help="Коэффициент отступа кропа"
    )
    parser.add_argument(
        "--save", action="store_true", help="Сохранить размеченное фото"
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        print(f"Ошибка: файл не найден -> {img_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    pipeline = TabmedPipeline(model_path=args.model)
    report, img_bgr, crops = pipeline.process(
        image_path=img_path,
        conf_thresh=args.conf,
        dist_thresh=args.dist,
        padding_ratio=args.padding,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.save:
        out_file = BASE_DIR / "runs" / "detect" / "predict" / f"pred_{img_path.name}"
        save_visualization(img_bgr, crops, report, out_file)
        print(f"\n[+] Размеченное фото сохранено в: {out_file}")


if __name__ == "__main__":
    main()
