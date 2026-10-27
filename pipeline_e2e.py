import argparse
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO

# Безопасное определение директории (работает и в CLI, и внутри интерактивного окружения)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DETECTOR_PATH = BASE_DIR / "best.pt"
EMBEDDER_PATH = BASE_DIR / "mobile_pill_embedder.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MobilePillEmbedder(nn.Module):
    """
    Легковесный мобильный экстрактор эмбеддингов (MobileNetV3-Small, ~6 МБ),
    дообученный с Triplet Loss для сжатия расстояния между аверсом и реверсом таблеток.
    """

    def __init__(self, emb_dim: int = 128):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=1)


embedder_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TabmedPipeline:
    """
    Сквозной пайплайн:
    1. Детекция объектов через YOLO (best.pt).
    2. Вырезание кропов таблеток в память с динамическим padding.
    3. Извлечение нормализованных 128-мерных эмбеддингов через MobileNetV3-Small.
    4. Иерархическая кластеризация без учителя по косинусному расстоянию.
    """

    def __init__(
        self,
        detector_path: Path = DETECTOR_PATH,
        embedder_path: Path = EMBEDDER_PATH,
        device: torch.device = DEVICE,
    ):
        if not detector_path.is_file():
            raise FileNotFoundError(f"Файл детектора не найден: {detector_path}")
        if not embedder_path.is_file():
            raise FileNotFoundError(f"Файл эмбеддера не найден: {embedder_path}")

        self.device = device
        self.detector = YOLO(str(detector_path))

        self.embedder = MobilePillEmbedder(emb_dim=128).to(device)
        self.embedder.load_state_dict(torch.load(embedder_path, map_location=device))
        self.embedder.eval()

    def crop_detections(
        self,
        img_bgr: np.ndarray,
        boxes: np.ndarray,
        classes: np.ndarray,
        confidences: np.ndarray,
        padding_ratio: float = 0.08,
    ) -> list[dict]:
        """Вырезает обнаруженные таблетки в память с динамическим отступом."""
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

            crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            crops_data.append(
                {
                    "crop_id": idx,
                    "box": [round(float(v), 2) for v in b],
                    "class_name": "capsule" if int(c) == 0 else "tablet",
                    "conf": round(float(cf), 4),
                    "crop_pil": crop_pil,
                    "crop_bgr": crop_bgr,
                }
            )

        return crops_data

    def extract_features(self, crops_data: list[dict]) -> np.ndarray:
        """Извлекает нормализованные эмбеддинги батчем."""
        if not crops_data:
            return np.empty((0, 128))

        tensors = [embedder_transform(c["crop_pil"]) for c in crops_data]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.embedder(batch).cpu().numpy()

        return embs

    def process(
        self,
        image_path: Path,
        conf_thresh: float = 0.35,
        dist_thresh: float = 0.38,
        padding_ratio: float = 0.08,
    ) -> tuple[dict, np.ndarray, list[dict]]:
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось прочитать файл: {image_path}")

        # 1. Детекция объектов
        results = self.detector.predict(
            source=img_bgr, conf=conf_thresh, verbose=False
        )[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        if len(boxes) == 0:
            return (
                {
                    "source_image": image_path.name,
                    "total_detected": 0,
                    "total_unique_drugs": 0,
                    "drug_groups": [],
                },
                img_bgr,
                [],
            )

        # 2. Нарезка кропов
        crops_data = self.crop_detections(
            img_bgr, boxes, classes, confs, padding_ratio=padding_ratio
        )

        # 3. Извлечение признаков эмбеддером
        embs = self.extract_features(crops_data)

        # 4. Кластеризация
        if len(crops_data) == 1:
            labels = [0]
        else:
            cl = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_thresh,
                metric="cosine",
                linkage="average",
            )
            labels = cl.fit_predict(embs).tolist()

        for it, lbl in zip(crops_data, labels):
            it["cluster_id"] = int(lbl)

        # 5. Сборка итогового структурированного отчета
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
            "source_image": image_path.name,
            "total_detected": len(crops_data),
            "total_unique_drugs": len(unique_groups),
            "drug_groups": groups_report,
        }
        return report, img_bgr, crops_data


def save_visual_report(
    img_bgr: np.ndarray, crops: list[dict], report: dict, out_path: Path
):
    """Сохраняет изображение с цветными рамками по группам кластеров."""
    if not crops:
        return

    viz = img_bgr.copy()
    n_groups = max(report["total_unique_drugs"], 1)
    cmap = plt.get_cmap("tab10")

    color_map = {}
    for gid in range(n_groups):
        rgba = cmap(gid % 10)
        color_map[gid] = (
            int(rgba[2] * 255),
            int(rgba[1] * 255),
            int(rgba[0] * 255),
        )

    for it in crops:
        x1, y1, x2, y2 = map(int, it["box"])
        gid = it["cluster_id"]
        color = color_map.get(gid, (0, 255, 0))

        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            viz,
            f"G{gid}:{it['class_name']}",
            (x1, max(y1 - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), viz)


def main():
    parser = argparse.ArgumentParser(
        description="Сквозной пайплайн Tabmed: детекция + кластеризация препаратов."
    )
    parser.add_argument("image", help="Путь к входному изображению")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Порог уверенности детектора (по умолчанию: 0.35)",
    )
    parser.add_argument(
        "--dist",
        type=float,
        default=0.38,
        help="Порог косинусного расстояния для кластеризации (по умолчанию: 0.38)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Сохранить результат с разметкой групп в runs/pipeline",
    )
    args = parser.parse_args()

    pipe = TabmedPipeline()
    report, img_bgr, crops = pipe.process(
        Path(args.image), conf_thresh=args.conf, dist_thresh=args.dist
    )

    if args.save:
        out_file = BASE_DIR / "runs" / "pipeline" / f"e2e_{Path(args.image).name}"
        save_visual_report(img_bgr, crops, report, out_file)
        report["saved_visualization"] = str(out_file)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
