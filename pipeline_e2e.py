import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

BASE_DIR = Path(__file__).resolve().parent

# 1. Пути к актуальным весам v8
SEG_MODEL_PATH = BASE_DIR / "weights" / "segmentation" / "best_v8_single_seg.pt"
if not SEG_MODEL_PATH.is_file():
    SEG_MODEL_PATH = (
        BASE_DIR
        / "runs"
        / "segment"
        / "tabmed_yolo26n_v8_single_seg"
        / "weights"
        / "best.pt"
    )

EMBEDDER_PATH = BASE_DIR / "weights" / "other" / "mobile_pill_embedder.pt"
if not EMBEDDER_PATH.is_file():
    EMBEDDER_PATH = BASE_DIR / "mobile_pill_embedder.pt"


def load_embedder(weights_path: Path, device: torch.device) -> nn.Module:
    """Загрузка легковесного мобильного экстрактора признаков."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(nn.Linear(in_features, 128), nn.BatchNorm1d(128))
    if weights_path.is_file():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class TabmedPipeline:
    def __init__(self, conf_thresh: float = 0.40):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Инициализация пайплайна на: {self.device}")

        # Загрузка сегментатора YOLO26n-seg
        assert SEG_MODEL_PATH.is_file(), f"Не найден сегментатор: {SEG_MODEL_PATH}"
        self.detector = YOLO(str(SEG_MODEL_PATH))
        self.conf_thresh = conf_thresh

        # Загрузка мобильного эмбеддера
        self.embedder = load_embedder(EMBEDDER_PATH, self.device)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_image(self, img_path: Path) -> dict:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Не удалось открыть: {img_path}")
        h, w = img_bgr.shape[:2]

        # 1. Инференс сегментатора (поиск объектов и масок)
        res = self.detector.predict(
            source=img_bgr, conf=self.conf_thresh, imgsz=640, verbose=False
        )[0]

        if res.masks is None or len(res.masks) == 0:
            return {
                "source_image": img_path.name,
                "total_detected": 0,
                "total_unique_drugs": 0,
                "drug_groups": [],
            }

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        masks = res.masks.data.cpu().numpy()

        crops_tensors = []
        valid_items = []

        # 2. Вырезание каждого объекта с занулением фона маской
        for idx in range(len(boxes)):
            bx1, by1, bx2, by2 = boxes[idx].astype(int)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)

            if (bx2 - bx1) < 10 or (by2 - by1) < 10:
                continue

            # Получаем бинарную маску объекта в исходном масштабе
            obj_mask = cv2.resize(masks[idx], (w, h), interpolation=cv2.INTER_NEAREST)
            crop_mask = obj_mask[by1:by2, bx1:bx2]
            crop_rgb = cv2.cvtColor(img_bgr[by1:by2, bx1:bx2], cv2.COLOR_BGR2RGB)

            # Фон -> строго [0, 0, 0]
            clean_crop = cv2.bitwise_and(crop_rgb, crop_rgb, mask=crop_mask)

            tensor = self.transform(clean_crop)
            crops_tensors.append(tensor)
            valid_items.append(
                {
                    "crop_id": idx,
                    "bbox": [round(float(v), 2) for v in [bx1, by1, bx2, by2]],
                    "confidence": round(float(confs[idx]), 4),
                }
            )

        if not crops_tensors:
            return {
                "source_image": img_path.name,
                "total_detected": 0,
                "total_unique_drugs": 0,
                "drug_groups": [],
            }

        # 3. Извлечение эмбеддингов
        batch_tensor = torch.stack(crops_tensors).to(self.device)
        with torch.no_grad():
            raw_feats = self.embedder(batch_tensor).cpu().numpy()

        # L2-нормализация векторов
        norm = np.linalg.norm(raw_feats, axis=1, keepdims=True)
        feats = raw_feats / np.maximum(norm, 1e-12)

        # 4. Кластеризация (агломеративная с косинусным расстоянием)
        n_objects = len(feats)
        if n_objects == 1:
            labels = [0]
        else:
            # Порог 0.25 отсекает разные препараты, но объединяет аверс/реверс и ракурсы
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.25,
                metric="cosine",
                linkage="average",
            )
            labels = clusterer.fit_predict(feats)

        # 5. Формирование итогового отчета
        groups_dict = {}
        for item, lbl in zip(valid_items, labels):
            lbl = int(lbl)
            if lbl not in groups_dict:
                groups_dict[lbl] = []
            groups_dict[lbl].append(item)

        drug_groups = []
        for g_id, items in groups_dict.items():
            drug_groups.append({"group_id": g_id, "count": len(items), "items": items})

        return {
            "source_image": img_path.name,
            "total_detected": len(valid_items),
            "total_unique_drugs": len(drug_groups),
            "drug_groups": drug_groups,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Tabmed Pipeline")
    parser.add_argument(
        "image", nargs="?", default="test.jpg", help="Путь к изображению"
    )
    args = parser.parse_args()

    target_img = Path(args.image)
    if not target_img.is_file():
        target_img = BASE_DIR / "test.jpg"

    pipeline = TabmedPipeline(conf_thresh=0.40)
    report = pipeline.process_image(target_img)
    print("\n=== ИТОГОВЫЙ JSON-ОТЧЕТ TABMED ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
