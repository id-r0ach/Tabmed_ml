import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from ultralytics import YOLO

# python pipeline_e2e.py test.jpg --conf 0.40 --save


class TabmedPipeline:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def process(
        self, image_path: str, conf: float = 0.25, dist_thresh: float = 0.38
    ) -> dict:
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {img_path}")

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        results = self.model.predict(source=img, conf=conf, verbose=False)
        r = results[0]

        if r.masks is None or len(r.masks.data) == 0:
            return {
                "source_image": img_path.name,
                "total_detected": 0,
                "drug_groups": [],
            }

        masks_data = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        crops = []
        crop_meta = []

        for i in range(len(masks_data)):
            # Ресайз бинарной маски
            mask = cv2.resize(
                masks_data[i], (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)

            # Попиксельное зануление фона
            isolated_bgr = cv2.bitwise_and(img, img, mask=mask)

            x1, y1, x2, y2 = map(int, boxes[i])
            crop = isolated_bgr[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]

            if crop.size == 0:
                continue

            crops.append(crop)
            cls_name = r.names[int(classes[i])]

            # Извлекаем полигон маски для красивой отрисовки
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            poly = []
            if contours:
                largest = max(contours, key=cv2.contourArea)
                eps = 0.002 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, eps, True)
                poly = approx.reshape(-1, 2).tolist()

            crop_meta.append(
                {
                    "crop_id": i,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "polygon": poly,
                    "confidence": round(float(confs[i]), 4),
                    "class_id": int(classes[i]),
                    "class_name": cls_name,
                }
            )

        if not crops:
            return {
                "source_image": img_path.name,
                "total_detected": 0,
                "drug_groups": [],
            }

        embed_results = self.model.embed(source=crops, verbose=False)

        feats = []
        for emb in embed_results:
            feats.append(emb.cpu().numpy().flatten())

        feats = np.array(feats)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        feats_norm = feats / norms

        dist_matrix = cosine_distances(feats_norm)

        for i in range(len(crop_meta)):
            for j in range(i + 1, len(crop_meta)):
                if crop_meta[i]["class_id"] != crop_meta[j]["class_id"]:
                    dist_matrix[i, j] = 2.0
                    dist_matrix[j, i] = 2.0

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_thresh,
            metric="precomputed",
            linkage="average",
        )
        labels = clusterer.fit_predict(dist_matrix).tolist()

        groups = {}
        for meta, lbl in zip(crop_meta, labels):
            groups.setdefault(lbl, []).append(meta)

        drug_groups = []
        for g_id, items in groups.items():
            classes_in_group = [it["class_name"] for it in items]
            dominant = max(set(classes_in_group), key=classes_in_group.count)
            drug_groups.append(
                {
                    "group_id": g_id,
                    "dominant_form": dominant,
                    "count": len(items),
                    "items": items,
                }
            )

        return {
            "source_image": img_path.name,
            "total_detected": len(crops),
            "total_unique_drugs": len(drug_groups),
            "drug_groups": drug_groups,
        }


def save_visual_report(img_path, report, out_dir="runs/detect/predict"):
    import matplotlib.pyplot as plt

    img = cv2.imread(str(img_path))
    overlay = img.copy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("tab10")
    for group in report["drug_groups"]:
        gid = group["group_id"]
        color = tuple(int(c * 255) for c in cmap(gid % 10)[:3][::-1])

        for item in group["items"]:
            # 1. Рисуем прозрачную заливку полигона маски
            if item.get("polygon"):
                pts = np.array(item["polygon"], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(img, [pts], True, color, 3)

            # 2. Тонкая рамка бокса и текст
            x1, y1, x2, y2 = map(int, item["bbox"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"G{gid}:{item['class_name']}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                3,
            )

    # Смешиваем заливку с оригинальным кадром
    final_img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    out_file = out_dir / f"pred_{Path(img_path).name}"
    cv2.imwrite(str(out_file), final_img)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Путь к тестовому фото")
    parser.add_argument("--conf", type=float, default=0.40, help="Порог детекции")
    parser.add_argument("--dist", type=float, default=0.38, help="Порог кластеризации")
    parser.add_argument(
        "--save", action="store_true", help="Сохранить фото с разметкой"
    )
    args = parser.parse_args()

    WEIGHTS = r"D:\Tabmed\Tabmed_ml\runs\segment\tabmed_yolo26n_seg\weights\best.pt"

    pipeline = TabmedPipeline(weights_path=WEIGHTS)
    result_json = pipeline.process(args.image, args.conf, args.dist)

    if args.save:
        saved_path = save_visual_report(args.image, result_json)
        result_json["saved_visualization"] = str(saved_path)

    # Удаляем громоздкие массивы полигонов перед печатью JSON в терминал
    for g in result_json.get("drug_groups", []):
        for item in g["items"]:
            item.pop("polygon", None)

    print(json.dumps(result_json, indent=2, ensure_ascii=False))
