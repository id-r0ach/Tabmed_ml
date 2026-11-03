import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from ultralytics import YOLO
import matplotlib.pyplot as plt

# python pipeline_e2e.py test.jpg --conf 0.35 --save


class TabmedPipeline:
    def __init__(self, weights_path: str | Path):
        """
        Единая модель для всего пайплайна.
        Сегментатор находит полигоны, зануляет фон и нативно отдает эмбеддинги.
        """
        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Файл весов не найден: {weights_path.resolve()}")

        self.model = YOLO(str(weights_path))

    def process(
        self, image_path: str | Path, conf: float = 0.35, dist_thresh: float = 0.38
    ) -> tuple[dict, np.ndarray, list[dict]]:
        img_path = Path(image_path)
        if not img_path.is_file():
            raise FileNotFoundError(f"Файл не найден: {img_path}")

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError("Ошибка чтения изображения")

        h, w = img.shape[:2]

        # --- ЭТАП 1: СЕГМЕНТАЦИЯ И ИЗОЛЯЦИЯ ОТ ФОНА ---
        results = self.model.predict(source=img, conf=conf, verbose=False)
        r = results[0]

        if r.masks is None or len(r.masks.data) == 0:
            return (
                {"source_image": img_path.name, "total_detected": 0, "drug_groups": []},
                img,
                [],
            )

        masks_data = r.masks.data.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        crops = []
        crop_meta = []

        for i in range(len(masks_data)):
            # 1. Ресайз бинарной маски объекта до размеров оригинального кадра
            # Вместо cv2.INTER_NEAREST используем линейную интерполяцию и порог
            mask_float = cv2.resize(
                masks_data[i], (w, h), interpolation=cv2.INTER_LINEAR
            )
            mask = (mask_float > 0.5).astype(np.uint8)

            # 2. Изоляция: зануление фона (всё за пределами таблетки = [0, 0, 0])
            isolated_bgr = cv2.bitwise_and(img, img, mask=mask)

            # 3. Вырезка чистого кропа по координатам рамки
            x1, y1, x2, y2 = map(int, boxes[i])
            crop = isolated_bgr[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)]

            if crop.size == 0:
                continue

            crops.append(crop)
            cls_name = r.names[int(classes[i])]
            crop_meta.append(
                {
                    "crop_id": i,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": round(float(confs[i]), 4),
                    "class_id": int(classes[i]),
                    "class_name": cls_name,
                    "mask_full": mask,  # Сохраняем маску для красивой визуализации
                }
            )

        if not crops:
            return (
                {"source_image": img_path.name, "total_detected": 0, "drug_groups": []},
                img,
                [],
            )

        # --- ЭТАП 2: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ (NATIVE YOLO EMBED) ---
        embed_results = self.model.embed(source=crops, verbose=False)

        feats = []
        for emb in embed_results:
            vec = emb.cpu().numpy().flatten()
            feats.append(vec)

        feats = np.array(feats)

        # L2-нормализация (строго обязательно для косинусного расстояния)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        feats_norm = feats / norms

        # --- ЭТАП 3: КЛАСТЕРИЗАЦИЯ ОЧИЩЕННЫХ КРОПОВ ---
        if len(crops) == 1:
            labels = [0]
        else:
            dist_matrix = cosine_distances(feats_norm)

            # Жесткий фильтр: капсулы и таблетки разводятся искусственным расстоянием
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

        for meta, lbl in zip(crop_meta, labels):
            meta["cluster_id"] = int(lbl)

        # --- СБОРКА ИТОГОВОГО ОТЧЕТА ---
        groups = {}
        for meta in crop_meta:
            groups.setdefault(meta["cluster_id"], []).append(meta)

        drug_groups = []
        for g_id, items in groups.items():
            classes_in_group = [it["class_name"] for it in items]
            dominant = max(set(classes_in_group), key=classes_in_group.count)

            # Удаляем тензор маски из JSON-отчета, он нужен только для отрисовки
            clean_items = []
            for it in items:
                clean_it = {k: v for k, v in it.items() if k != "mask_full"}
                clean_items.append(clean_it)

            drug_groups.append(
                {
                    "group_id": g_id,
                    "dominant_form": dominant,
                    "count": len(clean_items),
                    "items": clean_items,
                }
            )

        report = {
            "source_image": img_path.name,
            "total_detected": len(crops),
            "total_unique_drugs": len(drug_groups),
            "drug_groups": drug_groups,
        }

        return report, img, crop_meta


def save_visual_report(
    img_bgr: np.ndarray, crops_meta: list[dict], report: dict, out_path: Path
):
    """Отрисовывает полупрозрачные пиксельные маски и контуры кластеров."""
    if not crops_meta:
        return

    overlay = img_bgr.copy()
    output = img_bgr.copy()

    # Генерация палитры цветов для кластеров
    cmap = plt.get_cmap("tab10")
    colors = {
        g: (
            int(cmap(g % 10)[2] * 255),
            int(cmap(g % 10)[1] * 255),
            int(cmap(g % 10)[0] * 255),
        )
        for g in range(20)
    }

    for it in crops_meta:
        x1, y1, x2, y2 = map(int, it["bbox"])
        gid = it["cluster_id"]
        color = colors.get(gid, (0, 255, 0))
        mask = it["mask_full"]

        # Полупрозрачная заливка полигона
        colored_mask = np.zeros_like(img_bgr)
        colored_mask[mask > 0] = color
        idx = mask > 0
        overlay[idx] = colored_mask[idx]

        # Четкий контур по краю маски
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, 2)

        # Аккуратная подпись над объектом
        label = f"G{gid}:{it['class_name']}"
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    # Смешивание слоев (40% цвета маски, 60% оригинального фото)
    cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Путь к тестовому фото")
    parser.add_argument("--conf", type=float, default=0.35, help="Порог детекции")
    parser.add_argument("--dist", type=float, default=0.38, help="Порог кластеризации")
    parser.add_argument(
        "--save", action="store_true", help="Сохранить результат с разметкой"
    )
    args = parser.parse_args()

    # Строго новые веса сегментатора из последнего обучения!
    WEIGHTS = Path(
        r"D:\Tabmed\Tabmed_ml\runs\segment\tabmed_yolo26n_seg_clean\weights\best.pt"
    )

    pipeline = TabmedPipeline(weights_path=WEIGHTS)
    report_json, img_bgr, crops_meta = pipeline.process(
        args.image, args.conf, args.dist
    )

    if args.save and crops_meta:
        out_file = (
            Path(r"D:\Tabmed\Tabmed_ml\runs\pipeline") / f"e2e_{Path(args.image).name}"
        )
        save_visual_report(img_bgr, crops_meta, report_json, out_file)
        report_json["saved_visualization"] = str(out_file)

    print(json.dumps(report_json, indent=2, ensure_ascii=False))
