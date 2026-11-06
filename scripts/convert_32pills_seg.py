import json
import pandas as pd
from pathlib import Path

# Укажите пути
MEDISEG_DIR = Path(r"D:\Tabmed\Datasets\MEDISEG")
META_PATH = MEDISEG_DIR / "metadata.csv"
ANN_PATH = MEDISEG_DIR / "32pills" / "annotations.json"
LABELS_OUT_DIR = MEDISEG_DIR / "32pills" / "labels_yolo_seg"
LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_32pills_to_yolo_seg():
    print("1. Загрузка метаданных и создание маппинга классов...")
    if not META_PATH.is_file():
        print(f"[ОШИБКА] Файл метаданных не найден: {META_PATH}")
        return

    df = pd.read_csv(META_PATH)

    # Ищем капсулы и таблетки по названиям
    capsule_ids = df[df["name"].str.contains("CAP", case=False, na=False)][
        "id"
    ].tolist()
    tablet_ids = df[df["name"].str.contains("TAB", case=False, na=False)]["id"].tolist()

    print(f"   Найдено типов капсул: {len(capsule_ids)}")
    print(f"   Найдено типов таблеток: {len(tablet_ids)}")

    print("\n2. Парсинг COCO annotations.json...")
    if not ANN_PATH.is_file():
        print(f"[ОШИБКА] Файл аннотаций не найден: {ANN_PATH}")
        return

    with open(ANN_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Создаем словарь для быстрого перевода category_id в строковый id препарата (например, "HK-12345")
    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Словарь изображений
    images = {
        img["id"]: (img["file_name"], float(img["width"]), float(img["height"]))
        for img in coco["images"]
    }

    # Группируем аннотации по image_id
    img_annotations = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        if "segmentation" in ann and ann["segmentation"]:
            img_annotations[ann["image_id"]].append(ann)

    print("\n3. Конвертация полигонов в формат YOLO-seg...")
    converted_files = 0
    total_objects = 0

    for img_id, anns in img_annotations.items():
        if not anns:
            continue

        file_name, img_w, img_h = images[img_id]
        txt_path = LABELS_OUT_DIR / (Path(file_name).stem + ".txt")

        yolo_lines = []
        for ann in anns:
            cat_name = cat_id_to_name.get(ann["category_id"])

            # Назначаем целевой класс (0: capsule, 1: tablet)
            if cat_name in capsule_ids:
                class_id = 0
            else:
                class_id = 1  # Если явно не капсула, отправляем в таблетки (либо можно пропускать)

            # Обработка полигонов
            for poly in ann["segmentation"]:
                if len(poly) < 6:  # Полигон должен иметь минимум 3 точки (x,y)
                    continue

                norm_poly = []
                for i in range(0, len(poly), 2):
                    x = max(0.0, min(1.0, poly[i] / img_w))
                    y = max(0.0, min(1.0, poly[i + 1] / img_h))
                    norm_poly.extend([f"{x:.6f}", f"{y:.6f}"])

                yolo_lines.append(f"{class_id} " + " ".join(norm_poly))
                total_objects += 1

        if yolo_lines:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + "\n")
            converted_files += 1

    print(f"\n[УСПЕХ] Конвертация завершена!")
    print(f"Сохранено TXT файлов: {converted_files} в папке {LABELS_OUT_DIR}")
    print(f"Всего размечено объектов: {total_objects}")


if __name__ == "__main__":
    convert_32pills_to_yolo_seg()
