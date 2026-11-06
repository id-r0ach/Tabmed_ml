import cv2
from pathlib import Path
from PIL import Image
from rembg import remove
from ultralytics import YOLO
import numpy as np

# 1. Пути
BASE_DIR = Path(r"D:\Tabmed")
IMAGES_DIR = BASE_DIR / "dataset1" / "images"
YOLO_WEIGHTS = BASE_DIR / "Tabmed_ml" / "best.pt"

# Папки для выгрузки готовых кропов
OUT_DIR = BASE_DIR / "synthetic_data" / "crops"
CAPSULE_DIR = OUT_DIR / "capsule"
TABLET_DIR = OUT_DIR / "tablet"

# 2. Инициализация модели YOLO
print("Загрузка YOLO...")
model = YOLO(str(YOLO_WEIGHTS))

# Получаем список всех изображений
image_paths = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
print(f"Найдено фото для обработки: {len(image_paths)}")

padding_ratio = 0.1  # Небольшой отступ, чтобы края не срезались

for img_path in image_paths:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    h_img, w_img = img_bgr.shape[:2]

    # Запускаем детекцию
    results = model.predict(source=img_bgr, conf=0.5, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # Добавляем padding
        pw, ph = int(bw * padding_ratio), int(bh * padding_ratio)
        cx1 = max(0, int(x1 - pw))
        cy1 = max(0, int(y1 - ph))
        cx2 = min(w_img, int(x2 + pw))
        cy2 = min(h_img, int(y2 + ph))

        crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
        if crop_bgr.size == 0:
            continue

        # Конвертируем для rembg (RGB)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # МАГИЯ: Удаляем фон (остается прозрачный PNG)
        crop_transparent = remove(crop_pil)

        # Определяем папку и сохраняем
        class_name = "capsule" if int(cls) == 0 else "tablet"
        save_dir = CAPSULE_DIR if class_name == "capsule" else TABLET_DIR

        out_name = save_dir / f"{img_path.stem}_crop_{i}.png"
        crop_transparent.save(out_name, format="PNG")

    print(f"Обработан: {img_path.name}")

print("\nГотово! Проверь папки crops/capsule и crops/tablet.")
