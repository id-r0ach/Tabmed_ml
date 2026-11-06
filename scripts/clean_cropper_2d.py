import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(r"D:\Tabmed")
IMAGES_DIR = BASE_DIR / "dataset1" / "images"
YOLO_WEIGHTS = BASE_DIR / "Tabmed_ml" / "best.pt"

OUT_DIR = BASE_DIR / "Tabmed_ml" / "synthetic_data" / "crops"
CAPSULE_DIR = OUT_DIR / "capsule"
TABLET_DIR = OUT_DIR / "tablet"

CAPSULE_DIR.mkdir(parents=True, exist_ok=True)
TABLET_DIR.mkdir(parents=True, exist_ok=True)

# Очистим папки от предыдущего брака
for f in CAPSULE_DIR.glob("*.png"):
    f.unlink()
for f in TABLET_DIR.glob("*.png"):
    f.unlink()

print("Загрузка модели детекции...")
model = YOLO(str(YOLO_WEIGHTS))
image_paths = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))

saved_count = 0
rejected_count = 0

for img_p in image_paths:
    img_bgr = cv2.imread(str(img_p))
    if img_bgr is None:
        continue
    h_img, w_img = img_bgr.shape[:2]

    # Детекция
    results = model.predict(source=img_bgr, conf=0.45, verbose=False)[0]

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        bw, bh = x2 - x1, y2 - y1

        # 1. Фильтр: отсекаем таблетки, срезанные краем снимка
        if x1 <= 2 or y1 <= 2 or x2 >= w_img - 2 or y2 >= h_img - 2:
            rejected_count += 1
            continue

        # 2. Небольшой отступ для кропа
        pad = int(max(bw, bh) * 0.08)
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w_img, x2 + pad), min(h_img, y2 + pad)

        crop_bgr = img_bgr[cy1:cy2, cx1:cx2]
        if crop_bgr.size == 0:
            continue

        ch, cw = crop_bgr.shape[:2]

        # 3. Сегментация таблетки через GrabCut с подавлением цвета кожи
        # Создаем маску кожи в пространстве YCrCb
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        # Типичный диапазон кожи человека: Cr: 133-173, Cb: 77-127
        skin_mask = cv2.inRange(
            ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127])
        )

        # Инициализируем GrabCut
        gc_mask = np.zeros((ch, cw), np.uint8)
        gc_mask[:, :] = cv2.GC_PR_BGD  # все вокруг — вероятный фон

        # Центр кропа — вероятный объект
        mr_y, mr_x = int(ch * 0.15), int(cw * 0.15)
        gc_mask[mr_y : ch - mr_y, mr_x : cw - mr_x] = cv2.GC_PR_FGD

        # Области кожи гарантированно помечаем как фон
        gc_mask[skin_mask > 0] = cv2.GC_BGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(
                crop_bgr, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK
            )
        except Exception:
            rejected_count += 1
            continue

        # Бинарная маска объекта
        bin_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype("uint8")

        # 4. Анализ контура и геометрический фильтр
        contours, _ = cv2.findContours(
            bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            rejected_count += 1
            continue

        # Берем самый крупный контур в центре
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < (ch * cw * 0.20):  # слишком мелкий осколок
            rejected_count += 1
            continue

        # Выпуклая оболочка
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            rejected_count += 1
            continue

        solidity = float(area) / hull_area

        # Таблетки и капсулы всегда выпуклые (Solidity > 0.90)
        # Если прилип палец или соседняя таблетка — solidity падает
        if solidity < 0.90:
            rejected_count += 1
            continue

        # Создаем гладкую альфа-маску по контуру
        smooth_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.drawContours(smooth_mask, [c], -1, 255, -1)
        smooth_mask = cv2.GaussianBlur(smooth_mask, (3, 3), 0)

        # Собираем прозрачный RGBA PNG
        b, g, r = cv2.split(crop_bgr)
        rgba = cv2.merge([b, g, r, smooth_mask])

        # Подрезаем прозрачные поля вокруг
        ys, xs = np.where(smooth_mask > 10)
        if len(ys) == 0 or len(xs) == 0:
            rejected_count += 1
            continue
        rgba_crop = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

        target_dir = CAPSULE_DIR if cls_id == 0 else TABLET_DIR
        out_file = target_dir / f"clean_{img_p.stem}_{i}.png"
        cv2.imwrite(str(out_file), rgba_crop)
        saved_count += 1

print("\n" + "=" * 45)
print(f"Обработка завершена!")
print(f"Сохранено идеальных кропов: {saved_count}")
print(f"Отсеяно брака (пальцы, срезы, соседи): {rejected_count}")
print("=" * 45)
