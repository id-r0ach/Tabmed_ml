import shutil
from pathlib import Path

# --- Настройка путей ---
# Откуда берем данные (например, распакованный экспорт или рабочая папка)
SRC_IMAGES_DIR = Path(r"D:\Tabmed\Datasets\dataset_clear_mediseg\images")
SRC_LABELS_DIR = Path(r"D:\Tabmed\Datasets\MEDISEG\32pills\labels_yolo_seg")

# Куда складывать отобранные файлы
DST_DIR = Path(r"D:\Tabmed\Datasets\dataset_clear_mediseg\labels")
DST_IMAGES_DIR = DST_DIR / "images"
DST_LABELS_DIR = DST_DIR / "labels"

# Поддерживаемые форматы фотографий
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sync_and_copy_labels(
    src_img_dir: Path,
    src_lbl_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    copy_images: bool = True,
):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Индексируем все доступные текстовые файлы меток
    available_labels = {p.stem: p for p in src_lbl_dir.glob("*.txt")}

    # Ищем изображения
    image_files = [
        p
        for p in src_img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    copied_count = 0
    missing_labels = []

    for img_path in image_files:
        stem = img_path.stem

        if stem in available_labels:
            lbl_path = available_labels[stem]

            # Копируем файл метки
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

            # Копируем парную картинку (при необходимости)
            if copy_images:
                shutil.copy2(img_path, dst_img_dir / img_path.name)

            copied_count += 1
        else:
            missing_labels.append(img_path.name)

    print("=" * 50)
    print(f"Всего проверено изображений: {len(image_files)}")
    print(f"Успешно спарено и скопировано: {copied_count}")

    if missing_labels:
        print(f"Изображений без меток (пропущено): {len(missing_labels)}")
        if len(missing_labels) <= 10:
            print("Список файлов без разметки:", missing_labels)
        else:
            print(f"Примеры без разметки: {missing_labels[:5]}...")
    print(f"Файлы разметки сохранены в: {dst_lbl_dir.resolve()}")
    print("=" * 50)


# Запуск копирования
sync_and_copy_labels(
    SRC_IMAGES_DIR,
    SRC_LABELS_DIR,
    DST_IMAGES_DIR,
    DST_LABELS_DIR,
    copy_images=True,  # Поставь False, если картинки переносить не нужно, а только метки
)
