import argparse
import json
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"


def predict(image_path: str, conf: float = 0.25, save: bool = False) -> dict:
    image = Path(image_path)

    if not image.is_file():
        raise FileNotFoundError(f"Файл изображения не найден: {image}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Файл весов не найден: {MODEL_PATH}. Положите 'best.pt' в папку со скриптом."
        )

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(image),
        conf=conf,
        save=save,
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            detections.append(
                {
                    "class": result.names[class_id],
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(float(value), 2) for value in box.xyxy[0].tolist()],
                }
            )

    return {
        "image": image.name,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Детекция таблеток и капсул на изображении."
    )
    parser.add_argument("image", help="Путь к входному изображению")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности модели (от 0.0 до 1.0, по умолчанию: 0.25)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Сохранить изображение с размеченными рамками (в runs/detect/predict)",
    )
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf должен быть в диапазоне от 0.0 до 1.0")

    output = predict(args.image, args.conf, save=args.save)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
