from pathlib import Path
import argparse
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"


def predict(image_path: str, conf: float = 0.25) -> dict:
    image = Path(image_path)

    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")

    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(image),
        conf=conf,
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
        description="Detect tablets and capsules in an image."
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold (default: 0.25)",
    )
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    print(predict(args.image, args.conf))


if __name__ == "__main__":
    main()
