import cv2
import sys
import argparse
import numpy as np

# --ONNX modell útvonala--
MODEL_PATH = "pretrained_models/face_detection_yunet_2023mar.onnx"


def detect_faces_yunet(input_path, output_path):
    """
    Arcokat detektál YuNet (ONNX) modellel és elmenti az eredményt.
    """

    # --Kép beolvasása--
    image = cv2.imread(input_path)
    if image is None:
        print(f"Hiba: A kép ({input_path}) beolvasása sikertelen.")
        return

    height, width, _ = image.shape

    # --Detektor létrehozása--
    try:
        detector = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=(width, height),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000
        )
    except Exception as e:
        print("Hiba a YuNet modell betöltésekor!")
        print(f"Részletes hiba: {e}")
        return

    # --Detektálás futtatása--
    _, faces = detector.detect(image)

    framed_image = image.copy()
    count = 0

    # --Eredmények feldolgozása--
    if faces is not None:
        count = len(faces)
        for face in faces:
            box = face[0:4].astype(np.int32)
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(framed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Találat (YuNet): {count} db arc.")

    try:
        cv2.imwrite(output_path, framed_image)
        print(f"Eredmény sikeresen mentve: {output_path}")
    except Exception as e:
        print(f"Hiba a kimeneti kép mentésekor: {e}")


# --Futtatás--
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YuNet arcdetektálás GUI-hoz.")
    parser.add_argument("--input", required=True, help="Bemeneti kép.")
    parser.add_argument("--output", required=True, help="Kimeneti kép.")

    args = parser.parse_args()

    detect_faces_yunet(args.input, args.output)
