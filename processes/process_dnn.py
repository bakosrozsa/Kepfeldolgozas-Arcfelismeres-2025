import cv2
import sys
import numpy as np
import argparse

# -- Modell beállítások --
PROTOTXT_PATH = "pretrained_models/deploy.prototxt"
MODEL_PATH = "pretrained_models/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5


def detect_faces_dnn(input_path, output_path):
    """
    Arcokat detektál a DNN modellel és elmenti az eredményt.
    """

    # --Modell betöltése--
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
    except cv2.error as e:
        print(f"Hiba: A DNN modell betöltése sikertelen.")
        print(f"Ellenőrizd az útvonalakat: {PROTOTXT_PATH} és {MODEL_PATH}")
        print(f"Részletek: {e}")
        return

    # --Kép beolvasása és másolása--
    original_image = cv2.imread(input_path)
    if original_image is None:
        print(f"Hiba: A kép ({input_path}) beolvasása sikertelen.")
        return

    framed_image = original_image.copy()
    (h, w) = original_image.shape[:2]

    # --Kép előkészítése--
    blob = cv2.dnn.blobFromImage(cv2.resize(original_image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # --Detektálás futtatása--
    net.setInput(blob)
    detections = net.forward()

    found_faces = 0

    # --Eredmények feldolgozása--
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # --Szűrés a biztossági küszöb alapján--
        if confidence > CONFIDENCE_THRESHOLD:
            found_faces += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(framed_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    print(f"Találat: {found_faces} db arc ({CONFIDENCE_THRESHOLD * 100}% felett).")

    try:
        cv2.imwrite(output_path, framed_image)
        print(f"Eredmény sikeresen mentve: {output_path}")
    except Exception as e:
        print(f"Hiba a kimeneti kép mentésekor: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNN arcdetektálás GUI-hoz.")
    parser.add_argument("--input", required=True, help="A bemeneti kép útvonala.")
    parser.add_argument("--output", required=True, help="A kimeneti kép mentési útvonala.")

    args = parser.parse_args()

    detect_faces_dnn(args.input, args.output)