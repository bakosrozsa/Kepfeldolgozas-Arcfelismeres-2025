import cv2
import sys
import argparse


def convert_to_grayscale(color_image):
    """
    Egy betöltött BGR (színes) képet alakít át szürkeárnyalatossá
    és visszaadja
    """
    return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


def detect_faces_haar(input_path, output_path):
    """
    Arcokat detektál Haar Cascade-dal és elmenti az eredményt.
    """

    # Modellek betöltése
    frontal_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'

    frontal_face_cascade = cv2.CascadeClassifier(frontal_cascade_path)
    profile_face_cascade = cv2.CascadeClassifier(profile_cascade_path)

    if frontal_face_cascade.empty() or profile_face_cascade.empty():
        print("Hiba: Haar Cascade modell(ek) betöltése sikertelen.")
        return

    # -- Kép beolvasása --
    original_image = cv2.imread(input_path)
    if original_image is None:
        print(f"Hiba: A kép ({input_path}) beolvasása sikertelen.")
        return

    framed_image = original_image.copy()
    gray_image = convert_to_grayscale(original_image)

    faces_frontal = frontal_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_profile = profile_face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    all_faces = list(faces_frontal) + list(faces_profile)
    print(f"Találat (Összesen): {len(all_faces)} db.")

    for (x, y, w, h) in all_faces:
        cv2.rectangle(framed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    try:
        cv2.imwrite(output_path, framed_image)
        print(f"Eredmény sikeresen mentve: {output_path}")
    except Exception as e:
        print(f"Hiba a kimeneti kép mentésekor: {e}")


# --- Futtatás ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Haar Cascade arcdetektálás GUI-hoz.")
    parser.add_argument("--input", required=True, help="A bemeneti kép útvonala.")
    parser.add_argument("--output", required=True, help="A kimeneti kép mentési útvonala.")

    args = parser.parse_args()

    detect_faces_haar(args.input, args.output)