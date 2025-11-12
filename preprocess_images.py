import cv2
import sys


def convert_to_grayscale(color_image):
    """
    Egy betöltött BGR (színes) képet alakít át szürkeárnyalatossá
    és visszaadja
    """
    return cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


def detect_faces_haar(input_path):
    """
    Arcokat detektál Haar Cascade-dal.
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

    faces_frontal = frontal_face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    faces_profile = profile_face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    all_faces = list(faces_frontal) + list(faces_profile)
    print(f"Találat (Összesen): {len(all_faces)} db.")

    for (x, y, w, h) in all_faces:
        cv2.rectangle(framed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Original Image", original_image)
    cv2.imshow("Detected Faces (Haar)", framed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Futtatás ---

input_image_from_project = "project_data/train/image_data/10005.jpg"
detect_faces_haar(input_image_from_project)