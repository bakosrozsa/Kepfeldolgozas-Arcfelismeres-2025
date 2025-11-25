# Arcfelismerés és Összehasonlító Elemzés Projekt

Ez a repó egy **Python + OpenCV alapú arcdetektálási projektet** tartalmaz két különböző algoritmus implementációjával és interaktív GUI alkalmazásokkal.
A célunk:

* Arcok detektálása képeken két különböző módszerrel (Haar Cascade és DNN)
* Arcok megszámolása és bekeretezése
* Algoritmusok teljesítményének összehasonlítása statisztikai módszerekkel
* Interaktív GUI alkalmazások az eredmények megjelenítéséhez és elemzéséhez

---

## Fájlok és Funkcionalitás

### 1. **preprocess_images.py** - Haar Cascade Arcdetektálás
* OpenCV Haar Cascade osztályozókkal történő arcfelismerés
* Frontális és profil nézetű arcok detektálása
* Parancssori eszköz egyetlen kép feldolgozására

### 2. **process_dnn.py** - DNN Alapú Arcdetektálás
* ResNet-10 SSD DNN modell használata Caffe keretrendszerrel
* Magabiztossági küszöb alapú szűrés
* Parancssori eszköz egyetlen kép feldolgozására

### 3. **caffe_ssd_stats.py** - DNN Statisztikai Elemzés
* DNN algoritmus teljesítményének mérése a teljes train adathalmazon
* Fejlétszám pontosság, bounding box pontosság számítása
* Alul- és túlbecslések azonosítása

### 4. **main_window.py** - Egyedi Kép Feldolgozó GUI
* Interaktív alkalmazás egyetlen kép feldolgozására
* Algoritmus kiválasztása (Haar Cascade vagy DNN)
* Bemeneti és kimeneti képek párhuzamos megjelenítése
* Valós idejű feldolgozási log megjelenítése

### 5. **compare_algorithms.py** - Összehasonlító GUI Alkalmazás
* Mindkét algoritmus (Haar Cascade és DNN) futtatása a teljes train adathalmazon
* Valós idejű feldolgozási folyamat megjelenítése
* Részletes statisztikai összehasonlítás PyQt6 GUI-ban
* Automatikus elemzés és teljesítmény-összehasonlítás

---

## Tesztadatok

A tesztképeket a következő Kaggle datasetből használjuk:
👉 [Count the number of faces present in an image](https://www.kaggle.com/datasets/vin1234/count-the-number-of-faces-present-in-an-image)

* ~8000 kép a train halmazban, ebből ~5700 adatokkal van ellátva
* CSV fájlok az elvárt arcszámokkal és bounding box koordinátákkal
* Csak a train.csv-ben szereplő képeket dolgozzuk fel

---

## Telepítés és Függőségek

1. Klónozd a repót és navigálj a könyvtárba:

   ```bash
   git clone <repository-url>
   cd Kepfeldolgozas-Arcfelismeres-2025
   ```

2. Telepítsd a függőségeket:

   ```bash
   pip install opencv-python numpy pandas PyQt6
   ```

3. Győződj meg arról, hogy a pretrained modellek elérhetők:
   - `pretrained_models/deploy.prototxt`
   - `pretrained_models/res10_300x300_ssd_iter_140000.caffemodel`

---

## Használat

### GUI Alkalmazások

**Egyedi kép feldolgozása GUI-ban:**
```bash
python main_window.py
```
A fő GUI alkalmazás lehetővé teszi:
- Kép kiválasztását és algoritmus választását (Haar Cascade vagy DNN)
- Az eredeti kép és a feldolgozott eredmény párhuzamos megjelenítését
- A talált arcok bekeretezését a kimeneti képen
- Valós idejű feldolgozási folyamat követését

**Algoritmusok összehasonlítása GUI-ban:**
```bash
python compare_algorithms.py
```
Az összehasonlító GUI alkalmazás:
- A projektben található összes algoritmust futtatja a teljes adathalmazon
- Valós idejű feldolgozási folyamatot mutat
- Összehasonlító statisztikákat jelenít meg táblázatos formában
- Automatikus elemzést és teljesítmény-összehasonlítást végez
- Részletes metrikákat mutat (fejlétszám pontosság, bounding box pontosság, stb.)

### Parancssori Eszközök

**Egyedi kép feldolgozása parancssorból:**

Haar Cascade algoritmussal:
```bash
python preprocess_images.py --input image.jpg --output output.jpg
```

DNN algoritmussal:
```bash
python process_dnn.py --input image.jpg --output output.jpg
```

**Statisztikai elemzés konzolban:**
```bash
python caffe_ssd_stats.py
```

---

## Fejlesztési Állapot

* [x] Projekt inicializálása
* [x] Haar Cascade arcdetektálás implementálása
* [x] DNN alapú arcdetektálás implementálása
* [x] Tesztadatbázis integrálása
* [x] Statisztikai elemzés megvalósítása
* [x] Egyedi kép feldolgozó GUI alkalmazás (main_window.py)
* [x] Összehasonlító GUI alkalmazás (compare_algorithms.py)
* [x] Algoritmusok összehasonlító elemzése
* [ ] További saját modell tesztelése (tervezett)