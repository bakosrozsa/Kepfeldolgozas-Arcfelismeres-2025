# Arcfelismerés és Összehasonlító Elemzés Projekt

Ez a repó egy **Python + OpenCV + PyTorch alapú arcdetektálási projektet** tartalmaz **négy különböző algoritmus** implementációjával és interaktív GUI alkalmazásokkal.

A célunk:

* Arcok detektálása képeken négy különböző módszerrel (Haar Cascade, DNN, YuNet, és saját CNN)
* Arcok megszámolása és bekeretezése
* Algoritmusok teljesítményének összehasonlítása statisztikai módszerekkel (~5733 kép alapján)
* Interaktív GUI alkalmazások az eredmények megjelenítéséhez és elemzéséhez
* Egyedi kép feldolgozása különböző algoritmusokkal

---

## 🚀 Gyors Kezdés

1. **Telepítés**: `pip install opencv-python numpy pandas PyQt6 torch torchvision`
2. **Egyedi kép tesztelése**: `python main_window.py`
3. **Teljes összehasonlítás**: `python compare_algorithms.py`
4. **Parancssor**: `python process_yunet.py --input image.jpg --output result.jpg`

---

## Algoritmusok Áttekintése

| Algoritmus | Típus | Előnyök | Hátrányok |
|------------|-------|---------|-----------|
| **Haar Cascade** | Hagyományos feature-based | Gyors, kevés erőforrás igényes | Kevésbé pontos, sok téves detektálás |
| **DNN (ResNet-10 SSD)** | Deep Learning alapú | Pontosabb hagyományos módszereknél | Lassabb, több erőforrás igényes |
| **YuNet** | Modern ONNX alapú | Nagyon pontos, gyors inferencia | Összetett implementáció |
| **Saját CNN** | Egyedi neurális hálózat | Testreszabott megoldás | Tanítás szükséges, erőforrás igényes |

---

## Fájlok és Funkcionalitás

### Parancssori Feldolgozó Scriptek

#### 1. **preprocess_images.py** - Haar Cascade Arcdetektálás
* OpenCV Haar Cascade osztályozókkal történő arcfelismerés
* Frontális és profil nézetű arcok detektálása
* Parancssori eszköz egyetlen kép feldolgozására
* Kimenet: arcokkal bekeretezett kép

#### 2. **process_dnn.py** - DNN Alapú Arcdetektálás
* ResNet-10 SSD DNN modell használata Caffe keretrendszerrel
* Magabiztossági küszöb alapú szűrés (alapértelmezett: 50%)
* Parancssori eszköz egyetlen kép feldolgozására
* Kimenet: arcokkal bekeretezett kép

#### 3. **process_yunet.py** - YuNet Arcdetektálás
* Modern YuNet face detector ONNX modellel
* Nagy pontosságú arcfelismerés NMS (Non-Maximum Suppression) használatával
* Parancssori eszköz egyetlen kép feldolgozására
* Kimenet: arcokkal bekeretezett kép

#### 4. **scripts/run_cnn.py** - Saját CNN Modell
* Egyedi PyTorch alapú konvolúciós neurális hálózat
* Egyidejű fejlétszám becslés és bounding box detektálás
* Maximum 10 arc detektálására képes
* Parancssori eszköz egyetlen kép feldolgozására

### GUI Alkalmazások

#### 5. **main_window.py** - Egyedi Kép Feldolgozó GUI
* Interaktív alkalmazás egyetlen kép feldolgozására
* **Négy algoritmus** közül választható (Haar, DNN, YuNet, CNN)
* Bemeneti és kimeneti képek párhuzamos megjelenítése
* Valós idejű feldolgozási log megjelenítése
* Automatikus kimeneti kép mentés az `output/` mappába

#### 6. **compare_algorithms.py** - Összehasonlító GUI Alkalmazás
* **Mind a négy algoritmus** futtatása a teljes elérhető adathalmazon (~5733 kép)
* Valós idejű feldolgozási folyamat megjelenítése progress bár-al
* Szüneteltetési/folytatási funkció részleges eredmények megtekintéséhez
* 2x2-es táblázatos elrendezés minden algoritmus eredményeivel
* Automatikus teljesítmény-összehasonlítás és elemzés
* Részletes metrikák: fejlétszám pontosság, bounding box pontosság, hibák

---

## Tesztadatok és Adathalmaz

A projekt a következő Kaggle datasetet használja:
👉 **[Count the number of faces present in an image](https://www.kaggle.com/datasets/vin1234/count-the-number-of-faces-present-in-an-image)**

### Adathalmaz Statisztikák:
* **Összes kép**: ~8,196 kép az `image_data/` mappában
* **Címkézett képek**: ~5,733 kép van ellátva fejlétszám és bounding box adatokkal
* **Hiányzó címkék**: ~2,463 kép nincs címkézve (csak kép, nincs CSV adat)
* **Feldolgozott képek**: Az összehasonlító elemzés csak a címkézett képeken fut le

### CSV Fájlok:
* **`train.csv`**: Fejlétszám adatok (image_name, headcount)
* **`bbox_train.csv`**: Bounding box koordináták (image_name, xmin, ymin, xmax, ymax)

### Feldolgozási Logika:
* Az `compare_algorithms.py` csak azon képeket dolgozza fel, amelyek szerepelnek a `train.csv`-ben
* Hiányzó képeket kihagyja és figyelmeztető üzenetet ír a logba
* Minden algoritmus ugyanazon a ~5,733 képen fut le az összehasonlításhoz

---

## Telepítés és Függőségek

### 1. Repository Klónozása
```bash
git clone <repository-url>
cd Kepfeldolgozas-Arcfelismeres-2025
```

### 2. Python Környezet és Függőségek

**Minimális Python verzió**: 3.8+

**Szükséges csomagok telepítése**:
```bash
pip install opencv-python numpy pandas PyQt6 torch torchvision
```

**Ajánlott**: Virtuális környezet használata
```bash
python -m venv face_detection_env
# Windows:
face_detection_env\Scripts\activate
# Linux/Mac:
source face_detection_env/bin/activate
pip install opencv-python numpy pandas PyQt6 torch torchvision
```

### 3. Modell Fájlok Ellenőrzése

Győződj meg arról, hogy az összes pretrained modell elérhető:
- ✅ `pretrained_models/deploy.prototxt` (DNN modell architektúra)
- ✅ `pretrained_models/res10_300x300_ssd_iter_140000.caffemodel` (DNN súlyok)
- ✅ `pretrained_models/face_detection_yunet_2023mar.onnx` (YuNet modell)
- ✅ `models/face_count_bbox_cnn.pth` (Saját CNN modell)

### 4. Rendszerkövetelmények
- **RAM**: Minimum 4GB, ajánlott 8GB+
- **Tárhely**: ~2GB (modellek + adathalmaz)
- **GPU**: Opcionális, de ajánlott PyTorch CUDA támogatással gyorsabb feldolgozáshoz

### 5. Projekt Struktúra
```
Kepfeldolgozas-Arcfelismeres-2025/
├── 📁 pretrained_models/          # Pretrained modellek
│   ├── deploy.prototxt            # DNN architektúra
│   ├── res10_300x300_ssd_iter_140000.caffemodel  # DNN súlyok
│   └── face_detection_yunet_2023mar.onnx        # YuNet modell
├── 📁 models/                     # Saját modellek
│   └── face_count_bbox_cnn.pth    # Tanított CNN modell
├── 📁 scripts/                    # Segédscriptek
│   ├── run_cnn.py                 # CNN feldolgozó script
│   ├── face_count_nn.ipynb        # Jupyter notebook
│   └── train.py                   # Tanító script
├── 📁 project_data/train/         # Adathalmaz
│   ├── image_data/                # ~8196 kép
│   ├── train.csv                  # Fejlétszám adatok
│   └── bbox_train.csv             # Bounding box adatok
├── 📁 output/                     # Feldolgozott eredmények
├── 📄 compare_algorithms.py       # Összehasonlító GUI (4 algoritmus)
├── 📄 main_window.py              # Egyedi kép feldolgozó GUI
├── 📄 preprocess_images.py        # Haar Cascade script
├── 📄 process_dnn.py              # DNN script
├── 📄 process_yunet.py            # YuNet script
└── 📄 README.md                   # Ez a dokumentáció
```

---

## Használat

### GUI Alkalmazások

#### 🎯 **Egyedi Kép Feldolgozása** - `main_window.py`
```bash
python main_window.py
```

**Funkciók**:
- **4 algoritmus** közül választható (Haar Cascade, DNN, YuNet, Saját CNN)
- Interaktív képválasztás fájlböngészővel
- Script kiválasztás (bár általában az alapértelmezett script-ek használatosak)
- **Párhuzamos megjelenítés**: bemeneti kép és feldolgozott eredmény
- Valós idejű feldolgozási log és hibaüzenetek
- Automatikus eredmény mentés az `output/result.jpg` fájlba
- **Használat**: Tökéletes algoritmusok teszteléséhez és vizuális összehasonlításhoz

#### 📊 **Teljesítmény Összehasonlítás** - `compare_algorithms.py`
```bash
python compare_algorithms.py
```

**Funkciók**:
- **Mind a 4 algoritmus** automatikus futtatása ~5,733 képen
- **2x2-es GUI elrendezés**: külön táblázat minden algoritmushoz
- **Valós idejű progress**: százalékos előrehaladás és részletes log
- **Szüneteltetés/folytatás**: részleges eredmények megtekintése bármikor
- **Automatikus elemzés**: algoritmusok rangsorolása pontosság szerint
- **Részletes metrikák**:
  - Fejlétszám pontosság (%)
  - Bounding box pontosság (%)
  - Összes detektált arcok száma
  - Under-detection és over-detection hibák
  - Legjobb algoritmus kiemelése minden kategóriában

**Tippek a használathoz**:
- Szüneteltesd a folyamatot bármikor a részleges eredmények megtekintéséhez
- A teljes feldolgozás akár órákig is tarthat

### Parancssori Eszközök

**Egyedi kép feldolgozása különböző algoritmusokkal:**

**Haar Cascade** (leggyorsabb, hagyományos):
```bash
python preprocess_images.py --input image.jpg --output haar_output.jpg
```

**DNN** (kiegyensúlyozott teljesítmény):
```bash
python process_dnn.py --input image.jpg --output dnn_output.jpg
```

**YuNet** (legpontosabb, modern):
```bash
python process_yunet.py --input image.jpg --output yunet_output.jpg
```

**Saját CNN** (egyedi megoldás, fejlétszám + bounding box):
```bash
python scripts/run_cnn.py --input image.jpg --output cnn_output.jpg
```

**Példa workflow egy kép több algoritmussal való feldolgozására:**
```bash
# Haar Cascade
python preprocess_images.py --input test_image.jpg --output output_haar.jpg

# DNN
python process_dnn.py --input test_image.jpg --output output_dnn.jpg

# YuNet
python process_yunet.py --input test_image.jpg --output output_yunet.jpg

# CNN
python scripts/run_cnn.py --input test_image.jpg --output output_cnn.jpg
```

**Megjegyzés**: Az eredmények összehasonlításához használd a `main_window.py` GUI-t vagy vizuálisan hasonlítsd össze a kimeneti képeket.

---

### 💡 Használati Tippek

1. **Gyors teszteléshez**: Használd a Haar Cascade algoritmust
2. **Pontos eredményekhez**: YuNet vagy DNN algoritmusokat
3. **Teljes összehasonlításhoz**: `compare_algorithms.py` az automatikus elemzéshez
4. **Egyedi képekhez**: `main_window.py` a vizuális összehasonlításhoz
5. **Batch feldolgozáshoz**: A compare GUI alkalmas nagy mennyiségű kép elemzésére

## Fejlesztési Állapot és Mérföldkövek

### ✅ **Teljesített Feladatok**
* [x] Projekt inicializálása és alapstruktúra
* [x] **4 különböző arcfelismerő algoritmus** implementálása:
  - Haar Cascade (OpenCV hagyományos)
  - DNN ResNet-10 SSD (Caffe alapú)
  - YuNet (modern ONNX alapú)
  - Saját CNN (PyTorch egyedi modell)
* [x] Kaggle adathalmaz integrálása (~8,196 kép, ~5,733 címkézve)
* [x] Statisztikai elemzés és teljesítmény-mérés megvalósítása
* [x] **Két GUI alkalmazás** fejlesztése:
  - `main_window.py`: Egyedi kép feldolgozása 4 algoritmussal
  - `compare_algorithms.py`: Automatikus teljes adathalmaz összehasonlítás
* [x] Automatikus algoritmus-összehasonlítás
* [x] Szüneteltetési/folytatási funkció az interaktív elemzéshez
* [x] Részletes dokumentáció és README frissítése

### 🔄 **Aktuális Állapot**
* **Teljes funkcionalitás**: Minden algoritmus működik és összehasonlítható
* **GUI-k stabilak**: Mindkét alkalmazás teljesíti a követelményeket
* **Teljesítmény**: ~5,733 kép feldolgozása minden algoritmussal összehasonlítható eredményekkel