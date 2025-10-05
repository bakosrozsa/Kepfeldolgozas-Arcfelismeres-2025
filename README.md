# Arcdetektálás projekt

Ez a repó egy **Python + OpenCV alapú arcdetektálási projektet** tartalmaz.
A célunk:

* Arcok felismerése képeken
* Arcok megszámolása
* Arcok bekeretezése
* Statisztikák készítése a találati arányról

---

## Funkcionalitás (tervezett)

1. **Kép beolvasása**

   * A program betölti a bemeneti képet.

2. **Előfeldolgozás**

   * A képet szürkeárnyalatosra alakítjuk, hogy a detektálás gyorsabb és egyszerűbb legyen.

3. **Arcok detektálása**

   * OpenCV-vel (pl. Haar Cascade vagy DNN modell) arcokat keresünk a képen.
   * A talált arcokat bekeretezzük.

4. **Arcok megszámolása**

   * A program megszámolja, hány arcot talált a képen.

5. **Statisztikák kiírása**

   * Egy tesztadatbázison (kb. 8000 kép + csv fájl, amely tartalmazza az elvárt arcok számát) kiértékeljük az algoritmus pontosságát.
   * A program automatikusan ellenőrzi, hogy a megtalált arcok száma egyezik-e a várt értékkel.
   * Végül összesített statisztikát ad: hány képnél sikerült pontosan, hány képnél volt alul- vagy túlbecslés.

---

## Tesztadatok

A tesztképeket a következő Kaggle datasetből használjuk:
👉 [Count the number of faces present in an image](https://www.kaggle.com/datasets/vin1234/count-the-number-of-faces-present-in-an-image)

* ~8000 kép
* CSV fájl, amely minden képhez megadja a valós arcdarabszámot

---

## Telepítés és futtatás (kezdeti terv)

1. Klónozd a repót:

   ```bash
   git clone https://github.com/felhasznalo/arc-detektalas.git
   cd arc-detektalas
   ```

2. Telepítsd a függőségeket:

   ```bash
   pip install opencv-python numpy pandas
   ```

3. Futtatás (példa):

   ```bash
   python detect_faces.py --input teszt.jpg --output out.jpg --show
   ```

---

## Fejlesztési terv

* [x] Projekt inicializálása
* [ ] Alap arcdetektálás implementálása (kép beolvasása → szürkeárnyalat → arcok detektálása)
* [ ] Arcok bekeretezése és megszámolása
* [ ] Tesztadatbázis betöltése és kiértékelés
* [ ] Pontossági statisztikák előállítása