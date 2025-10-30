# DataMinds Food Nutrition Detector

## Kelompok Machine Learning
- Kelompok: DataMinds
- Sandy Agre Nicola - 221110040
- ALVIN . LO - 221110546

## Tujuan
Klasifikasi foto makanan -> prediksi jenis makanan -> hitung kalori, protein, lemak, karbo per porsi gram.
Frontend (Streamlit) dan backend (FastAPI).

## 1. Setup environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Siapkan dataset (Food-101 subset)
```bash
python prepare_food101_subset.py
```
Script ini akan:
- download Food-101
- bikin folder (memisahkan train dan val):
  data/food_images/train/<kelas>/*.jpg
  data/food_images/val/<kelas>/*.jpg

## 3. Training model klasifikasi makanan
```bash
python train_food_classifier.py
```
Output:
- food_classifier_dataminds.pt (berisi weight model + nama kelas)
Program juga print val_acc tiap epoch buat laporan.

## 4. Jalankan backend (FastAPI)
```bash
uvicorn backend.api:app --reload --port 8000
```
Endpoint:
- POST /predict
  - form-data: file=<image>
  - query param: portion_g=<gram>

Response JSON: nama makanan, confidence, top5, nutrisi per 100g dan per porsi.
Url Frontend https://food-nutrient-detector.streamlit.app/ #hosting via streamlit.


## 5. Jalankan frontend (UI Streamlit)
```bash
streamlit run frontend/app_frontend.py
```
Frontend akan call backend di http://localhost:8000/predict #local
https://detect-food-nutrient-backend-production.up.railway.app/predict #hosting via railway (mungkin akan ter-shutdown dalam beberapa hari)

## 6. Estimasi nutrisi
data/nutrition_db.csv punya kalori/protein/lemak/karbo per 100g untuk setiap kelas.
Backend skala sesuai porsi gram user.
