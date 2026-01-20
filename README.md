Template repository untuk proyek UAS: Implementasi Machine Learning End-to-End — Prediksi Harga Rumah.

## Struktur
- app/
  - main.py           -- FastAPI app
  - model.py          -- logic load model / predict
  - schemas.py        -- pydantic request/response
- models/             -- tempat model yang disimpan (output dari train.py)
- web/
  - index.html        -- UI sederhana (static HTML + JS)
- train.py            -- contoh script training (sklearn) untuk dataset California
- requirements.txt
- Dockerfile
- tests/
  - test_predict.py   -- unit test sederhana
- .gitignore

## Setup (local, Python)
1. Buat virtualenv dan aktifkan:
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows

2. Install dependencies:
   pip install -r requirements.txt

3. Latih model contoh:
   python train.py
   -> Ini akan menyimpan models/model.joblib dan models/feature_names.json

4. Jalankan API:
   uvicorn app.main:app --host 0.0.0.0 --port 8000

5. Buka UI:
   Buka web/index.html di browser (pastikan API berjalan di http://localhost:8000).
   Atau panggil endpoint:
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"features": {"MedInc":8,"HouseAge":30,"AveRooms":5,"AveBedrms":1,"Population":1000,"AveOccup":3,"Latitude":34,"Longitude":-118}}'

## Docker
Build:
  docker build -t uas-ml-housing:latest .

Run:
  docker run --rm -p 8000:80 uas-ml-housing:latest

API akan tersedia di http://localhost:8000

## Notes
- Template menggunakan dataset California (sklearn.fetch_california_housing) sebagai contoh. Jika Anda memilih dataset lain (Ames/Kaggle), sesuaikan train.py (fitur, encoding, pipeline).
- train.py membuat pipeline sklearn (preprocessor + regressor) dan menyimpan artefak model sebagai models/model.joblib.
- app.main akan coba memuat model dari models/model.joblib. Jika tidak ada, API masih akan berjalan menggunakan dummy predictor (berguna saat review awal sebelum training).

## Tests
Jalankan:
  pytest -q

## Checklist pengumpulan (sesuaikan dengan soal UAS)
- [ ] Notebook laporan (EDA, eksperimen, analisis)
- [ ] Model & artefak disimpan di /models
- [ ] FastAPI app di /app
- [ ] Dockerfile dan instruksi build/run (README)
- [ ] Web UI di /web
- [ ] Unit tests di /tests