# Sistem Cerdas: Rekomendasi Resep (Demo)

Proyek demo kecil: sistem cerdas sederhana yang merekomendasikan resep berdasarkan bahan yang dimasukkan pengguna.

Langkah cepat menjalankan (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Setelah berjalan, buka http://127.0.0.1:5000/ di browser.

Struktur singkat:
- `app.py`: backend Flask + endpoint `/api/recommend` dan halaman UI
- `model/train.py`: script untuk membangun TF-IDF dan menyimpan artefak
- `model/recipes.json`: dataset contoh resep
- `templates/index.html`: frontend HTML
- `static/main.js`, `static/styles.css`: frontend assets

Catatan:
- Jika `model/artifacts.pkl` belum ada, `app.py` akan memanggil `train_and_save()` secara otomatis untuk membuatnya dari `model/recipes.json`.
