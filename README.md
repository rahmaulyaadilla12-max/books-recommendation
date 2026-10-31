# Book Recommendation Dashboard

Dashboard Streamlit untuk dataset Book-Crossing dengan:
- Statistika deskriptif
- Distribusi rating eksplisit dan implisit
- Buku populer, buku dengan rata-rata rating tinggi, penulis populer, dan pengguna aktif
- Item-Based Collaborative Filtering menggunakan cosine similarity
- Rekomendasi Top-N per pengguna dan unduhan CSV
- Pengaturan threshold preprocessing melalui sidebar

## Menjalankan secara lokal

1. Pastikan Python 3.10+ terpasang.
2. Install dependensi:

   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:

   ```bash
   streamlit run app.py
   ```

4. Upload tiga file dataset pada sidebar:
   - `BX-Books.csv`
   - `BX-Users.csv`
   - `BX-Book-Ratings.csv`

Dataset Book-Crossing menggunakan delimiter titik koma (`;`) dan encoding `latin-1`.

## Deploy ke Streamlit Community Cloud

1. Upload `app.py` dan `requirements.txt` ke repository GitHub.
2. Buka https://share.streamlit.io/
3. Pilih repository dan file utama `app.py`.
4. Deploy, lalu upload ketiga CSV melalui sidebar dashboard.

## Catatan metodologi

- Rating 0 diperlakukan sebagai interaksi implisit dan tidak digunakan sebagai rating eksplisit dalam model.
- Model rekomendasi menggunakan item-item nearest neighbors dengan cosine distance.
- Threshold minimum rating dan batas jumlah buku dapat diubah pada sidebar.
- Skor rekomendasi adalah skor berbobot kemiripan, bukan probabilitas.
- Evaluasi offline train-test belum diimplementasikan pada versi ini.
