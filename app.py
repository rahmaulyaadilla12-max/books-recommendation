import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Book Recommendation Dashboard",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
    .main {background-color: #f7f8fa;}
    [data-testid="stMetric"] {
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e7eaf0;
    }
    .dashboard-subtitle {color: #64748b; margin-top: -12px;}
</style>
""", unsafe_allow_html=True)

st.title("📚 Book Recommendation Dashboard")
st.markdown(
    '<p class="dashboard-subtitle">Eksplorasi data Book-Crossing dan rekomendasi buku dengan Item-Based Collaborative Filtering.</p>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Pengaturan Data")
    st.write("Upload ketiga file CSV Book-Crossing. Format file asli menggunakan pemisah titik koma (`;`).")
    books_file = st.file_uploader("BX-Books.csv", type=["csv"], key="books")
    users_file = st.file_uploader("BX-Users.csv", type=["csv"], key="users")
    ratings_file = st.file_uploader("BX-Book-Ratings.csv", type=["csv"], key="ratings")
    st.divider()
    st.subheader("Filter model")
    min_user_ratings = st.slider("Minimum rating eksplisit per pengguna", 1, 20, 5)
    min_book_ratings = st.slider("Minimum rating eksplisit per buku", 1, 20, 5)
    max_items = st.slider("Maksimum buku untuk model", 500, 5000, 2000, step=500)
    st.caption("Semakin tinggi ambang filter, semakin kecil data model dan biasanya semakin cepat prosesnya.")

@st.cache_data(show_spinner=False)
def load_csv(file_bytes, dtype_isbn=False):
    import io
    kwargs = {"sep": ";", "encoding": "latin-1", "on_bad_lines": "skip"}
    if dtype_isbn:
        kwargs["dtype"] = {"ISBN": str}
    return pd.read_csv(io.BytesIO(file_bytes), **kwargs)

@st.cache_data(show_spinner=False)
def prepare_data(books_bytes, users_bytes, ratings_bytes):
    books = load_csv(books_bytes, True)
    users = load_csv(users_bytes, False)
    ratings = load_csv(ratings_bytes, True)

    # Normalize column names by stripping accidental spaces.
    books.columns = books.columns.str.strip()
    users.columns = users.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()

    required_ratings = {"User-ID", "ISBN", "Book-Rating"}
    required_books = {"ISBN", "Book-Title"}
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"Kolom ratings harus memuat: {sorted(required_ratings)}")
    if not required_books.issubset(books.columns):
        raise ValueError(f"Kolom books harus memuat: {sorted(required_books)}")

    ratings = ratings[["User-ID", "ISBN", "Book-Rating"]].copy()
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    ratings["ISBN"] = ratings["ISBN"].astype(str)
    ratings = ratings[ratings["Book-Rating"].between(0, 10)]
    ratings = ratings.drop_duplicates(["User-ID", "ISBN"], keep="last")

    books["ISBN"] = books["ISBN"].astype(str)
    if "Book-Title" not in books.columns:
        books["Book-Title"] = "Judul tidak tersedia"
    if "Book-Author" not in books.columns:
        books["Book-Author"] = "Penulis tidak tersedia"
    if "Year-Of-Publication" not in books.columns:
        books["Year-Of-Publication"] = np.nan
    if "Publisher" not in books.columns:
        books["Publisher"] = "Penerbit tidak tersedia"

    return books, users, ratings

def book_display_table(df):
    cols = [c for c in ["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "ISBN", "Jumlah Rating", "Rata-rata Rating"] if c in df.columns]
    return df[cols]

if not (books_file and users_file and ratings_file):
    st.info("👈 Upload BX-Books.csv, BX-Users.csv, dan BX-Book-Ratings.csv pada sidebar untuk mulai menggunakan dashboard.")
    st.markdown("### Dashboard yang tersedia")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📊 Statistika deskriptif**\n\nJumlah buku, pengguna, interaksi, distribusi rating, dan buku terpopuler.")
    c2.markdown("**🔎 Eksplorasi data**\n\nPola rating, pengguna aktif, penulis populer, serta ringkasan kualitas data.")
    c3.markdown("**🎯 Rekomendasi buku**\n\nPilih pengguna dan dapatkan rekomendasi buku berdasarkan kemiripan pola rating.")
    st.stop()

try:
    with st.spinner("Membaca dan menyiapkan dataset..."):
        books, users, ratings_all = prepare_data(
            books_file.getvalue(), users_file.getvalue(), ratings_file.getvalue()
        )
except Exception as e:
    st.error(f"Gagal membaca dataset: {e}")
    st.stop()

# Ratings of 0 are implicit feedback, not explicit low ratings.
ratings_explicit = ratings_all[ratings_all["Book-Rating"] > 0].copy()
if ratings_explicit.empty:
    st.error("Tidak ditemukan rating eksplisit (nilai 1–10). Periksa file BX-Book-Ratings.csv.")
    st.stop()

# Descriptive data uses all explicit ratings, while recommendation model is filtered below.
book_meta = books.drop_duplicates("ISBN").copy()
ratings_with_books = ratings_explicit.merge(
    book_meta[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]],
    on="ISBN", how="left"
)
ratings_with_books["Book-Title"] = ratings_with_books["Book-Title"].fillna("Judul tidak tersedia")
ratings_with_books["Book-Author"] = ratings_with_books["Book-Author"].fillna("Penulis tidak tersedia")

# Repeatedly filter users and books until both meet the thresholds.
model_ratings = ratings_explicit.copy()
for _ in range(5):
    user_counts = model_ratings.groupby("User-ID").size()
    eligible_users = user_counts[user_counts >= min_user_ratings].index
    model_ratings = model_ratings[model_ratings["User-ID"].isin(eligible_users)]
    book_counts = model_ratings.groupby("ISBN").size()
    eligible_books = book_counts[book_counts >= min_book_ratings].index
    model_ratings = model_ratings[model_ratings["ISBN"].isin(eligible_books)]

# Cap the item universe to most-rated books for manageable model size.
if not model_ratings.empty:
    popular_isbns = model_ratings["ISBN"].value_counts().head(max_items).index
    model_ratings = model_ratings[model_ratings["ISBN"].isin(popular_isbns)]

# Sidebar overview
with st.sidebar:
    st.divider()
    st.subheader("📌 Ringkasan")
    st.write(f"Rating eksplisit: **{len(ratings_explicit):,}**")
    st.write(f"Interaksi untuk model: **{len(model_ratings):,}**")
    st.write(f"Buku dalam model: **{model_ratings['ISBN'].nunique():,}**")
    st.write(f"Pengguna dalam model: **{model_ratings['User-ID'].nunique():,}**")

tab_overview, tab_books, tab_recommend, tab_method = st.tabs([
    "📊 Ringkasan", "📚 Eksplorasi Buku", "🎯 Rekomendasi", "ℹ️ Metode & Data"
])

with tab_overview:
    st.subheader("Statistika Deskriptif")
    total_books = books["ISBN"].nunique()
    total_users = users["User-ID"].nunique() if "User-ID" in users.columns else ratings_all["User-ID"].nunique()
    total_interactions = len(ratings_all)
    implicit_count = int((ratings_all["Book-Rating"] == 0).sum())
    explicit_count = len(ratings_explicit)
    mean_rating = ratings_explicit["Book-Rating"].mean()
    median_rating = ratings_explicit["Book-Rating"].median()
    n_rated_books = ratings_explicit["ISBN"].nunique()
    n_rating_users = ratings_explicit["User-ID"].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Buku", f"{total_books:,}")
    m2.metric("Total Pengguna", f"{total_users:,}")
    m3.metric("Total Interaksi", f"{total_interactions:,}")
    m4.metric("Rata-rata Rating Eksplisit", f"{mean_rating:.2f}/10")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Rating Eksplisit (1–10)", f"{explicit_count:,}")
    m6.metric("Rating Implisit (0)", f"{implicit_count:,}")
    m7.metric("Buku yang Dinilai", f"{n_rated_books:,}")
    m8.metric("Pengguna yang Memberi Rating", f"{n_rating_users:,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Distribusi rating eksplisit")
        rating_dist = ratings_explicit["Book-Rating"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(rating_dist.index.astype(int).astype(str), rating_dist.values)
        ax.set_xlabel("Nilai rating")
        ax.set_ylabel("Jumlah rating")
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Jenis interaksi")
        feedback = pd.Series({
            "Rating eksplisit (1–10)": explicit_count,
            "Rating 0 (implisit)": implicit_count
        })
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.pie(feedback.values, labels=feedback.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Ringkasan statistik rating")
    summary = ratings_explicit["Book-Rating"].describe().rename({
        "count": "Jumlah", "mean": "Rata-rata", "std": "Standar deviasi",
        "min": "Minimum", "25%": "Kuartil 1", "50%": "Median",
        "75%": "Kuartil 3", "max": "Maksimum"
    })
    st.dataframe(summary.to_frame("Nilai").round(2), use_container_width=True)

    st.markdown("#### 10 buku dengan jumlah rating terbanyak")
    top_rated = (
        ratings_with_books.groupby(["ISBN", "Book-Title", "Book-Author"], dropna=False)
        .agg(**{"Jumlah Rating": ("Book-Rating", "size"),
                "Rata-rata Rating": ("Book-Rating", "mean")})
        .sort_values("Jumlah Rating", ascending=False)
        .head(10).reset_index()
    )
    st.dataframe(book_display_table(top_rated).round(2), use_container_width=True, hide_index=True)

with tab_books:
    st.subheader("Eksplorasi Buku dan Pengguna")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Buku terpopuler berdasarkan jumlah rating")
        top_n = st.selectbox("Jumlah buku yang ditampilkan", [10, 15, 20], index=0, key="popular_n")
        popularity = (
            ratings_with_books.groupby(["ISBN", "Book-Title", "Book-Author"], dropna=False)
            .agg(**{"Jumlah Rating": ("Book-Rating", "size"),
                    "Rata-rata Rating": ("Book-Rating", "mean")})
            .sort_values("Jumlah Rating", ascending=False)
            .head(top_n).reset_index()
        )
        st.dataframe(book_display_table(popularity).round(2), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### Buku dengan rata-rata rating tertinggi")
        min_count_for_avg = st.slider("Minimum jumlah rating agar rata-rata ditampilkan", 5, 100, 20, step=5)
        avg_books = (
            ratings_with_books.groupby(["ISBN", "Book-Title", "Book-Author"], dropna=False)
            .agg(**{"Jumlah Rating": ("Book-Rating", "size"),
                    "Rata-rata Rating": ("Book-Rating", "mean")})
            .query("`Jumlah Rating` >= @min_count_for_avg")
            .sort_values(["Rata-rata Rating", "Jumlah Rating"], ascending=False)
            .head(top_n).reset_index()
        )
        if avg_books.empty:
            st.warning("Tidak ada buku yang memenuhi batas minimum rating ini.")
        else:
            st.dataframe(book_display_table(avg_books).round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Penulis dengan rating terbanyak")
    author_stats = (
        ratings_with_books.groupby("Book-Author")
        .agg(**{"Jumlah Rating": ("Book-Rating", "size"),
                "Jumlah Judul": ("ISBN", "nunique"),
                "Rata-rata Rating": ("Book-Rating", "mean")})
        .sort_values("Jumlah Rating", ascending=False).head(15).reset_index()
    )
    st.dataframe(author_stats.round(2), use_container_width=True, hide_index=True)

    st.markdown("#### Pengguna paling aktif")
    active_users = ratings_explicit.groupby("User-ID").agg(
        **{"Jumlah Rating Eksplisit": ("Book-Rating", "size"),
           "Rata-rata Rating": ("Book-Rating", "mean")}
    ).sort_values("Jumlah Rating Eksplisit", ascending=False).head(15).reset_index()
    if "Age" in users.columns:
        active_users = active_users.merge(users[["User-ID", "Age"]].drop_duplicates("User-ID"), on="User-ID", how="left")
    st.dataframe(active_users.round(2), use_container_width=True, hide_index=True)

with tab_recommend:
    st.subheader("Rekomendasi Buku untuk Pengguna")
    st.write("Pilih pengguna yang termasuk dalam data model. Sistem mencari buku yang mirip dengan buku-buku yang pernah diberi rating oleh pengguna tersebut.")

    if model_ratings.empty or model_ratings["User-ID"].nunique() < 2 or model_ratings["ISBN"].nunique() < 2:
        st.warning("Data model terlalu sedikit setelah filtering. Turunkan nilai minimum rating di sidebar.")
    else:
        # Build sparse item-user matrix and fit nearest neighbors.
        from scipy.sparse import csr_matrix
        user_codes, user_ids = pd.factorize(model_ratings["User-ID"])
        item_codes, item_isbns = pd.factorize(model_ratings["ISBN"])
        matrix = csr_matrix(
            (model_ratings["Book-Rating"].astype(float).values, (item_codes, user_codes)),
            shape=(len(item_isbns), len(user_ids))
        )
        nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(21, len(item_isbns)))
        nn_model.fit(matrix)

        model_user_counts = model_ratings.groupby("User-ID").size().sort_values(ascending=False)
        user_options = model_user_counts.index.tolist()
        chosen_user = st.selectbox(
            "Pilih User-ID",
            user_options,
            format_func=lambda x: f"User {x} — {int(model_user_counts.loc[x])} rating dalam model"
        )
        num_recs = st.slider("Jumlah rekomendasi", 5, 20, 10)
        exclude_seen = st.checkbox("Sembunyikan buku yang sudah pernah dinilai pengguna", value=True)

        if st.button("✨ Buat Rekomendasi", type="primary", use_container_width=True):
            user_history = model_ratings[model_ratings["User-ID"] == chosen_user].copy()
            isbn_to_idx = {isbn: i for i, isbn in enumerate(item_isbns)}
            idx_to_isbn = np.asarray(item_isbns)
            seen = set(user_history["ISBN"])
            scores = {}

            # For each item rated by this user, retrieve nearest books and aggregate similarity-weighted ratings.
            for _, row in user_history.iterrows():
                source_isbn = row["ISBN"]
                source_idx = isbn_to_idx.get(source_isbn)
                if source_idx is None:
                    continue
                distances, indices = nn_model.kneighbors(matrix[source_idx], return_distance=True)
                for dist, neighbor_idx in zip(distances[0], indices[0]):
                    neighbor_isbn = idx_to_isbn[neighbor_idx]
                    similarity = max(0.0, 1.0 - float(dist))
                    if similarity <= 0 or neighbor_isbn == source_isbn:
                        continue
                    if exclude_seen and neighbor_isbn in seen:
                        continue
                    # Give more influence to items the user rated highly.
                    user_rating = float(row["Book-Rating"])
                    contribution = similarity * user_rating
                    if neighbor_isbn not in scores:
                        scores[neighbor_isbn] = [0.0, 0.0, 0]
                    scores[neighbor_isbn][0] += contribution
                    scores[neighbor_isbn][1] += similarity
                    scores[neighbor_isbn][2] += 1

            if not scores:
                st.warning("Belum ada rekomendasi yang dapat dibuat. Coba turunkan threshold filtering atau pilih pengguna lain.")
            else:
                recs = pd.DataFrame([
                    {"ISBN": isbn,
                     "Skor Rekomendasi": val[0] / val[1] if val[1] else 0,
                     "Total Kemiripan": val[1],
                     "Jumlah Sumber Mirip": val[2]}
                    for isbn, val in scores.items()
                ])
                recs = recs.sort_values(["Skor Rekomendasi", "Total Kemiripan"], ascending=False).head(num_recs)
                recs = recs.merge(
                    book_meta[["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]].drop_duplicates("ISBN"),
                    on="ISBN", how="left"
                )
                global_book_stats = ratings_explicit.groupby("ISBN").agg(
                    **{"Jumlah Rating": ("Book-Rating", "size"),
                       "Rata-rata Rating": ("Book-Rating", "mean")}
                ).reset_index()
                recs = recs.merge(global_book_stats, on="ISBN", how="left")
                recs["Book-Title"] = recs["Book-Title"].fillna("Judul tidak tersedia")
                recs["Book-Author"] = recs["Book-Author"].fillna("Penulis tidak tersedia")
                st.success(f"Berikut rekomendasi untuk User {chosen_user}.")
                st.dataframe(
                    recs[["Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "ISBN",
                          "Skor Rekomendasi", "Jumlah Rating", "Rata-rata Rating", "Jumlah Sumber Mirip"]]
                    .round(3),
                    use_container_width=True, hide_index=True
                )
                st.caption("Skor rekomendasi adalah rata-rata rating pengguna yang dibobot dengan kemiripan item; skor ini bukan probabilitas.")
                csv = recs.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ Unduh hasil rekomendasi (CSV)", csv, f"rekomendasi_user_{chosen_user}.csv", "text/csv")

        with st.expander("Lihat riwayat rating pengguna terpilih"):
            hist = ratings_with_books[ratings_with_books["User-ID"] == chosen_user].sort_values("Book-Rating", ascending=False)
            st.dataframe(
                hist[["Book-Title", "Book-Author", "Book-Rating", "ISBN"]].head(50),
                use_container_width=True, hide_index=True
            )

with tab_method:
    st.subheader("Metode, Asumsi, dan Catatan")
    st.markdown("""
    **Metode rekomendasi:** Item-Based Collaborative Filtering dengan Cosine Similarity.

    - Setiap buku direpresentasikan sebagai vektor rating dari para pengguna.
    - Cosine Similarity mengukur kemiripan pola rating dua buku.
    - Buku yang mirip dengan buku yang diberi rating tinggi oleh pengguna dapat memperoleh skor rekomendasi lebih tinggi.
    - Buku yang sudah pernah dinilai dapat dikecualikan dari hasil rekomendasi.

    **Preprocessing yang digunakan**
    - File dibaca dengan delimiter titik koma (`;`) dan encoding `latin-1`.
    - Rating 0 diperlakukan sebagai interaksi implisit dan tidak digunakan sebagai rating eksplisit pada model.
    - Pasangan User-ID–ISBN duplikat disederhanakan dengan menyimpan baris terakhir.
    - Pengguna dan buku disaring berdasarkan jumlah rating minimum.
    - Model dibatasi pada sejumlah buku paling sering dinilai untuk menjaga penggunaan memori.

    **Hal yang perlu diperhatikan**
    - Popularitas bukan berarti kualitas; buku dengan rating terbanyak belum tentu memiliki rata-rata tertinggi.
    - Dataset Book-Crossing bersifat sparse sehingga sebagian pengguna mungkin tidak mendapatkan rekomendasi.
    - Skor rekomendasi bukan probabilitas dan bukan prediksi rating yang sudah dikalibrasi.
    - Dashboard ini belum menghitung metrik evaluasi offline seperti Precision@K, Recall@K, RMSE, atau MAE. Evaluasi tersebut sebaiknya ditambahkan dengan pembagian data train-test yang tepat.
    """)
    st.markdown("#### Ukuran data setelah filtering")
    if model_ratings.empty:
        st.warning("Tidak ada data yang tersisa. Kurangi threshold pada sidebar.")
    else:
        density = len(model_ratings) / (
            model_ratings["User-ID"].nunique() * model_ratings["ISBN"].nunique()
        )
        st.write(f"- Interaksi model: **{len(model_ratings):,}**")
        st.write(f"- Pengguna model: **{model_ratings['User-ID'].nunique():,}**")
        st.write(f"- Buku model: **{model_ratings['ISBN'].nunique():,}**")
        st.write(f"- Kepadatan matriks (density): **{density:.4%}**")
        st.write(f"- Sparsity matriks: **{1-density:.4%}**")
