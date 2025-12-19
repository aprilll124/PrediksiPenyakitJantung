import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
from io import StringIO
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="🩺",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = "Beranda"
if "df" not in st.session_state:
    st.session_state.df = None
if "winsorized" not in st.session_state:
    st.session_state.winsorized = False

# SIDEBAR – NAVIGASI
with st.sidebar:
    st.title("Navigasi")

    if st.button("Beranda", use_container_width=True):
        st.session_state.page = "Beranda"

    if st.button("Business Understanding", use_container_width=True):
        st.session_state.page = "Business"

    if st.button("Data Understanding", use_container_width=True):
        st.session_state.page = "Data"

    if st.button("Data Preparation", use_container_width=True):
        st.session_state.page = "Preparation"

    if st.button("Modeling", use_container_width=True):
        st.session_state.page = "Modeling"

    if st.button("Evaluasi Model", use_container_width=True):
        st.session_state.page = "Evaluasi"

    st.divider()

    if st.button("Prediksi Penyakit Jantung", use_container_width=True):
        st.session_state.page = "Prediksi"


# MODEL PATH & LOAD MODEL
MODEL_PATH = "model_rf_hf.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return None

model = load_model()

# KONTEN HALAMAN
page = st.session_state.page

# BERANDA
if page == "Beranda":
    st.markdown("<h1 style='text-align:center;color:crimson;'>❤️ Prediksi Penyakit Jantung</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:18px;'>Aplikasi Data Science untuk Deteksi Dini Penyakit Jantung menggunakan <b>Random Forest</b></p>", unsafe_allow_html=True)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Latar Belakang")
        st.markdown("<p style='text-align:justify;'>Penyakit jantung merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini sangat penting untuk membantu tenaga medis dalam mengambil keputusan yang cepat dan tepat.</p>", unsafe_allow_html=True)
    with col2:
        st.subheader("🎯 Tujuan Aplikasi")
        st.write("""
        - Memprediksi risiko penyakit jantung
        - Membantu analisis data klinis berbasis data
        - Menyediakan sistem prediksi berbasis machine learning
        """)
    st.divider()
    st.subheader("🔬 Metodologi Pengembangan")
    st.write("Aplikasi ini dikembangkan menggunakan pendekatan **CRISP-DM**.")
    st.subheader("✨ Fitur Utama")
    st.write("""
    - 🔍 Prediksi individu
    - 📂 Upload dataset untuk analisis & preprocessing
    - 📊 Visualisasi outlier & feature importance
    - ⬇️ Download data hasil preprocessing
    """)
    st.success("👉 Gunakan menu **Prediksi Penyakit Jantung** pada sidebar untuk mulai menggunakan aplikasi.")

# ===== BUSINESS UNDERSTANDING =====
elif page == "Business":
    st.markdown(
        """
        <h1 style='text-align:center; font-size:42px;'>
            📌 Business Understanding
        </h1>
        <p style='text-align:center; font-size:22px;'>
            Analisis kebutuhan bisnis dalam deteksi dini penyakit jantung
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.info(
    """
    **🎯 Fokus Bisnis**

    Meningkatkan kemampuan deteksi dini penyakit jantung menggunakan
    data pasien agar proses skrining menjadi lebih cepat, efisien,
    dan akurat.
    """
    )

    st.subheader("📖 Latar Belakang Masalah")
    st.markdown(
        "<p style='text-align:justify;'>"
        "Jantung merupakan organ vital yang berfungsi memompa darah untuk memenuhi "
        "kebutuhan oksigen dan nutrisi ke seluruh tubuh. Gangguan pada jantung dapat "
        "menyebabkan terganggunya peredaran darah dan berisiko menimbulkan penyakit "
        "jantung yang bersifat fatal. Penyakit jantung dipengaruhi oleh berbagai "
        "faktor risiko, baik yang tidak dapat diubah seperti usia dan jenis kelamin, "
        "maupun faktor yang dapat diubah seperti hipertensi, kolesterol, dan pola "
        "hidup. Kurangnya deteksi dini dan pemanfaatan data klinis secara optimal "
        "menyebabkan risiko keterlambatan diagnosis. Oleh karena itu, diperlukan "
        "sebuah sistem klasifikasi berbasis machine learning untuk membantu "
        "memprediksi risiko penyakit jantung secara dini."
        "</p>",
        unsafe_allow_html=True
    )

    st.subheader("📉 Permasalahan Bisnis")
    st.markdown(
        """
        <ul style='font-size:19px; line-height:1.8;'>
            <li>Skrining penyakit jantung masih memerlukan waktu yang lama</li>
            <li>Risiko keterlambatan dalam mendeteksi pasien berisiko</li>
            <li>Data klinis pasien belum dimanfaatkan secara optimal</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.subheader("💡 Solusi yang Diusulkan")
    st.markdown(
        """
        <ul style='font-size:19px; line-height:1.8;'>
            <li>Memanfaatkan data pasien sebagai dasar analisis</li>
            <li>Membangun sistem prediksi berbasis machine learning</li>
            <li>Mendukung tenaga medis dalam pengambilan keputusan awal</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.subheader("🎯 Tujuan")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style='font-size:19px;'>
            <b>Tujuan Bisnis</b>
            <ul>
                <li>Mempercepat proses skrining</li>
                <li>Meningkatkan akurasi deteksi dini</li>
                <li>Mendukung keputusan berbasis data</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style='font-size:19px;'>
            <b>Tujuan Sistem</b>
            <ul>
                <li>Mengklasifikasikan risiko penyakit jantung</li>
                <li>Menyajikan probabilitas risiko</li>
                <li>Menyediakan sistem yang mudah digunakan</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📌 Dampak yang Diharapkan")
    st.markdown(
        """
        <ul style='font-size:20px; line-height:1.9;'>
            <li>🧑‍⚕️ Membantu tenaga medis</li>
            <li>⏱️ Efisiensi waktu pemeriksaan</li>
            <li>📊 Optimalisasi data pasien</li>
            <li>❤️ Pencegahan keterlambatan penanganan</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.info(
    """
    ✅ Keberhasilan sistem diukur melalui tingkat akurasi dan kemampuan model dalam mengklasifikasikan pasien berisiko secara konsisten.
    """
    )

    st.info(
    """
    ✅ Tahap Business Understanding menjadi dasar untuk tahapan
        Data Understanding, Data Preparation, Modeling, dan Evaluasi.
    """
    )

# ===== DATA UNDERSTANDING =====
elif page == "Data":
    st.header("📊 Data Understanding")
    st.divider()

    uploaded_file = st.file_uploader("📂 Upload Dataset CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset berhasil diupload!")

    if st.session_state.df is None:
        st.warning("⚠️ Silakan upload dataset CSV untuk melanjutkan")
        st.stop()

    df = st.session_state.df

    st.subheader("📚 Deskripsi Dataset")
    st.write("""
    Dataset yang digunakan merupakan data klinis pasien yang berasal dari
    studi penyakit jantung (Cleveland dan Hungary).
    Dataset ini digunakan untuk membangun model machine learning
    yang bertujuan memprediksi keberadaan penyakit jantung berdasarkan
    indikator kesehatan pasien.
    """)

    st.write(f"""
    - Jumlah data: **{df.shape[0]} baris**
    - Jumlah atribut: **{df.shape[1]} kolom**
    - Jenis data: **Data klinis pasien**
    """)

    # Preview Dataset
    st.subheader("📄 Preview Dataset")
    st.dataframe(df.head(), use_container_width=True)

    # Informasi Struktur Data
    st.subheader("🧾 Informasi Struktur Dataset")
    buffer = StringIO()
    df.info(buf=buffer)
    st.code(buffer.getvalue())

    # Statistik Deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe(), use_container_width=True)

    # Identifikasi Variabel
    st.subheader("🏷️ Identifikasi Variabel")
    st.write("""
    - **Fitur (Features)**: seluruh atribut klinis pasien
    - **Target**: kolom `target` yang menunjukkan keberadaan penyakit jantung
      (0 = tidak ada penyakit jantung, 1 = ada penyakit jantung)
    """)

    # Distribusi Target
    if "target" in df.columns:
        st.subheader("⚖️ Distribusi Variabel Target")
        st.bar_chart(df["target"].value_counts())

    # Penjelasan Atribut 
    with st.expander("📌 Penjelasan Atribut Dataset"):
        st.markdown("""
        - **age**: usia pasien
        - **sex**: jenis kelamin (1 = laki-laki, 0 = perempuan)
        - **cp**: jenis nyeri dada
        - **trestbps**: tekanan darah saat istirahat
        - **chol**: kadar kolesterol
        - **thalach**: denyut jantung maksimum
        - **exang**: angina akibat aktivitas fisik
        - **oldpeak**: depresi segmen ST
        - **slope**: kemiringan segmen ST
        - **target**: status penyakit jantung
        """)

    st.success("✅ Tahap Data Understanding selesai. Lanjut ke Data Preparation!")


# ===== DATA PREPARATION =====
elif page == "Preparation":
    st.header("🧹 Data Preparation")
    
    if st.session_state.df is None:
        st.warning("⚠️ Belum ada dataset. Upload dulu di halaman **Data Understanding**.")
        st.stop()
    
    df = st.session_state.df.copy()
    
    st.subheader("🔍 Preview Data Saat Ini")
    st.dataframe(df.head())
    st.write(f"**Jumlah baris:** {df.shape[0]} | **Jumlah kolom:** {df.shape[1]}")
    st.divider()

    # 1. Missing Value
    st.subheader("1️⃣ Missing Value")
    missing_count = df.isnull().sum()
    st.dataframe(missing_count.to_frame("Jumlah Missing").style.background_gradient(cmap="Reds"))

    if missing_count.sum() > 0:
        st.warning(f"Terdapat **{missing_count.sum()}** nilai missing.")

        # Pilih metode penanganan missing value
        metode = st.selectbox(
            "Pilih metode penanganan missing value:",
            ["--Pilih--", "Mean (numerik)", "Median (numerik)", "Zero / Konstanta", "Hapus Baris"]
        )

        if metode != "--Pilih--":
            if st.button("🧹 Terapkan"):
                df_new = df.copy()
                
                if metode == "Mean (numerik)":
                    for col in df_new.select_dtypes(include=["int64", "float64"]):
                        df_new[col].fillna(df_new[col].mean(), inplace=True)
                elif metode == "Median (numerik)":
                    for col in df_new.select_dtypes(include=["int64", "float64"]):
                        df_new[col].fillna(df_new[col].median(), inplace=True)
                elif metode == "Zero / Konstanta":
                    for col in df_new.select_dtypes(include=["int64", "float64"]):
                        df_new[col].fillna(0, inplace=True)
                    for col in df_new.select_dtypes(include=["object"]):
                        df_new[col].fillna("Unknown", inplace=True)
                elif metode == "Hapus Baris":
                    before = df_new.shape[0]
                    df_new = df_new.dropna().reset_index(drop=True)
                    st.success(f"✅ Berhasil menghapus {before - df_new.shape[0]} baris!")

                st.session_state.df = df_new
                st.success("✅ Missing value berhasil ditangani!")
                st.rerun()
    else:
        st.success("✅ Tidak ada missing value.")


    # 2. Duplikasi
    st.subheader("2️⃣ Duplikasi Data")
    dup_count = df.duplicated().sum()
    st.write(f"**Jumlah baris duplikat:** {dup_count}")

    if dup_count > 0:
        duplicated_indices = df[df.duplicated(keep=False)].index.tolist()
        df_display = df.copy()
        df_display['__Status_Duplikat'] = 'Unik'
        df_display.loc[df.duplicated(keep='first'), '__Status_Duplikat'] = 'Duplikat (akan dihapus)'
        df_display.loc[df.duplicated(keep=False), '__Status_Duplikat'] = df_display.loc[df.duplicated(keep=False), '__Status_Duplikat'].replace('Unik', 'Asli (dipertahankan)')
        
        df_duplikat_detail = df_display.loc[duplicated_indices].copy().reset_index().rename(columns={'index': 'Nomor Baris (Indeks Asli)'})
        
        def highlight_duplikat(row):
            color = "#ffcccc" if row['__Status_Duplikat'] == 'Duplikat (akan dihapus)' else "#ccffcc"
            return [f'background-color: {color}' for _ in row]
        
        styled_table = df_duplikat_detail.style.apply(highlight_duplikat, axis=1)
        st.dataframe(styled_table, use_container_width=True)
        st.caption("🔴 Merah: duplikat | 🟢 Hijau: asli yang dipertahankan")

        if st.button("🧹 Hapus Data Duplikat", use_container_width=True):
            before = df.shape[0]
            st.session_state.df = df.drop_duplicates(keep='first').reset_index(drop=True)
            st.success(f"✅ Berhasil menghapus {before - st.session_state.df.shape[0]} baris duplikat!")
            st.rerun()
    else:
        st.success("✅ Tidak ada data duplikat.")

    # 3. Winsorizing
    st.subheader("3️⃣ Winsorizing - Penanganan Outlier")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if numeric_cols:
        st.write(f"Ditemukan **{len(numeric_cols)}** kolom numerik: {', '.join(numeric_cols)}")
        col1, col2 = st.columns(2)
        with col1:
            lower_perc = st.slider("Lower percentile (%)", 0, 20, 5)
        with col2:
            upper_perc = st.slider("Upper percentile (%)", 80, 100, 95)

        if st.button("🚀 Terapkan Winsorizing", use_container_width=True):
            for col in numeric_cols:
                low_val = df[col].quantile(lower_perc / 100)
                high_val = df[col].quantile(upper_perc / 100)
                st.session_state.df[col] = df[col].clip(lower=low_val, upper=high_val)
            st.session_state.winsorized = True
            st.success(f"✅ Winsorizing diterapkan ({lower_perc}% - {upper_perc}%)")
            st.rerun()

        # Visualisasi sebelum
        with st.expander("📊 Visualisasi Sebelum Winsorizing", expanded=not st.session_state.winsorized):
            st.dataframe(df[numeric_cols].describe().style.format("{:.2f}"))
            if numeric_cols:
                cols_per_row = 3
                rows = (len(numeric_cols) + cols_per_row - 1) // cols_per_row
                fig, axes = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 6*rows))
                axes = axes.flatten() if rows > 1 else [axes]
                for i, col in enumerate(numeric_cols):
                    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                                    boxprops=dict(facecolor="#FF9999"),
                                    medianprops=dict(color="red"))
                    axes[i].set_title(col, fontweight="bold", color="darkred")
                for i in range(len(numeric_cols), len(axes)):
                    fig.delaxes(axes[i])
                plt.tight_layout()
                st.pyplot(fig)

        # Visualisasi setelah
        if st.session_state.winsorized:
            df_after = st.session_state.df
            st.success("✅ Outlier telah ditangani dengan winsorizing.")
            st.dataframe(df_after[numeric_cols].describe().style.format("{:.2f}").background_gradient(cmap="Greens"))
            if numeric_cols:
                fig_after, axes_after = plt.subplots(rows, cols_per_row, figsize=(6*cols_per_row, 6*rows))
                axes_after = axes_after.flatten() if rows > 1 else [axes_after]
                for i, col in enumerate(numeric_cols):
                    axes_after[i].boxplot(df_after[col].dropna(), patch_artist=True,
                                          boxprops=dict(facecolor="#90EE90"),
                                          medianprops=dict(color="darkgreen"))
                    axes_after[i].set_title(col, fontweight="bold", color="darkgreen")
                for i in range(len(numeric_cols), len(axes_after)):
                    fig_after.delaxes(axes_after[i])
                plt.tight_layout()
                st.pyplot(fig_after)
    
    # 4️⃣ Pemilihan Kolom (Feature Selection Manual)
    # ==============================
    st.divider()
    st.subheader("4️⃣ Pemilihan Kolom (Feature Selection)")

    df_fs = st.session_state.df.copy()
    all_columns = df_fs.columns.tolist()

    # Default: semua kolom dipilih
    selected_columns = st.multiselect(
        "📌 Pilih kolom yang akan digunakan untuk modeling:",
        options=all_columns,
        default=all_columns
    )

    if not selected_columns:
        st.warning("⚠️ Minimal pilih 1 kolom.")
    else:
        if st.button("✅ Terapkan Pemilihan Kolom", use_container_width=True):
            st.session_state.df = df_fs[selected_columns]
            st.success(f"✅ {len(selected_columns)} kolom berhasil dipilih")
            st.rerun()
    
    # ===== ENCODING DATA KATEGORIKAL =====
    st.divider()
    st.subheader("5️⃣ Encoding Data Kategorikal")

    df_enc = st.session_state.df.copy()

    binary_cols = ['sex', 'fasting blood sugar', 'exercise angina']

    categorical_cols = ['chest pain type', 'resting ecg', 'ST slope']

    if categorical_cols:
        st.write(f"Kolom kategori (lebih dari 2 kelas) yang akan di-one-hot encode: {', '.join(categorical_cols)}")

        if st.button("🔄 Terapkan Encoding Otomatis", use_container_width=True):
            df_enc = pd.get_dummies(df_enc, columns=categorical_cols, drop_first=True)
            st.session_state.df = df_enc
            st.success("✅ Encoding berhasil diterapkan")
            st.rerun()
    else:
        st.success("✅ Tidak ada kolom kategorikal untuk di-encode, kolom biner tetap 0/1")


    # FEATURE SCALING (STANDARDIZATION)
    st.divider()
    st.subheader("6️⃣ Feature Scaling (Standardization)")

    df_scaled = st.session_state.df.copy()

    target_col = "target"
    if target_col in df_scaled.columns:
        feature_cols = df_scaled.drop(columns=[target_col]).select_dtypes(include=["int64", "float64"]).columns.tolist()
    else:
        feature_cols = df_scaled.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if feature_cols:
        if st.button("📐 Terapkan StandardScaler", use_container_width=True):
            scaler = StandardScaler()
            df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

            st.session_state.df = df_scaled
            st.session_state.scaler = scaler
            st.success("✅ Feature scaling berhasil diterapkan")
            st.rerun()
    else:
        st.warning("⚠️ Tidak ada fitur numerik untuk distandarisasi")

    # Standarisasi nama kolom sebelum download & modeling
    rename_dict = {
        'chest pain type': 'cp',
        'resting bp s': 'rbp',
        'cholesterol': 'chol',
        'fasting blood sugar': 'fbs',
        'resting ecg': 'resting_ecg',
        'max heart rate': 'max_hr',
        'exercise angina': 'exang',
        'ST slope': 'slope'
    }
    st.session_state.df = st.session_state.df.rename(columns=rename_dict)

    st.divider()
    st.subheader("📊 Preview Data Akhir")
    st.dataframe(st.session_state.df.head())
    st.write(f"**Jumlah baris akhir:** {st.session_state.df.shape[0]}")

    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Data Preprocessed (CSV)",
        data=csv,
        file_name="data_jantung_preprocessed.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.success("✅ Data siap untuk modeling!")

# ===== MODELING =====
elif page == "Modeling":
    st.header("🤖 Modeling")
    
    if st.session_state.df is None:
        st.warning("⚠️ Belum ada dataset yang dipreprocess.")
        st.stop()
    
    df = st.session_state.df.copy()
    
    if 'target' not in df.columns:
        st.error("❌ Kolom 'target' tidak ditemukan.")
        st.stop()
    
    # Daftar fitur default (boleh ada yang hilang)
    feature_cols = [
        'age', 'sex', 'cp', 'rbp', 'chol', 'fbs',
        'resting_ecg', 'max_hr', 'exang', 'oldpeak', 'slope'
    ]
    
    # 🔧 AUTO-SELEKSI FITUR YANG ADA
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        st.warning(f"⚠️ Kolom diabaikan (tidak ditemukan): {missing_features}")
    
    if len(available_features) == 0:
        st.error("❌ Tidak ada fitur yang bisa digunakan untuk modeling.")
        st.stop()
    
    feature_cols = available_features
    
    X = df[feature_cols]
    y = df['target']
    
    st.write(f"**Total data:** {len(df)} | **Fitur digunakan:** {len(feature_cols)}")
    st.write("**Daftar fitur:**", feature_cols)
    st.write("**Distribusi Target:**")
    st.write(y.value_counts())
    
    if st.button("🚀 Train Model Random Forest Sekarang", use_container_width=True):
        with st.spinner("Melatih model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            rf_model = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            rf_model.fit(X_train, y_train)
            
            st.session_state.model_features = feature_cols
            st.session_state.model = rf_model

            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Simpan hasil untuk evaluasi
            st.session_state.last_accuracy = accuracy
            st.session_state.last_y_test = y_test
            st.session_state.last_y_pred = y_pred
            
            # Simpan model
            joblib.dump(rf_model, MODEL_PATH)
            
            st.success(f"✅ Model berhasil dilatih & disimpan sebagai `{MODEL_PATH}`")
            st.success(f"🎯 **Akurasi:** {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            st.subheader("📊 Classification Report")
            report = classification_report(
                y_test, y_pred, output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"))
            
            st.subheader("📊 Feature Importance (Random Forest)")
            importances = rf_model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            
            st.dataframe(
                importance_df.style.format({"Importance": "{:.4f}"})
            )
            
            fig, ax = plt.subplots()
            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"]
            )
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance - Random Forest")
            ax.invert_yaxis()
            st.pyplot(fig)
            
            # Clear cache agar model baru langsung ter-load
            st.cache_resource.clear()


# ===== EVALUASI =====

elif page == "Evaluasi":
    st.header("📈 Evaluasi Model")
    
    if not os.path.exists(MODEL_PATH):
        st.warning("⚠️ Belum ada model. Lakukan training di halaman Modeling.")
        st.stop()
    
    if "last_accuracy" in st.session_state:
        st.success(f"🎯 **Akurasi Terakhir:** {st.session_state.last_accuracy:.4f} ({st.session_state.last_accuracy*100:.2f}%)")
        
        st.subheader("📊 Classification Report")
        report = classification_report(st.session_state.last_y_test, st.session_state.last_y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))
    else:
        st.info("📌 Lakukan training untuk melihat metrik evaluasi terbaru.")
    
    st.write("""
    **Penjelasan Metrik:**
    - Accuracy: Proporsi prediksi benar
    - Precision: Akurasi prediksi positif
    - Recall: Kemampuan mendeteksi kasus positif
    - F1-Score: Harmonik mean precision & recall
    """)

    st.subheader("🧩 Confusion Matrix")

    cm = confusion_matrix(
        st.session_state.last_y_test,
        st.session_state.last_y_pred
    )

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    # Tampilkan angka di dalam sel
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)


# ===== PREDIKSI =====
elif page == "Prediksi":
    st.markdown("<h1 style='text-align:center;color:crimson;'>❤️ Prediksi Penyakit Jantung</h1>", unsafe_allow_html=True)
    st.divider()

    if model is None or "model_features" not in st.session_state:
        st.error("❌ Model atau fitur belum tersedia. Lakukan training ulang.")
        st.stop()

    feature_cols = st.session_state.model_features

    tab1, tab2, tab3 = st.tabs(
        ["🔍 Prediksi Individu", "📂 Prediksi Massal", "📊 Feature Importance"]
    )

    with tab1:
        st.subheader("Masukkan Data Pasien")

        input_dict = {}
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'age' in feature_cols:
                input_dict['age'] = st.number_input("Usia", 1, 120, 45)

            if 'sex' in feature_cols:
                input_dict['sex'] = st.selectbox(
                    "Jenis Kelamin",
                    [0, 1],
                    format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki"
                )

            if 'cp' in feature_cols:
                input_dict['cp'] = st.selectbox(
                    "Tipe Nyeri Dada",
                    [1, 2, 3, 4],
                    format_func={
                        1: "Angina Tipikal",
                        2: "Angina Atipikal",
                        3: "Non-angina",
                        4: "Asimtomatik"
                    }.get
                )

            if 'rbp' in feature_cols:
                input_dict['rbp'] = st.number_input(
                    "Tekanan Darah Istirahat (mmHg)", 80, 200, 120
                )

        with col2:
            if 'chol' in feature_cols:
                input_dict['chol'] = st.number_input(
                    "Kolesterol", 100, 600, 200
                )

            if 'fbs' in feature_cols:
                input_dict['fbs'] = st.selectbox(
                    "Gula Darah Puasa >120 mg/dl",
                    [0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )

            if 'resting_ecg' in feature_cols:
                input_dict['resting_ecg'] = st.selectbox(
                    "Resting ECG",
                    [0, 1, 2],
                    format_func={
                        0: "Normal",
                        1: "Kelainan ST-T",
                        2: "Hipertrofi Ventrikel Kiri"
                    }.get
                )

            if 'max_hr' in feature_cols:
                input_dict['max_hr'] = st.number_input(
                    "Denyut Jantung Maksimum", 60, 220, 150
                )

        with col3:
            if 'exang' in feature_cols:
                input_dict['exang'] = st.selectbox(
                    "Angina saat Olahraga",
                    [0, 1],
                    format_func=lambda x: "Tidak" if x == 0 else "Ya"
                )

            if 'oldpeak' in feature_cols:
                input_dict['oldpeak'] = st.number_input(
                    "Oldpeak (Depresi ST)", 0.0, 6.0, 1.0, step=0.1
                )

            if 'slope' in feature_cols:
                input_dict['slope'] = st.selectbox(
                    "Kemiringan ST Segment",
                    [1, 2, 3],
                    format_func={
                        1: "Upsloping",
                        2: "Flat",
                        3: "Downsloping"
                    }.get
                )

        if st.button("🔮 Prediksi Sekarang", use_container_width=True):
            input_df = pd.DataFrame([input_dict])[feature_cols]

            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            if pred == 1:
                st.error("⚠️ **Risiko Penyakit Jantung TERDETEKSI**")
            else:
                st.success("✅ **Tidak Terdeteksi Risiko Penyakit Jantung**")

            st.markdown(f"**Probabilitas Risiko:** {prob*100:.2f}%")

    with tab3:
        st.subheader("📊 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(
            importance_df.style.format({"Importance": "{:.4f}"})
        )
