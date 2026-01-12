import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve,
    classification_report
)

st.set_page_config(
    page_title="Klasifikasi Tingkat Kemiskinan",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ModelLoader:
    @staticmethod
    @st.cache_resource
    def load_artifacts():
        model = joblib.load("xgboost_model.pkl")
        model.set_params(predictor="cpu_predictor")
        scaler = joblib.load("scaler.pkl")
        try:
            X_test = joblib.load("X_test.pkl")
            y_test = joblib.load("y_test.pkl")
        except:
            X_test, y_test = None, None
        try:
            id_keluarga = joblib.load("id_keluarga.pkl")
        except:
            id_keluarga = None
        return model, scaler, X_test, y_test, id_keluarga


class InputPreprocessor:
    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, df_input: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df_input)


class Evaluator:
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_pred_proba)

        return {
            "accuracy": accuracy,
            "report": report,
            "auc": auc,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        }


model, scaler, X_test, y_test, id_keluarga = ModelLoader.load_artifacts()
input_preprocessor = InputPreprocessor(scaler)
evaluator = Evaluator()

st.title("🧩 Sistem Klasifikasi Tingkat Kemiskinan")

st.sidebar.header("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Input Data Baru", "Evaluasi & Kesimpulan"])

pendidikan_opsi = [
    "Paket A","SDLB","SD","MI","SPM/PDF Ula","Paket B","SMP LB","SMP",
    "MTs","SPM/PDF Wustha","Paket C","SMLB","SMA","MA","SMK","MAK",
    "SPM/PDF Ulya","D1/D2","D3","D4","S1","Profesi","S2","S3",
    "Tidak punya ijazah SD"
]

kegiatan_opsi = ["Bekerja","Sekolah","Mengurus rumah tangga","Lainnya","Tidak ada kegiatan"]

pekerjaan_opsi = [
    "Berusaha sendiri",
    "Berusaha dibantu pekerja tidak tetap",
    "Berusaha dibantu pekerja tetap",
    "Buruh/Karyawan",
    "Pekerja bebas",
    "Pekerja keluarga/tidak dibayar"
]

if page == "Input Data Baru":

    st.subheader("📥 Input Data Rumah Tangga")

    with st.form("prediksi_form"):

        input_id = st.text_input("Nomor Urut Rumah Tangga (5 Digit Terakhir NO. KK)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Pengeluaran")
            pengeluaran_rt = st.number_input("Pengeluaran Rumah Tangga", min_value=0.0, value=0.0)
            pengeluaran_makanan = st.number_input("Pengeluaran Makanan", min_value=0.0, value=0.0)
            pengeluaran_nonmakanan = st.number_input("Pengeluaran Non Makanan", min_value=0.0, value=0.0)
            jumlah_anggota = st.number_input("Jumlah Anggota", min_value=1, value=1)

        with col2:
            st.markdown("### Aset & Bantuan")
            status_rumah = st.selectbox("Status Kepemilikan Rumah", ["Milik sendiri", "Lainnya"])
            motor = st.selectbox("Memiliki Sepeda Motor", ["Ya", "Tidak"])
            mobil = st.selectbox("Memiliki Mobil", ["Ya", "Tidak"])
            emas = st.selectbox("Memiliki Emas ≥10gr", ["Ya", "Tidak"])
            kks = st.selectbox("Penerima KKS", ["Ya", "Tidak"])
            pkh = st.selectbox("Penerima PKH", ["Ya", "Tidak"])
            bpnt = st.selectbox("Penerima BPNT", ["Ya", "Tidak"])

        with col3:
            st.markdown("### Kepala Rumah Tangga")
            umur_kepala = st.number_input("Umur Kepala", min_value=1, value=1)
            status_perkawinan = st.selectbox(
                "Status Perkawinan",
                ["Belum Kawin","Kawin","Cerai Hidup","Cerai Mati"]
            )
            pendidikan = st.selectbox("Pendidikan Tertinggi", pendidikan_opsi)
            kegiatan = st.selectbox("Status Kegiatan", kegiatan_opsi)
            pekerjaan = st.selectbox("Pekerjaan Utama", pekerjaan_opsi)

        submit = st.form_submit_button("🔍 Klasifikasi")

    if submit:
        errors = []
        if not input_id:
            errors.append("Nomor urut rumah tangga belum diisi.")
        if pengeluaran_rt == 0:
            errors.append("Pengeluaran rumah tangga belum diisi.")
        if pengeluaran_makanan == 0:
            errors.append("Pengeluaran makanan belum diisi.")
        if pengeluaran_nonmakanan == 0:
            errors.append("Pengeluaran non-makanan belum diisi.")
        if umur_kepala == 1:
            errors.append("Umur kepala rumah tangga belum diisi.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            rumah = 1 if status_rumah == "Milik sendiri" else 5
            motor_e = 1 if motor == "Ya" else 5
            mobil_e = 1 if mobil == "Ya" else 5
            emas_e = 1 if emas == "Ya" else 5
            kks_e = 1 if kks == "Ya" else 5
            pkh_e = 1 if pkh == "Ya" else 5
            bpnt_e = 1 if bpnt == "Ya" else 5

            pernikahan_map = {"Belum Kawin":1,"Kawin":2,"Cerai Hidup":3,"Cerai Mati":4}
            pendidikan_e = pendidikan_opsi.index(pendidikan)+1
            kegiatan_map = {v:i+1 for i,v in enumerate(kegiatan_opsi)}
            pekerjaan_map = {v:i+1 for i,v in enumerate(pekerjaan_opsi)}

            pengeluaran_perkapita = pengeluaran_rt / jumlah_anggota

            df_input = pd.DataFrame([{
                "Rata_pengeluaran_rumah_tangga": pengeluaran_rt,
                "Rata_pengeluaran_makanan": pengeluaran_makanan,
                "Rata_pengeluaran_bukan_makanan": pengeluaran_nonmakanan,
                "Jumlah_anggota_rumah_tangga": jumlah_anggota,
                "Status_kepemilikan_rumah": rumah,
                "Memiliki_sepeda_motor": motor_e,
                "Memiliki_mobil": mobil_e,
                "Memiliki_emas_perhiasan_10gram": emas_e,
                "Penerima_KKS": kks_e,
                "Penerima_PKH": pkh_e,
                "Penerima_BPNT": bpnt_e,
                "Umur_kepala_rumah_tangga": umur_kepala,
                "Status_perkawinan_kepala": pernikahan_map[status_perkawinan],
                "Pendidikan_tinggi_kepala": pendidikan_e,
                "Status_kegiatan_krt": kegiatan_map[kegiatan],
                "Status_pekerjaan_utama_krt": pekerjaan_map[pekerjaan],
            }])

            scaled = input_preprocessor.transform(df_input)
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]

            st.subheader("📊 Hasil Prediksi")

            if pred == 1:
                st.error("Rumah Tangga Tergolong **Miskin**")
            else:
                st.success("Rumah Tangga **Tidak Miskin**")

            st.write(f"**Probabilitas Miskin:** {prob:.2f}")

            # Cek kesesuaian data: tidak miskin tapi dapat bantuan
            menerima_bantuan = (kks == "Ya" or pkh == "Ya" or bpnt == "Ya")
            
            if pred == 1 and prob >= 0.70:
                st.info("**Rekomendasi:** Risiko tinggi kemiskinan. Perlu verifikasi lapangan dan layak menjadi prioritas penerima bantuan sosial.")
            elif pred == 1 and 0.50 <= prob < 0.70:
                st.warning("**Rekomendasi:** Risiko sedang kemiskinan. Perlu peninjauan tambahan sebelum penetapan status.")
            elif pred == 0 and prob < 0.30:
                
                if menerima_bantuan:
                    bantuan_list = []
                    if kks == "Ya": bantuan_list.append("KKS")
                    if pkh == "Ya": bantuan_list.append("PKH")
                    if bpnt == "Ya": bantuan_list.append("BPNT")
                    bantuan_str = ", ".join(bantuan_list)
                    
                    st.warning("**PERHATIAN - Data Tidak Sesuai**")
                    st.write(f"""
                    Rumah tangga diprediksi **TIDAK MISKIN** (probabilitas miskin hanya {prob:.2f}), 
                    namun tercatat sebagai penerima bantuan sosial: **{bantuan_str}**.
                    
                    **Rekomendasi Tindak Lanjut:**
                    - Lakukan verifikasi ulang data lapangan
                    - Periksa validitas kepesertaan program bantuan
                    - Evaluasi apakah rumah tangga ini masih layak menerima bantuan
                    - Pertimbangkan realokasi bantuan ke rumah tangga yang lebih membutuhkan
                    
                    Kasus ini bisa mengindikasikan:
                    - Salah sasaran program bantuan sosial
                    - Perubahan kondisi ekonomi rumah tangga yang belum terupdate
                    - Kemungkinan error dalam pencatatan data
                    """)
                else:
                    st.success("**Rekomendasi:** Rumah tangga kemungkinan besar tidak miskin dan tidak memerlukan bantuan sosial prioritas.")
            elif pred == 0 and menerima_bantuan:
               
                bantuan_list = []
                if kks == "Ya": bantuan_list.append("KKS")
                if pkh == "Ya": bantuan_list.append("PKH")
                if bpnt == "Ya": bantuan_list.append("BPNT")
                bantuan_str = ", ".join(bantuan_list)
                
                st.info("**Catatan Evaluasi**")
                st.write(f"""
                Rumah tangga diprediksi **TIDAK MISKIN**, namun menerima bantuan: **{bantuan_str}**.
                Probabilitas miskin: {prob:.2f}
                
                **Saran:** Lakukan monitoring berkala untuk memastikan bantuan tepat sasaran.
                """)

if page == "Evaluasi & Kesimpulan":

    if X_test is None or y_test is None:
        st.error("Data uji tidak ditemukan. Jalankan script training terlebih dahulu.")
    else:
        st.subheader("📊 Evaluasi Model (Data Uji)")

        hasil_eval = evaluator.evaluate(model, X_test, y_test)
        y_pred = hasil_eval["y_pred"]
        y_pred_proba = hasil_eval["y_pred_proba"]
        report = hasil_eval["report"]
        auc = hasil_eval["auc"]
        accuracy = hasil_eval["accuracy"]

        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        metrik_df = pd.DataFrame({
            "Label": ["Tidak Miskin", "Miskin"],
            "Accuracy": [accuracy, accuracy],
            "Precision": [report["0"]["precision"], report["1"]["precision"]],
            "Recall": [report["0"]["recall"], report["1"]["recall"]],
            "F1-score": [report["0"]["f1-score"], report["1"]["f1-score"]],
            "AUC-ROC": [auc, auc] 
        })

        st.subheader("Tabel Evaluasi Metrik")
        st.dataframe(
            metrik_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-score": "{:.4f}",
                "AUC-ROC": "{:.4f}"
            })
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap="Blues", cbar=True,
                annot_kws={"size": 14}, ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted", fontsize=12)
            ax_cm.set_ylabel("Actual", fontsize=12)
            ax_cm.set_xticklabels(["Tidak Miskin", "Miskin"], fontsize=10)
            ax_cm.set_yticklabels(["Tidak Miskin", "Miskin"], fontsize=10)
            st.pyplot(fig_cm)
            
            st.markdown("**Penjelasan:**")
            st.write(f"""
            Confusion Matrix menampilkan hasil prediksi model dibandingkan dengan label sebenarnya:
            - **True Negative (TN)**: {cm[0][0]} - Model benar memprediksi rumah tangga tidak miskin
            - **False Positive (FP)**: {cm[0][1]} - Model salah memprediksi miskin padahal tidak miskin
            - **False Negative (FN)**: {cm[1][0]} - Model salah memprediksi tidak miskin padahal miskin
            - **True Positive (TP)**: {cm[1][1]} - Model benar memprediksi rumah tangga miskin
            
            Nilai FN yang rendah sangat penting dalam konteks kemiskinan karena hal itu membantu menghindari 
            kasus rumah tangga yang sebenarnya miskin tetapi tidak terdeteksi.
            """)
            
        with col2:
            # Kurva ROC
            st.subheader("Kurva ROC")
            fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
            ax_roc.plot(fpr, tpr, linewidth=2, label=f'Model (AUC = {auc:.4f})')
            ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1, label='Random Classifier')
            ax_roc.set_xlabel("False Positive Rate (FPR)", fontsize=11)
            ax_roc.set_ylabel("True Positive Rate (TPR)", fontsize=11)
            ax_roc.set_title("ROC Curve", fontsize=12)
            ax_roc.legend(fontsize=10)
            ax_roc.grid(alpha=0.3)
            ax_roc.tick_params(labelsize=10)
            st.pyplot(fig_roc)
            
            st.markdown("**Penjelasan:**")
            st.write(f"""
            Kurva ROC menunjukkan trade-off antara True Positive Rate (sensitivitas) dan False Positive Rate:
            - **AUC (Area Under Curve)**: {auc:.4f} - Semakin mendekati 1.0, semakin baik performa model
            - Kurva yang mendekati pojok kiri atas menunjukkan model yang baik
            - Garis putus-putus diagonal merepresentasikan model random (AUC = 0.5)
            
            AUC {auc:.4f} menunjukkan bahwa model memiliki kemampuan diskriminasi yang {'sangat baik' if auc > 0.9 else 'baik' if auc > 0.8 else 'cukup baik'} 
            dalam membedakan rumah tangga miskin dan tidak miskin.
            """)

        with col3:
            # Visualisasi Metrik 
            st.subheader("Visualisasi Metrik Evaluasi")
            fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
            
            # Visualisasi metrik weighted average
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
            values = [
                accuracy,
                report["weighted avg"]["precision"],
                report["weighted avg"]["recall"],
                report["weighted avg"]["f1-score"],
                auc 
            ]
            
            bars = ax_bar.bar(metrics, values, color=['#4da3ff', '#5cb85c', '#f0ad4e', '#d9534f', '#5bc0de'])
            ax_bar.set_ylim([0, 1])
            ax_bar.set_ylabel("Score", fontsize=11)
            ax_bar.set_title("Performa Model (Weighted Average)", fontsize=12, pad=20)
            ax_bar.tick_params(labelsize=10)
            ax_bar.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=10)
            
            st.pyplot(fig_bar)
            
            st.markdown("**Penjelasan:**")
            st.write(f"""
            Grafik ini menampilkan berbagai metrik evaluasi model (weighted average):
            - **Accuracy ({accuracy:.4f})**: Proporsi prediksi yang benar dari keseluruhan data
            - **Precision ({report["weighted avg"]["precision"]:.4f})**: Dari yang diprediksi miskin, berapa yang benar-benar miskin
            - **Recall ({report["weighted avg"]["recall"]:.4f})**: Dari yang benar-benar miskin, berapa yang berhasil terdeteksi
            - **F1-Score ({report["weighted avg"]["f1-score"]:.4f})**: Harmonic mean dari Precision dan Recall
            - **AUC-ROC ({auc:.4f})**: Kemampuan model membedakan antara kelas miskin dan tidak miskin
            
            Semua metrik menunjukkan nilai yang {'sangat baik (>0.9)' if min(values) > 0.9 else 'baik (>0.8)' if min(values) > 0.8 else 'cukup baik'}, 
            mengindikasikan model dapat diandalkan untuk klasifikasi tingkat kemiskinan.
            """)

        st.subheader("Kesimpulan")
        st.write(f"""
        Berdasarkan hasil pengujian menggunakan **data uji (test set)**, 
        model XGBoost menunjukkan performa yang baik dalam mengidentifikasi
        rumah tangga miskin dan tidak miskin dengan:
        
        - **Accuracy**: {accuracy:.2%}
        - **AUC-ROC**: {auc:.4f}
        - **Precision**: {report["weighted avg"]["precision"]:.2%}
        - **Recall**: {report["weighted avg"]["recall"]:.2%}
        
        Nilai AUC-ROC yang tinggi ({auc:.4f}) menunjukkan kemampuan model membedakan kedua kelas dengan sangat baik.
        """)


