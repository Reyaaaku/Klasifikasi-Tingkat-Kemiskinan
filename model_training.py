import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# DATA LOADER

class DataLoader:
    def __init__(self, file_path="dataset_kemiskinan_susenas.csv"):
        self.file_path = file_path
        self.delimiter = ";"
        self.df = None
        self.garis_kemiskinan = 539283
        self.id_keluarga = None

    def load(self):
        print("STEP 1: Load data")
        self.df = pd.read_csv(self.file_path, sep=self.delimiter)
        self.id_keluarga = self.df.get("renumbering nurt", None)
        print("Jumlah baris & kolom:", self.df.shape)
        return self.df

    def compute_perkapita(self):
 
        self.df["Rata_pengeluaran_rumah_tangga"] = pd.to_numeric(
            self.df["Rata_pengeluaran_rumah_tangga"], errors="coerce"
        )
        self.df["Jumlah_anggota_rumah_tangga"] = pd.to_numeric(
            self.df["Jumlah_anggota_rumah_tangga"], errors="coerce"
        )

        before = len(self.df)
        self.df = self.df[self.df["Jumlah_anggota_rumah_tangga"] > 0]
        after = len(self.df)
        print(f"Baris invalid (anggota RT ≤ 0) dibuang: {before - after}")
        if before - after > 0:
            print("Distribusi setelah buang baris invalid:")
            print(self.df["Label"].value_counts())
            print()

        self.df["pengeluaran_perkapita"] = (
            self.df["Rata_pengeluaran_rumah_tangga"] /
            self.df["Jumlah_anggota_rumah_tangga"]
        )
        return self.df

    def create_label(self):
 
        self.df["Label"] = (self.df["pengeluaran_perkapita"] < self.garis_kemiskinan).astype(int)
        print("\nSTEP 1b: Membuat Label Kemiskinan")
        print("Distribusi label (0=Tidak Miskin, 1=Miskin):")
        print(self.df["Label"].value_counts())
        print(f"  - Tidak Miskin (0): {(self.df['Label'] == 0).sum()}")
        print(f"  - Miskin (1): {(self.df['Label'] == 1).sum()}")
        print()
        return self.df


# PREPROCESSOR

class Preprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()

    def clean(self, df):
        print("STEP 2: Preprocessing")
        
        print(f"Jumlah data sebelum cleaning: {len(df)}")
        print("Distribusi sebelum cleaning:")
        print(df["Label"].value_counts())
        print()

        df = df.drop(columns=["renumbering nurt", "Provinsi", "Kabupaten_Kota"], errors="ignore")

        before = len(df)
        
        # Hitung missing values per kolom
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print("Kolom dengan missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  - {col}: {count} missing")
            print()
        
        df_after_dropna = df.dropna()
        print(f"Baris hilang karena dropna(): {before - len(df_after_dropna)}")
        if before - len(df_after_dropna) > 0:
            print("Distribusi setelah dropna:")
            print(df_after_dropna["Label"].value_counts())
            print()
        
        df = df_after_dropna.drop_duplicates()
        after = len(df)
        
        print(f"Baris hilang karena drop_duplicates(): {len(df_after_dropna) - after}")
        print(f"Total baris dibuang saat cleaning: {before - after}")
        print("Distribusi setelah cleaning:")
        print(df["Label"].value_counts())
        print()
        
        return df

    def encode(self, df):
        print("Encoding data...")
        
        binary_cols = [
            "Memiliki_sepeda_motor", "Memiliki_mobil",
            "Memiliki_emas_perhiasan_10gram", "Penerima_KKS",
            "Penerima_PKH", "Penerima_BPNT"
        ]
        for col in binary_cols:
            df[col] = df[col].map({1: 1, 5: 0})

        df["Status_perkawinan_kepala"] = df["Status_perkawinan_kepala"].map({1:0,2:1,3:2,4:3})
        df["Status_kegiatan_krt"] = df["Status_kegiatan_krt"].map({1:0,2:1,3:2,4:3,5:4})
        df["Status_pekerjaan_utama_krt"] = df["Status_pekerjaan_utama_krt"].map(
            {1:0,2:1,3:2,4:3,5:4,6:5}
        )

        df["Status_kepemilikan_rumah"] = self.label_encoder.fit_transform(df["Status_kepemilikan_rumah"])
        df["Pendidikan_tinggi_kepala"] = self.label_encoder.fit_transform(df["Pendidikan_tinggi_kepala"])

        print("Encoding selesai.")
        print("Distribusi setelah encoding:")
        print(df["Label"].value_counts())
        print()
        
        return df

    def select_features(self, df):
        print("Memilih fitur yang digunakan:")

        fitur = [
            "Rata_pengeluaran_rumah_tangga",
            "Rata_pengeluaran_makanan",
            "Rata_pengeluaran_bukan_makanan",
            "Jumlah_anggota_rumah_tangga",
            "Status_kepemilikan_rumah",
            "Memiliki_sepeda_motor",
            "Memiliki_mobil",
            "Memiliki_emas_perhiasan_10gram",
            "Penerima_KKS",
            "Penerima_PKH",
            "Penerima_BPNT",
            "Umur_kepala_rumah_tangga",
            "Status_perkawinan_kepala",
            "Pendidikan_tinggi_kepala",
            "Status_kegiatan_krt",
            "Status_pekerjaan_utama_krt",
            "Label"
        ]

        print("Total fitur:", len(fitur) - 1)
        for f in fitur[:-1]:
            print("-", f)
        print()
        
        before = len(df)
        df = df[fitur].dropna()
        after = len(df)
        
        if before - after > 0:
            print(f"PERINGATAN: {before - after} baris hilang saat select_features (ada NaN setelah encoding)")
            print("Distribusi setelah select features:")
            print(df["Label"].value_counts())
            print()
        
        return df

# SMOTE & SCALING

class Transformer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.smote = SMOTE(random_state=42)

    def apply_smote(self, X_train, y_train):
        print("STEP 4: SMOTE")
        print("Distribusi SEBELUM SMOTE:")
        print(y_train.value_counts())
        print(f"  - Tidak Miskin (0): {(y_train == 0).sum()}")
        print(f"  - Miskin (1): {(y_train == 1).sum()}")

        X_sm, y_sm = self.smote.fit_resample(X_train, y_train)

        print("\nDistribusi SESUDAH SMOTE:")
        print(y_sm.value_counts())
        print(f"  - Tidak Miskin (0): {(y_sm == 0).sum()}")
        print(f"  - Miskin (1): {(y_sm == 1).sum()}")
        print(f"Total sebelum: {len(y_train)}")
        print(f"Total sesudah: {len(y_sm)}")
        print()

        return X_sm, y_sm

    def scale_train(self, X):
        print("STEP 5: MinMax Scaling")
        print("Scaler di-fit pada data training.\n")
        return self.scaler.fit_transform(X)

    def scale_test(self, X):
        print("Normalisasi data testing.\n")
        return self.scaler.transform(X)


# MODEL 

class PovertyModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            random_state=42
        )

    def fit(self, X, y):
        print("STEP 6: Training model XGBoost")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get(self):
        return self.model


# EVALUATOR

class Evaluator:
    def evaluate(self, y_true, y_pred, y_proba, name):
        print(f"\nEVALUASI {name}")
        print("Accuracy :", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred, zero_division=0))
        print("Recall   :", recall_score(y_true, y_pred, zero_division=0))
        print("F1-Score :", f1_score(y_true, y_pred, zero_division=0))
        print("AUC      :", roc_auc_score(y_true, y_proba))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print(
            classification_report(
                y_true, y_pred,
                target_names=["Tidak Miskin", "Miskin"]
            )
        )

# SAVER

class ModelSaver:
    @staticmethod
    def save(model, scaler, label_encoder, X_test, y_test, id_keluarga):
        print("STEP 7: Saving Model\n")

        joblib.dump(model, "xgboost_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")
        joblib.dump(X_test, "X_test.pkl")
        joblib.dump(y_test, "y_test.pkl")
        joblib.dump(id_keluarga, "id_keluarga.pkl")

        print("  - xgboost_model.pkl")
        print("  - scaler.pkl")
        print("  - label_encoder.pkl")
        print("  - X_test.pkl (DATA UJI)")
        print("  - y_test.pkl (LABEL UJI)")
        print("  - id_keluarga.pkl\n")


# MAIN PIPELINE

if __name__ == "__main__":

    # LOAD DATA
    loader = DataLoader()
    df_raw = loader.load()
    df_raw = loader.compute_perkapita()
    df_raw = loader.create_label()

    # PREPROCESS
    prep = Preprocessor()
    df = prep.clean(df_raw.copy())
    df = prep.encode(df)
    df = prep.select_features(df)

    X = df.drop("Label", axis=1)
    y = df["Label"]

    print("=" * 60)
    print("STEP 3: Split Data (80% Train : 20% Test)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Jumlah data train: {X_train.shape[0]}")
    print(f"Jumlah data test : {X_test.shape[0]}")
    print()
    print("Distribusi data TRAIN sebelum SMOTE:")
    print(y_train.value_counts())
    print(f"  - Tidak Miskin (0): {(y_train == 0).sum()}")
    print(f"  - Miskin (1): {(y_train == 1).sum()}")
    print()

    transformer = Transformer()
    X_train_sm, y_train_sm = transformer.apply_smote(X_train, y_train)

    df_smote = pd.DataFrame(X_train_sm, columns=X_train.columns)
    df_smote["Label"] = y_train_sm.values
    df_smote.to_csv("data_train_smote.csv", index=False)
    print("data_train_smote.csv disimpan\n")

    X_train_scaled = transformer.scale_train(X_train_sm)
    X_test_scaled = transformer.scale_test(X_test)

    # TRAINING
    model = PovertyModel()
    model.fit(X_train_scaled, y_train_sm)

    evaluator = Evaluator()
    evaluator.evaluate(y_train_sm, model.predict(X_train_scaled),
                       model.predict_proba(X_train_scaled), "DATA TRAIN (SMOTE)")
    evaluator.evaluate(y_test, model.predict(X_test_scaled),
                       model.predict_proba(X_test_scaled), "DATA TEST")

    # SAVE ARTIFACTS
    ModelSaver.save(
        model.get(),
        transformer.scaler,
        prep.label_encoder,
        X_test_scaled,
        y_test,
        loader.id_keluarga
    )