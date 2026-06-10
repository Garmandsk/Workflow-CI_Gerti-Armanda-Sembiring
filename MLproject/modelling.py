import json
import warnings

import dagshub
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.utils import estimator_html_repr

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # dagshub.init(
    #     repo_owner="xmiagertiarmanda",
    #     repo_name="SMSML_Gerti-Armanda-Sembiring",
    #     mlflow=True,
    # )

    print("Memuat dataset...")
    file_path = "email_preprocessing.csv"
    df = pd.read_csv(file_path)

    # Teks MENTAH (sebelum TF-IDF)
    df["cleaned_message"] = df["cleaned_message"].fillna("")

    # Jika kolom 'message_length' belum ada di CSV, akan dibuat secara otomatis
    if "message_length" not in df.columns:
        df["message_length"] = df["cleaned_message"].apply(len)

    # X HARUS berupa DataFrame berisi semua fitur yang mau diproses
    X = df[["cleaned_message", "message_length"]]
    y = df["label"]

    # 1. Split Holdout (80/20) untuk evaluasi paling akhir
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Setup K-Fold Stratified (menjaga rasio spam & ham di tiap fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Ruang Pencarian Hyperparameter
    space = {
        "n_estimators": scope.int(hp.quniform("n_estimators", 50, 200, 10)),
        "max_depth": hp.choice(
            "max_depth", [None, scope.int(hp.quniform("max_depth_int", 10, 30, 5))]
        ),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    }

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                TfidfVectorizer(
                    max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.8
                ),
                "cleaned_message",
            ),
            ("num_features", "passthrough", ["message_length"]),
        ]
    )

    # 4. Fungsi Objektif
    def objective(params):
        with mlflow.start_run(nested=True):
            # Gabungkan TF-IDF dan RF ke dalam satu PIPELINE
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=params["n_estimators"],
                            max_depth=params["max_depth"],
                            min_samples_split=params["min_samples_split"],
                            random_state=42,
                            class_weight="balanced",
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

            # Eksekusi 5-Fold Cross Validation secara paralel (n_jobs=-1)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, cv=skf, scoring="f1", n_jobs=-1
            )
            avg_f1 = cv_scores.mean()

            # Log param dan metrik CV ke sub-run MLflow
            mlflow.log_params(params)
            mlflow.log_metric("cv_avg_f1", avg_f1)

            # Hyperopt butuh nilai loss yang makin kecil makin baik (jadi F1 di-minus)
            return {"loss": -avg_f1, "status": STATUS_OK}

    # 5. Jalankan Eksperimen Utama
    print("Memulai Optimasi Bayesian dengan 5-Fold CV...")
    with mlflow.start_run(run_name="RF_Hyperopt_CV_Pipeline"):
        # Menambahkan tags
        mlflow.set_tags(
            {
                "mlflow.note.content": "Eksperimen klasifikasi Email Spam menggunakan Random Forest dan Hyperopt.",
                "project_name": "Email Spam Detection",
                "model_type": "Random Forest Pipeline",
                "developer": "Gerti Armanda",
                "tuning_method": "Bayesian Optimization (Hyperopt)",
                "validation": "5-Fold Stratified CV",
            }
        )

        # Mengambil profil df (DataFrame awal) dan menandai targetnya
        dataset_info = mlflow.data.from_pandas(df, targets="label", name="email_raw")
        mlflow.log_input(dataset_info, context="training")

        trials = Trials()
        best_params_raw = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10,  # jumlah kombinasi yang mau dicoba
            trials=trials,
        )

        # Proses format parameter dari Hyperopt
        best_params = {
            "n_estimators": int(best_params_raw["n_estimators"]),
            "max_depth": None
            if best_params_raw["max_depth"] == 0
            else int(best_params_raw["max_depth_int"]),
            "min_samples_split": int(best_params_raw["min_samples_split"]),
        }

        print(f"Hyperparameter Terbaik Ditemukan: {best_params}")

        # 6. Latih Ulang Model Final dengan Seluruh Data Latih & Parameter Terbaik
        print("Melatih Pipeline Final...")
        final_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=best_params["n_estimators"],
                        max_depth=best_params["max_depth"],
                        min_samples_split=best_params["min_samples_split"],
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        final_pipeline.fit(X_train, y_train)

        # 7. Evaluasi pada Test Set (Data yang tidak tersentuh)
        y_pred = final_pipeline.predict(X_test)

        test_metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
        }

        # Simpan Log params dan metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics(test_metrics)

        # Simpan Log model
        mlflow.sklearn.log_model(
            sk_model=final_pipeline,
            name="best_rf_pipeline",
            registered_model_name="Email_Spam_Classifier_Model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
        )

        # Simpan log tambahan
        # Log Artifact confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Random Forest")

        # Simpan gambar secara lokal lalu kirim ke MLflow
        cm_file = "training_confusion_matrix.png"
        plt.savefig(cm_file)
        mlflow.log_artifact(cm_file)
        plt.close()  # Bersihkan memory plot

        # Log artifact metric
        metric_file = "metric_info.json"
        with open(metric_file, "w") as f:
            # Menyimpan test_metrics ke dalam format JSON
            json.dump(test_metrics, f, indent=4)
        mlflow.log_artifact(metric_file)

        # Log artifact estimator
        html_file = "estimator.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(estimator_html_repr(final_pipeline))
        mlflow.log_artifact(html_file)

        # Log artifact learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            estimator=final_pipeline,
            X=X_train,
            y=y_train,
            cv=skf,
            scoring="f1",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5),
        )

        # Hitung rata-rata dan standar deviasi dari 5-Fold CV
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # Mulai menggambar grafik
        plt.figure(figsize=(8, 6))

        # Garis Training (Warna Biru)
        plt.plot(
            train_sizes,
            train_mean,
            color="blue",
            marker="o",
            markersize=5,
            label="Training F1-Score",
        )
        plt.fill_between(
            train_sizes,
            train_mean + train_std,
            train_mean - train_std,
            alpha=0.15,
            color="blue",
        )

        # Garis Validation (Warna Hijau)
        plt.plot(
            train_sizes,
            val_mean,
            color="green",
            linestyle="--",
            marker="s",
            markersize=5,
            label="Validation F1-Score",
        )
        plt.fill_between(
            train_sizes,
            val_mean + val_std,
            val_mean - val_std,
            alpha=0.15,
            color="green",
        )

        # Kosmetik Grafik
        plt.title("Learning Curve - Overfitting Check")
        plt.xlabel("Jumlah Sampel Data Latih")
        plt.ylabel("F1-Score")
        plt.grid(True)
        plt.legend(loc="lower right")

        # Simpan dan Log ke MLflow
        lc_file = "learning_curve.png"
        plt.savefig(lc_file)
        mlflow.log_artifact(lc_file)
        plt.close()  # Bersihkan memori

        print(
            f"Selesai! Cek DagsHub kamu sekarang. F1-Score pada Test Set: {test_metrics['test_f1']:.4f}"
        )
