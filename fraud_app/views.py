from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import joblib
import numpy as np
import pandas as pd
import os
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "fraud_app", "fraud_detection_model.pkl")
SCALER_PATH = os.path.join(settings.BASE_DIR, "fraud_app", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ---- Single Transaction ----
def home(request):
    if request.method == "POST" and "csv_file" not in request.FILES:
        try:
            V1 = float(request.POST.get("V1"))
            V2 = float(request.POST.get("V2"))
            V3 = float(request.POST.get("V3"))
            Amount = float(request.POST.get("Amount"))

            features = np.array([[V1, V2, V3, Amount]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            prob = model.predict_proba(features_scaled)[0][1]

            result = "⚠️ Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

            return render(request, "result.html", {"result": result, "prob": round(prob, 2)})

        except Exception as e:
            return render(request, "home.html", {"error": str(e)})

    return render(request, "home.html")


def upload_csv(request):
    if request.method == "POST":
        try:
            csv_file = request.FILES["csv_file"]
            df = pd.read_csv(csv_file)

            # ✅ Same features as training
            required_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

            missing = [col for col in required_features if col not in df.columns]
            if missing:
                return render(request, "home.html", {"error": f"Missing columns: {missing}"})

            X = df[required_features]
            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)[:, 1]

            df["Prediction"] = predictions
            df["Fraud_Probability"] = probs

            # --- Fraud summary ---
            fraud_count = int((predictions == 1).sum())
            total = len(predictions)
            summary = f"✅ File processed successfully. Found {fraud_count}/{total} fraudulent transactions."

            # --- Preview table (first 10 rows) ---
            preview_html = df.head(10).to_html(classes="table table-bordered", index=False)

            # Save results (optional)
            output_path = os.path.join(settings.BASE_DIR, "fraud_app", "predictions.csv")
            df.to_csv(output_path, index=False)

            return render(
                request,
                "result.html",
                {
                    "result": summary,
                    "prob": None,
                    "table": preview_html,
                },
            )

        except Exception as e:
            return render(request, "home.html", {"error": str(e)})

    return render(request, "home.html")
