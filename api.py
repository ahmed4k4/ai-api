import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# =============== إعداد البيانات وتدريب الموديل =================
df = pd.read_csv("adult.csv").drop(columns=['fnlwgt', 'education'])

# نخزن الـ LabelEncoders لكل عمود
encoders = {}
categorical_options = {}

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].replace("?", np.nan)
        encoder = LabelEncoder()
        not_null = df[col].dropna().astype(str)
        df.loc[not_null.index, col] = encoder.fit_transform(not_null)
        encoders[col] = encoder
        categorical_options[col] = list(encoder.classes_)  # القيم الأصلية

# تحويل للأرقام + معالجة NaN
df = df.apply(pd.to_numeric, errors="coerce").astype(float)
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)

X = df.drop(columns=["income"])
y = df["income"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تطبيع
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تدريب Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# النتائج
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()
report = classification_report(y_test, y_pred, output_dict=True)

# =============== إعداد السيرفر =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/model_info")
async def model_info():
    return JSONResponse(content={
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        # remove "income" from options
        "categorical_options": {k: v for k, v in categorical_options.items() if k != "income"},
        "numeric_columns": [col for col in X.columns if col not in categorical_options]
    })


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    try:
        # تحويل القيم النصية لأرقام
        row = {}
        for col in X.columns:
            if col in categorical_options:
                # تحويل النص للرقم
                row[col] = encoders[col].transform([data[col]])[0]
            else:
                row[col] = float(data[col])

        df_input = pd.DataFrame([row])
        df_input = df_input[X.columns]
        df_input = scaler.transform(df_input)

        prediction = model.predict(df_input)[0]
        # رجع النص الأصلي للتوقع
        prediction_label = encoders["income"].inverse_transform([int(prediction)])[0]

        return JSONResponse(content={"prediction": prediction_label})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
