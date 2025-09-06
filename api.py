from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd

# استيراد الموديل وكل المتغيرات من الملف الثاني
from ml_model import model, scaler, X, accuracy, conf_matrix, report, encoders, categorical_options

# ========= API =========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/model_info")
async def model_info():
    clean_options = {
        col: [val for val in options if str(val).lower() != "nan"]
        for col, options in categorical_options.items()
    }
    return JSONResponse(content={
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "categorical_options": clean_options,
        "numeric_columns": [col for col in X.columns if col not in categorical_options]
    })

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    try:
        # تجهيز الصف الجديد
        row = {}
        for col in X.columns:
            if col in categorical_options:
                row[col] = encoders[col].transform([data[col]])[0]
            else:
                row[col] = float(data[col])

        df_input = pd.DataFrame([row])[X.columns]
        df_input = scaler.transform(df_input)

        prediction = model.predict(df_input)[0]
        return JSONResponse(content={"prediction": str(prediction)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

if __name__=="__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
