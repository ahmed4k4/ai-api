import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ========= إعداد البيانات =========
df = pd.read_csv("laptopData.csv").replace('?', np.nan)

# تحويل أعمدة رقمية
df["Weight"] = pd.to_numeric(df["Weight"].str.replace("kg",""), errors="coerce")
df["Ram"]    = pd.to_numeric(df["Ram"].str.replace("GB",""), errors="coerce")
df[["Screen_Width","Screen_Height"]] = df["ScreenResolution"].str.extract(r'(\d+)x(\d+)').astype(float)
df["Cpu_Speed"] = pd.to_numeric(df["Cpu"].str.extract(r'(\d+(\.\d+)?)GHz')[0], errors="coerce")
df["Memory_Size"] = pd.to_numeric(df["Memory"].str.extract(r'(\d+)')[0], errors="coerce")
for m in ["SSD","HDD","Flash"]:
    df[f"Memory_{m}"] = df["Memory"].str.contains(m, na=False).astype(int)

# LabelEncoder للأعمدة النصية
encoders, categorical_options = {}, {}
for c in ["Company","TypeName","Gpu","OpSys"]:
    enc = LabelEncoder()
    df[c] = enc.fit_transform(df[c].astype(str))
    encoders[c], categorical_options[c] = enc, list(enc.classes_)

# ملء القيم الناقصة بالوسيط
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median(numeric_only=True)).fillna(0)

# تقسيم الهدف
df["Price_Category"] = pd.qcut(df["Price"], 3, labels=["Low","Medium","High"])
X = df.drop(["laptop_ID","Price","ScreenResolution","Cpu","Memory","Price_Category"], axis=1)
y = df["Price_Category"]

# Train/Test
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

# تدريب الموديل
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train,y_train)

# --------- Evaluation ---------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()  # JSON-friendly
report = classification_report(y_test, y_pred, output_dict=True)
