from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import os
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression, SGDRegressor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "cat_dog_model.h5")

tf_model = None
tf_classes = ["고양이 🐱", "개 🐶"]

if os.path.exists(model_path):
    tf_model = load_model(model_path)

weights = models.MobileNet_V2_Weights.DEFAULT
torch_model = models.mobilenet_v2(weights=weights).eval()
torch_preprocess = weights.transforms()
torch_classes = weights.meta["categories"]

X_movie = np.array([
    [10, 300], [25, 550], [40, 800], [60, 950],
    [80, 1100], [100, 1300], [120, 1500], [150, 1800],
    [180, 2000], [220, 2300], [30, 450], [70, 900],
    [90, 1200], [130, 1600], [160, 1900], [200, 2100]
], dtype=np.float32)

y_movie = np.array([
    30, 85, 150, 260,
    380, 510, 620, 850,
    980, 1250, 95, 310,
    430, 680, 890, 1100
], dtype=np.float32)

movie_reg_model = LinearRegression()
movie_reg_model.fit(X_movie, y_movie)
plt.rcParams['axes.unicode_minus'] = False

# ================= CSV 연동 및 SGD 모델 초기화 =================
csv_file = os.path.join(BASE_DIR, "movie_a_ratings.csv")

if os.path.exists(csv_file):
    df_ratings = pd.read_csv(csv_file)
else:
    df_ratings = pd.DataFrame({"rating": [4.0, 4.5, 3.5]})
    df_ratings.to_csv(csv_file, index=False)

sgd_model = SGDRegressor(max_iter=1, tol=None, learning_rate='constant', eta0=0.01, warm_start=True)
if len(df_ratings) > 0:
    X_init = np.zeros((len(df_ratings), 1))
    sgd_model.partial_fit(X_init, df_ratings["rating"].values)

def generate_rating_chart(ratings):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    avg = float(np.mean(ratings)) if len(ratings) > 0 else 0.0
    ax.bar(['Movie A'], [avg], color='#38bdf8', width=0.4)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Rating (1-5)")
    ax.set_title(f"Real-time Average: {avg:.2f} / 5.0 (Count: {len(ratings)})")
    ax.text(0, avg + 0.15, f"{avg:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.post("/predict/tf")
async def predict_tf(file: UploadFile = File(...)):
    if tf_model is None:
        return {"label": "models 폴더에 cat_dog_model.h5가 없습니다!", "confidence": 0.0}

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = float(tf_model.predict(img_array)[0][0])
    
    if pred >= 0.5:
        label = tf_classes[1]
        confidence = pred * 100.0
    else:
        label = tf_classes[0]
        confidence = (1.0 - pred) * 100.0

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }

@app.post("/predict/torch")
async def predict_torch(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    input_tensor = torch_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = torch_model(input_tensor)
        probs = F.softmax(output[0], dim=0)

    top3_prob, top3_idx = torch.topk(probs, 3)
    
    results = []
    for i in range(3):
        results.append({
            "label": torch_classes[top3_idx[i].item()],
            "confidence": round(top3_prob[i].item() * 100, 2)
        })

    return {"results": results}

class BoxOfficeRequest(BaseModel):
    budget: float
    screens: float

@app.post("/predict/box-office-3d")
async def predict_box_office_3d(data: BoxOfficeRequest):
    new_data = np.array([[data.budget, data.screens]])
    pred_audience = float(movie_reg_model.predict(new_data)[0])
    pred_audience = max(0.0, round(pred_audience, 1))

    x1_range = np.linspace(X_movie[:, 0].min(), max(X_movie[:, 0].max(), data.budget + 20), 10)
    x2_range = np.linspace(X_movie[:, 1].min(), max(X_movie[:, 1].max(), data.screens + 200), 10)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    y_grid = movie_reg_model.coef_[0] * x1_grid + movie_reg_model.coef_[1] * x2_grid + movie_reg_model.intercept_

    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_movie[:, 0], X_movie[:, 1], y_movie, color='red', label='Train Data')
    ax.scatter(new_data[:, 0], new_data[:, 1], [pred_audience], color='green', s=130, label='Predicted Point')
    ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.35, cmap='Blues')

    ax.set_title('Box Office 3D Linear Regression')
    ax.set_xlabel('Budget (100M KRW)')
    ax.set_ylabel('Screens (Count)')
    ax.set_zlabel('Audience (10K People)')
    ax.legend(loc='upper left')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        "predicted_audience": pred_audience,
        "graph_image": f"data:image/png;base64,{img_base64}"
    }

class RatingInput(BaseModel):
    rating: float

@app.get("/rating/status")
async def get_rating_status():
    ratings = df_ratings["rating"].tolist()
    avg = float(np.mean(ratings)) if len(ratings) > 0 else 0.0
    img_base64 = generate_rating_chart(ratings)
    return {
        "average": round(avg, 2),
        "count": len(ratings),
        "chart_image": f"data:image/png;base64,{img_base64}"
    }

@app.post("/rating/add")
async def add_rating(data: RatingInput):
    global df_ratings
    user_rating = float(data.rating)

    new_row = pd.DataFrame({"rating": [user_rating]})
    df_ratings = pd.concat([df_ratings, new_row], ignore_index=True)
    df_ratings.to_csv(csv_file, index=False)

    X_new = np.array([[0]])
    sgd_model.partial_fit(X_new, [user_rating])

    ratings = df_ratings["rating"].tolist()
    avg = float(np.mean(ratings))
    img_base64 = generate_rating_chart(ratings)

    return {
        "average": round(avg, 2),
        "count": len(ratings),
        "chart_image": f"data:image/png;base64,{img_base64}"
    }