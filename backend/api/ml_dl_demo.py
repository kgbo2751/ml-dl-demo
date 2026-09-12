import io
import os
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import models
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(BACKEND_DIR)

load_dotenv(os.path.join(CURRENT_DIR, "key.env"))
load_dotenv(os.path.join(BACKEND_DIR, "key.env"))
load_dotenv(os.path.join(ROOT_DIR, "key.env"))
load_dotenv()

BASE_DIR = BACKEND_DIR

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "").strip(' "\'')
youtube = None
if YOUTUBE_API_KEY:
    print(f"[SUCCESS] YouTube API Key 로드 완료: {YOUTUBE_API_KEY[:8]}********")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
else:
    print("[ERROR] YouTube API Key를 찾을 수 없습니다! key.env 파일 위치와 내용을 확인하세요.")

tf_model = MobileNetV2(weights="imagenet")

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

def train_youtube_model(target_keywords):
    texts = []
    labels = []

    for kw in target_keywords:
        texts.extend([
            f"{kw} 강의", f"{kw} 강좌", f"{kw} 튜토리얼", f"{kw} 기초",
            f"{kw} tutorial", f"{kw} lecture", f"{kw} guide", f"{kw} course"
        ])
        labels.extend([1, 1, 1, 1, 1, 1, 1, 1])

    negatives = [
        "먹방", "mukbang", "장난", "몰카", "prank", "funny", "comedy",
        "개그", "웃긴", "reaction", "리액션", "브이로그", "vlog", "일상",
        "게임 방송", "플레이", "gameplay", "쇼츠", "shorts", "이슈", "뉴스",
        "셰프", "쉐프"
    ]
    for n in negatives:
        texts.append(n)
        labels.append(0)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, labels)
    return vectorizer, model

def is_youtube_relevant(text, vectorizer, model):
    X_test = vectorizer.transform([text])
    prob = model.predict_proba(X_test)[0][1]
    return prob > 0.5