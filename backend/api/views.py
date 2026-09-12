import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from . import ml_dl_demo as ml

@csrf_exempt
def predict_tf(request):
    if request.method != "POST" or "file" not in request.FILES:
        return JsonResponse({"error": "이미지 파일이 필요합니다."}, status=400)

    file = request.FILES["file"]
    contents = file.read()

    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = ml.tf_model.predict(img_array)
    decoded = decode_predictions(preds, top=10)[0]

    cat_score = 0.0
    dog_score = 0.0
    cat_keywords = ['cat', 'tabby', 'tiger_cat', 'persian', 'siamese', 'egyptian_cat', 'cougar', 'lynx', 'leopard']

    for _, name, score in decoded:
        name_lower = name.lower()
        score_val = float(score)
        if any(k in name_lower for k in cat_keywords):
            cat_score += score_val
        elif any(k in name_lower for k in ['dog', 'terrier', 'retriever', 'hound', 'shepherd', 'poodle', 'corgi', 'spaniel', 'boxer', 'bulldog', 'chihuahua', 'pug']):
            dog_score += score_val

    if cat_score + dog_score > 0:
        total = cat_score + dog_score
        cat_prob = (cat_score / total) * 100.0
        dog_prob = (dog_score / total) * 100.0
    else:
        top_name = decoded[0][1].lower()
        if any(k in top_name for k in cat_keywords):
            cat_prob, dog_prob = 85.0, 15.0
        else:
            cat_prob, dog_prob = 15.0, 85.0

    if cat_prob >= dog_prob:
        final_label = "고양이 🐱"
        final_confidence = round(cat_prob, 2)
    else:
        final_label = "개 🐶"
        final_confidence = round(dog_prob, 2)

    return JsonResponse({
        "label": final_label,
        "confidence": final_confidence
    })

@csrf_exempt
def predict_torch(request):
    if request.method != "POST" or "file" not in request.FILES:
        return JsonResponse({"error": "이미지 파일이 필요합니다."}, status=400)

    file = request.FILES["file"]
    contents = file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = ml.torch_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = ml.torch_model(input_tensor)
        probs = F.softmax(output[0], dim=0)

    top3_prob, top3_idx = torch.topk(probs, 3)

    results = []
    for i in range(3):
        results.append({
            "label": ml.torch_classes[top3_idx[i].item()],
            "confidence": round(top3_prob[i].item() * 100, 2)
        })

    return JsonResponse({"results": results})

@csrf_exempt
def predict_box_office_3d(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 지원합니다."}, status=405)

    data = json.loads(request.body.decode("utf-8"))
    budget = float(data.get("budget", 0))
    screens = float(data.get("screens", 0))

    new_data = np.array([[budget, screens]])
    pred_audience = float(ml.movie_reg_model.predict(new_data)[0])
    pred_audience = max(0.0, round(pred_audience, 1))

    x1_range = np.linspace(ml.X_movie[:, 0].min(), max(ml.X_movie[:, 0].max(), budget + 20), 10)
    x2_range = np.linspace(ml.X_movie[:, 1].min(), max(ml.X_movie[:, 1].max(), screens + 200), 10)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    y_grid = ml.movie_reg_model.coef_[0] * x1_grid + ml.movie_reg_model.coef_[1] * x2_grid + ml.movie_reg_model.intercept_

    fig = plt.figure(figsize=(7, 5.2))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(ml.X_movie[:, 0], ml.X_movie[:, 1], ml.y_movie, color='red', label='Train Data')
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

    return JsonResponse({
        "predicted_audience": pred_audience,
        "graph_image": f"data:image/png;base64,{img_base64}"
    })

def get_rating_status(request):
    ratings = ml.df_ratings["rating"].tolist()
    avg = float(np.mean(ratings)) if len(ratings) > 0 else 0.0
    img_base64 = ml.generate_rating_chart(ratings)
    return JsonResponse({
        "average": round(avg, 2),
        "count": len(ratings),
        "chart_image": f"data:image/png;base64,{img_base64}"
    })

@csrf_exempt
def add_rating(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 지원합니다."}, status=405)

    data = json.loads(request.body.decode("utf-8"))
    user_rating = float(data.get("rating", 5))

    new_row = pd.DataFrame({"rating": [user_rating]})
    ml.df_ratings = pd.concat([ml.df_ratings, new_row], ignore_index=True)
    ml.df_ratings.to_csv(ml.csv_file, index=False)

    X_new = np.array([[0]])
    ml.sgd_model.partial_fit(X_new, [user_rating])

    ratings = ml.df_ratings["rating"].tolist()
    avg = float(np.mean(ratings))
    img_base64 = ml.generate_rating_chart(ratings)

    return JsonResponse({
        "average": round(avg, 2),
        "count": len(ratings),
        "chart_image": f"data:image/png;base64,{img_base64}"
    })

@csrf_exempt
def filter_youtube_videos(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST 요청만 지원합니다."}, status=405)

    if not ml.youtube:
        return JsonResponse({"videos": [], "error": "API 키를 확인해주세요."})

    data = json.loads(request.body.decode("utf-8"))
    query = data.get("query", "")
    include_kw = data.get("include_kw", "")
    exclude_kw = data.get("exclude_kw", "")
    use_ml = bool(data.get("use_ml", False))

    request_api = ml.youtube.search().list(
        part="snippet",
        q=query,
        maxResults=20,
        type="video"
    )
    response = request_api.execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "title": item["snippet"]["title"],
            "description": item["snippet"]["description"],
            "videoId": item["id"]["videoId"]
        })

    include_list = [k.strip().lower() for k in include_kw.split(",") if k.strip()]
    exclude_list = [k.strip().lower() for k in exclude_kw.split(",") if k.strip()]

    auto_block = [
        "먹방", "mukbang", "장난", "몰카", "prank", "브이로그", "vlog",
        "funny video", "reaction", "게임 방송", "shorts", "셰프", "쉐프"
    ]

    ml_keywords = include_list if include_list else [query.strip().lower()]

    vectorizer, model = None, None
    if use_ml and ml_keywords:
        vectorizer, model = ml.train_youtube_model(ml_keywords)

    filtered = []
    for v in videos:
        text = (v["title"] + " " + v["description"]).lower()

        if use_ml:
            if any(k in text for k in auto_block):
                continue

        if exclude_list:
            if any(k in text for k in exclude_list):
                continue

        if include_list:
            score = sum(1 for k in include_list if k in text)
            if score == 0:
                continue

        if use_ml and vectorizer:
            if not ml.is_youtube_relevant(text, vectorizer, model):
                continue

        filtered.append(v)

    return JsonResponse({"videos": filtered})