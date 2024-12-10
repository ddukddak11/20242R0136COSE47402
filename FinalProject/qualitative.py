import torch
import clip
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 1. CLIP 및 YOLO 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 Nano 모델 사용

# 2. 이미지 로드
image_path = "ex3.jpg"  # 분석할 이미지 경로
original_image = Image.open(image_path).convert("RGB")

# 3. YOLO로 "컵" 객체 탐지
results = yolo_model.predict(image_path, conf=0.5)  # Confidence Threshold: 0.5
boxes = []  # "cup" 클래스만 저장

for result in results[0].boxes:
    if result.cls == 41:  # COCO 클래스 ID 41: "cup"
        boxes.append(result.xyxy.cpu().numpy().astype(int).tolist()[0])  # [x_min, y_min, x_max, y_max]

# 4. CLIP으로 텍스트 프롬프트와 매칭
prompt = "a blue cup"  # 판별할 텍스트 프롬프트
text_input = clip.tokenize([prompt]).to(device)
detected_boxes = []
similarities = []  # 각 박스의 유사도 점수 저장

for box in boxes:
    # YOLO로 얻은 Bounding Box 영역 자르기
    x_min, y_min, x_max, y_max = box
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    cropped_clip_image = preprocess(cropped_image).unsqueeze(0).to(device)

    # CLIP으로 유사도 계산
    with torch.no_grad():
        image_features = clip_model.encode_image(cropped_clip_image)
        text_features = clip_model.encode_text(text_input)
        logits_per_image = torch.matmul(image_features, text_features.T)  # 유사도 계산
        similarity_score = logits_per_image.softmax(dim=-1).cpu().numpy()[0][0]  # 소프트맥스 확률

    # 유사도가 높은 경우(Bounding Box 필터링)
    threshold = 0.5  # 임계값
    if similarity_score > threshold:
        detected_boxes.append(box)
        similarities.append(similarity_score)

# 5. 결과 시각화 (Bounding Box + 유사도 표시)
draw = ImageDraw.Draw(original_image)
for box, similarity in zip(detected_boxes, similarities):
    x_min, y_min, x_max, y_max = box
    # Bounding Box 그리기
    draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)
    # 유사도 텍스트 추가
    draw.text((x_min, y_min - 10), f"{similarity:.2f}", fill="blue")

# 6. 최종 결과 출력
plt.figure(figsize=(8, 8))
plt.imshow(original_image)
plt.axis("off")
plt.title("Detected: Blue Cups with Similarity Scores")
plt.show()