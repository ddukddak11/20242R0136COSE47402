import torch
import clip
from PIL import Image, ImageDraw
from ultralytics import YOLO
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
yolo_model = YOLO("yolov8n.pt")

image_path = "path/to/image.jpg"  # 분석할 이미지 경로
original_image = Image.open(image_path).convert("RGB")

results = yolo_model.predict(image_path, conf=0.5)  # Confidence Threshold: 0.5
boxes = [] 

for result in results[0].boxes:
    if result.cls == 41:  #프롬프트로 입력할 개체의 클래스로 값 지정
        boxes.append(result.xyxy.cpu().numpy().astype(int).tolist()[0])  # [x_min, y_min, x_max, y_max]

prompt = "write your text prompt"  # 판별할 텍스트 프롬프트
text_input = clip.tokenize([prompt]).to(device)
detected_boxes = []
similarities = [] 

for box in boxes:
    x_min, y_min, x_max, y_max = box
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    cropped_clip_image = preprocess(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(cropped_clip_image)
        text_features = clip_model.encode_text(text_input)
        logits_per_image = torch.matmul(image_features, text_features.T) 
        similarity_score = logits_per_image.softmax(dim=-1).cpu().numpy()[0][0] 

    threshold = 0.5  # 임계값
    if similarity_score > threshold:
        detected_boxes.append(box)
        similarities.append(similarity_score)

draw = ImageDraw.Draw(original_image)
for box, similarity in zip(detected_boxes, similarities):
    x_min, y_min, x_max, y_max = box

    draw.rectangle([x_min, y_min, x_max, y_max], outline="blue", width=3)

    draw.text((x_min, y_min - 10), f"{similarity:.2f}", fill="blue")

plt.figure(figsize=(8, 8))
plt.imshow(original_image)
plt.axis("off")
plt.title("Detected: Blue Cups with Similarity Scores")
plt.show()