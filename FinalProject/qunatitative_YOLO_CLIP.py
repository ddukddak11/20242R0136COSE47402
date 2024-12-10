import torch
import clip
from ultralytics import YOLO
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def iou_calculate(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov8n.pt")  
clip_model, preprocess = clip.load("ViT-B/32", device=device)

coco_annotation_path = "/content/coco/annotations/instances_val2017.json"
image_dir = "/content/coco/images/val2017/"
coco = COCO(coco_annotation_path)

categories = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())} 

text_prompts = [f"a {categories[cat_id]}" for cat_id in categories]
text_inputs = clip.tokenize(text_prompts).to(device)

true_labels, predicted_labels = [], []

image_ids = coco.getImgIds()
for img_id in image_ids[:10]:
    img_info = coco.loadImgs(img_id)[0]
    img_path = f"{image_dir}/{img_info['file_name']}"
    original_image = Image.open(img_path).convert("RGB")

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_boxes = [(ann['bbox'], categories[ann['category_id']]) for ann in anns]  

    results = yolo_model.predict(img_path, conf=0.5)
    boxes, labels = [], []
    for result in results[0].boxes:
        cls_id = int(result.cls)
        if cls_id in categories:
            boxes.append(result.xyxy.cpu().numpy().astype(int).tolist()[0]) 
            labels.append(categories[cls_id])  

    for box, yolo_label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
        cropped_clip_image = preprocess(cropped_image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(cropped_clip_image)
            text_features = clip_model.encode_text(text_inputs)
            logits_per_image = torch.matmul(image_features, text_features.T)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        detected_class = np.argmax(probs)
        predicted_label = text_prompts[detected_class].split(" ")[1]  
        predicted_labels.append(predicted_label)

        gt_class = next((gt[1] for gt in gt_boxes if iou_calculate(box, gt[0]) > 0.001), "None")
        true_labels.append(gt_class)

precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(true_labels, predicted_labels)

print(f"Pipeline Precision: {precision:.2f}")
print(f"Pipeline Recall: {recall:.2f}")
print(f"Pipeline F1-Score: {f1:.2f}")
print(f"Pipeline Accuracy: {accuracy:.2f}")
