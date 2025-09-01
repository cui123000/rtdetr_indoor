import os
import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_ema import ExponentialMovingAverage
from peft import LoraConfig, get_peft_model
import json

# 自定义数据集类
class IndoorSceneDataset(Dataset):
    def __init__(self, annotations, processor, base_dir, augment=False):
        self.annotations = annotations
        self.processor = processor
        self.base_dir = base_dir  # 添加基础目录以解析相对路径
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ]) if augment else None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, label = self.annotations[idx]
        # 将相对路径转换为绝对路径
        image = Image.open(os.path.join(self.base_dir, img_path)).convert("RGB")
        if self.augment and self.transform:
            image = self.transform(image)
            image = transforms.ToPILImage()(image)
        text = [f"a {label.replace('_', ' ')}", f"a modern {label}", f"a {label} in a room"]
        return {"image": image, "text": text, "label": label}

# 自定义collate_fn
def custom_collate_fn(batch, processor, classes):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    texts = [text for item in batch for text in item["text"]]
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
    
    num_texts_per_image = len(batch[0]["text"])
    print("num_texts_per_image:", num_texts_per_image)
    input_ids = inputs["input_ids"].view(len(batch) * num_texts_per_image, -1)
    attention_mask = inputs["attention_mask"].view(len(batch) * num_texts_per_image, -1)
    label_indices = torch.tensor([classes.index(label) for label in labels], dtype=torch.long)
    
    return {
        "pixel_values": inputs["pixel_values"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_indices": label_indices
    }

# 零样本分类
def zero_shot_classification(image_path, classes, model, processor):
    image = Image.open(image_path).convert("RGB")
    text_inputs = [f"a {cls.replace('_', ' ')}" for cls in classes]
    text_inputs += [f"a modern {cls}" for cls in classes]
    inputs = processor(text=text_inputs, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image.view(1, len(text_inputs) // len(classes), len(classes)).mean(dim=1)
        probs = logits.softmax(dim=1).detach().numpy()[0]
    
    predicted_class = classes[np.argmax(probs)]
    return predicted_class, probs

# 微调模型
def fine_tune_clip(train_dataset, model, processor, classes, epochs=3, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=lambda batch: custom_collate_fn(batch, processor, classes))
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_indices = batch["label_indices"].to(device)
            
            batch_size, num_texts, seq_length = input_ids.shape
            pixel_values = pixel_values.repeat_interleave(num_texts, dim=0)
            input_ids = input_ids.view(-1, seq_length)
            attention_mask = attention_mask.view(-1, seq_length)
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            # 假设 logits_per_image 应为 [batch_size * num_texts, len(classes)]
            logits_per_image = outputs.logits_per_image.view(-1, len(classes))  # 自动调整第一维
            logits_per_image = logits_per_image.view(batch_size, num_texts, len(classes)).mean(dim=1)
            loss = torch.nn.functional.cross_entropy(logits_per_image, label_indices)
            
            print("pixel_values shape:", pixel_values.shape)
            print("input_ids shape:", input_ids.shape)
            print("attention_mask shape:", attention_mask.shape)
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            print("outputs.logits_per_image shape:", outputs.logits_per_image.shape)
            logits_per_image = outputs.logits_per_image.view(batch_size, num_texts, len(classes)).mean(dim=1)
            loss = torch.nn.functional.cross_entropy(logits_per_image, label_indices)
            progress_bar.set_postfix({"loss": loss.item()})
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
        
        scheduler.step()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")
    
    ema.copy_to(model.parameters())
    return model

# 测试模型
def test_fine_tuned_model(test_dataset, model, processor, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    true_labels = []
    
    test_loader = DataLoader(test_dataset, batch_size=1,
                            collate_fn=lambda batch: custom_collate_fn(batch, processor, classes))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_label_idx = batch["label_indices"][0].item()
            
            batch_size, num_texts, seq_length = input_ids.shape
            pixel_values = pixel_values.repeat_interleave(num_texts, dim=0)
            input_ids = input_ids.view(-1, seq_length)
            attention_mask = attention_mask.view(-1, seq_length)
            
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            logits_per_image = outputs.logits_per_image.view(1, num_texts, len(classes)).mean(dim=1)
            predicted_class_idx = torch.argmax(logits_per_image, dim=1).cpu().item()
            
            predictions.append(predicted_class_idx)
            true_labels.append(true_label_idx)
    
    accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)
    f1 = f1_score(true_labels, predictions, average="weighted")
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    
    errors = [(i, classes[p], classes[t]) for i, (p, t) in enumerate(zip(predictions, true_labels)) if p != t]
    print(f"Errors: {len(errors)} samples misclassified")
    for idx, pred, true in errors[:5]:
        print(f"Sample {idx}: Predicted {pred}, True {true}")

# 主程序
def main():
    # 加载CLIP模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  
    
    # 加载整合数据集
    data_dir = "/home/cui/robot_vlm_project/data"  # 基础目录，包含 coco_data, imagenet_data 等
    with open(os.path.join(data_dir, "home_dataset.json"), "r") as f:
        data = json.load(f)
    annotations = data["annotations"]
    classes = data["classes"]
    
    print("Number of classes:", len(classes))
    
    # 分割训练和测试集
    train_annotations, test_annotations = train_test_split(annotations, test_size=0.2, random_state=42)
    
    # 创建数据集
    train_dataset = IndoorSceneDataset(train_annotations, processor, data_dir, augment=True)
    test_dataset = IndoorSceneDataset(test_annotations, processor, data_dir, augment=False)
    
    # 零样本分类测试
    if test_annotations:
        test_image = os.path.join(data_dir, test_annotations[0][0])  # 转换为绝对路径
        predicted_class, probs = zero_shot_classification(test_image, classes, model, processor)
        print(f"Zero-shot Predicted: {predicted_class}")
        for cls, prob in zip(classes, probs):
            print(f"{cls}: {prob:.4f}")
    
    # 微调模型
    model = fine_tune_clip(train_dataset, model, processor, classes, epochs=3)
    
    # 测试微调模型
    test_fine_tuned_model(test_dataset, model, processor, classes)

if __name__ == "__main__":
    main()