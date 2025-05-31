from transformers import pipeline
from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset
import torch
from torchvision.transforms import CenterCrop, ToTensor, Normalize, Compose
from preprocess import image_processor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
aMRI_test = load_dataset("Falah/Alzheimer_MRI", split="test")


model = AutoModelForImageClassification.from_pretrained("Fuscosucof/fusco_alzheimerMRI_model")

size = (128, 128)
_transforms = Compose([
    CenterCrop(size),
    ToTensor(),
    normalize
])

count = 0
sample_leng = len(aMRI_test)
for n in range(sample_leng):
    image = aMRI_test[n]["image"]
    image_tensor = _transforms(image.convert("RGB")).unsqueeze(0)
    # Inference  
    with torch.no_grad():
        logits = model(pixel_values=image_tensor).logits
    # (Single Prediction)
    predicted_label = logits.argmax(-1).item()
    real_label = aMRI_test[n]["label"]
    #print("Predicted label:", model.config.id2label[predicted_label])
    #print(f"Real label: {model.config.id2label[real_label]}")
    if model.config.id2label[predicted_label] == model.config.id2label[real_label]:
        count += 1
print(count)
# 1232
# Print final accuracy
print(f"\nFinal Accuracy: {count/sample_leng:.2%}")    
# 96.25%