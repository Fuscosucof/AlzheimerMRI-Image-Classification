
# Initialize
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator
import numpy as np
import tensorflow as tf
from PIL import Image
#login()

aMRI_train = load_dataset("Falah/Alzheimer_MRI", split="train[:90%]")
aMRI_val = load_dataset("Falah/Alzheimer_MRI", split="train[90%:]")
aMRI_test = load_dataset("Falah/Alzheimer_MRI", split="test")

#print(aMRI_train[0])

labels = aMRI_train.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

checkpoint = "microsoft/resnet-50"

image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)  


from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, ToTensor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"]))

_transforms = Compose([
    CenterCrop(size),  # Take the center portion
    ToTensor(),  # Convert to tensor
    normalize
])
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    # Removes the original "image" key since we now have the processed version
    return examples

aMRI_train = aMRI_train.with_transform(transforms)
aMRI_val = aMRI_val.with_transform(transforms)
aMRI_test = aMRI_test.with_transform(transforms)


data_collator = DefaultDataCollator(
)

