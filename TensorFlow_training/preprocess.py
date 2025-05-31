
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

size = (image_processor.size["height"], image_processor.size["width"])

# Data augmentation

#Extracts the target image dimensions from the image processor configuration.
train_data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(scale=1.0 / 128.0, offset=-1),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(0.1),
    ],
    name="train_data_augmentation",
)

val_data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(scale=1.0 / 128, offset=-1),
    ],
    name="val_data_augmentation",
) 

def convert_to_tf_tensor(image: Image):
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    return tf.expand_dims(tf_image, 0) # Adds a batch dimension

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
        # Shape here: [1, height, width, 1]

    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image) for image in images)]
    # tf.squeeze removes the batch dimension (index 0)
    # tf.transpose then reorders the remaining dimensions
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image.convert("RGB"))) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image) for image in images)]
    return example_batch

def preprocess_test(example_batch):
    """Apply test_transforms across a batch."""
    images = [convert_to_tf_tensor(image.convert("RGB")) for image in example_batch["image"]]
    example_batch["pixel_values"] = [tf.squeeze(image) for image in images]
    return example_batch

# Apply the preprocessing functions to the datasets
aMRI_train = aMRI_train.with_transform(preprocess_train)
aMRI_val = aMRI_val.with_transform(preprocess_val)
aMRI_test = aMRI_test.with_transform(preprocess_test)

data_collator = DefaultDataCollator(
    return_tensors="tf",
)
