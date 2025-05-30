# Initialize
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator
from tensorflow import keras
from keras import layers
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
size = (128, 128)  # Original image size
image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True, size={"height": 128, "width": 128})  # Override default size of 224x224


from torchvision.transforms import Resize, CenterCrop, Compose, Normalize, ToTensor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

_transforms = Compose([
    Resize(size),  # Resize while maintaining aspect ratio
    CenterCrop(size),  # Take the center portion
    ToTensor(),
    normalize
])
def transforms(examples):
    examples["pixel_values"] = [_transforms(img) for img in examples["image"]]
    del examples["image"]
    # Removes the original "image" key since we now have the processed version
    return examples

aMRI_train = aMRI_train.with_transform(transforms)
aMRI_val = aMRI_val.with_transform(transforms)
aMRI_test = aMRI_test.with_transform(transforms)

# Data augmentation

#Extracts the target image dimensions from the image processor configuration.
train_data_augmentation = keras.Sequential(
    [
        layers.Rescaling(scale=1.0 / 128.0, offset=-1),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(0.1),
    ],
    name="train_data_augmentation",
)

val_data_augmentation = keras.Sequential(
    [
        layers.Rescaling(scale=1.0 / 128, offset=-1),
    ],
    name="val_data_augmentation",
) 

def convert_to_tf_tensor(image: Image):
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    np_image = np.array(image)
    tf_image = tf.convert_to_tensor(np_image)
    # Add channel dimension for grayscale (1 channel)
    tf_image = tf.expand_dims(tf_image, -1)
    # Add batch dimension
    return tf.expand_dims(tf_image, 0)

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    images = [
        train_data_augmentation(convert_to_tf_tensor(image)) for image in example_batch["image"]
    ]
        # Shape here: [1, height, width, 1]

    example_batch["pixel_values"] = [tf.squeeze(image, axis=0) for image in images]
    # tf.squeeze removes the batch dimension (index 0)
    # tf.transpose then reorders the remaining dimensions
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    images = [
        val_data_augmentation(convert_to_tf_tensor(image)) for image in example_batch["image"]
    ]
    example_batch["pixel_values"] = [tf.squeeze(image, axis=0) for image in images]
    return example_batch

def preprocess_test(example_batch):
    """Apply test_transforms across a batch."""
    images = [convert_to_tf_tensor(image) for image in example_batch["image"]]
    example_batch["pixel_values"] = [tf.squeeze(image, axis=0) for image in images]
    return example_batch

# Apply the preprocessing functions to the datasets
aMRI_train = aMRI_train.with_transform(preprocess_train)
aMRI_val = aMRI_val.with_transform(preprocess_val)
aMRI_test = aMRI_test.with_transform(preprocess_test)

data_collator = DefaultDataCollator(
    return_tensors="tf",
)

