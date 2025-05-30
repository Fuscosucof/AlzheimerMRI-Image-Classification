from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoImageProcessor, DefaultDataCollator
from tensorflow import keras
from keras import layers
import numpy as np
import tensorflow as tf
from PIL import Image
from preprocess import train_data_augmentation, _transforms, labels, label2id, id2label, checkpoint, image_processor, convert_to_tf_tensor
#login()

aMRI_train = load_dataset("Falah/Alzheimer_MRI", split="train[:90%]")
#my_list = [1, 2, 3, 4, 5]
#print(my_list[:4])
#print(my_list[4:])

#print("Original image shape:", aMRI_train[0]["image"].size)  # Will show (width, height)
# Original image shape: (128, 128)


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
aMRI_train = aMRI_train.with_transform(preprocess_train)
example = aMRI_train[0]
image_tensor = example["pixel_values"]
print("Image shape:", image_tensor.shape)
# Image shape: (128, 128, 1) finally f*uck!!!!!