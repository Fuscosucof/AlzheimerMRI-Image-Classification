Alzheimer's MRI Image Classification Using ResNet-50
This project implements a deep learning model for classifying Alzheimer's disease stages using MRI images.

Disclaimer: So I'm actually a highschooler grade 12 who is learning ML, LLM, AI by trial and error
This might look like a bad code since I've zero experience literally.
I followed isntruction and modified some code from hugging face docs, and some help of github copilot to understand some process and know how to train this model.

Please tell me if what part of the process could change and why would it raise an issues or it could be improved!!!

Dataset
Source: Falah/Alzheimer_MRI # Alzheimer's MRI Image Classification Using ResNet-50

This project implements a deep learning model for classifying Alzheimer's disease stages using MRI images.

## Dataset

- **Source**: [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI)
- **Classes**:
  - Mild Demented (0)
  - Moderate Demented (1)
  - Non Demented (2)
  - Very Mild Demented (3)
- **Split**: 
  - Training: 90% of training data
  - Validation: 10% of training data
  - Test: Separate test set

## Model Architecture

- Base model: ResNet-50 (microsoft/resnet-50)
- Modified for Alzheimer's classification (4 classes)
- Input size: 128x128 pixels
- Image format: RGB (converted from grayscale)

## Training Configuration

```python
training_args = TrainingArguments(
    learning_rate=0.002,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    warmup_ratio=0.1
)
```

## Setup Instructions

1. **Environment Setup**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. **Required Dependencies**
```txt
transformers
torch
torchvision
datasets
evaluate
pillow
numpy
```

3. **Training the Model**
```bash
python train.py
```

4. **Testing the Model**
```bash
python test.py
```

## Model Performance

- Training accuracy: ~96%
- Validation accuracy: ~95%
- Test accuracy: ~96%

## Important Notes

1. **RGB Conversion Consideration**
   - MRI images are inherently grayscale
   - RGB conversion increases computational overhead
   - Consider modifying ResNet's first layer for grayscale input

2. **Model Limitations**
   - Potential overfitting due to data augmentation
   - ResNet-50 might be oversized for this task

3. **Future Improvements**
   - Implement true grayscale processing
   - Add more robust data augmentation
   - Experiment with smaller architectures

## Repository Structure

```
├── train.py           # Training script
├── test.py           # Testing script
├── preprocess.py     # Data preprocessing
└── requirements.txt  # Dependencies
```

## Authors

[Fuscosucof]

## Acknowledgments

- HuggingFace Transformers
- Microsoft ResNet-50 Team
- Falah for the Alzheimer's MRI dataset
Classes:
Mild Demented (0)
Moderate Demented (1)
Non Demented (2)
Very Mild Demented (3)
Split:
Training: 90% of training data
Validation: 10% of training data
Test: Separate test set
Model Architecture
Base model: ResNet-50 (microsoft/resnet-50)
Modified for Alzheimer's classification (4 classes)
Input size: 128x128 pixels
Image format: RGB (converted from grayscale)



error occurs because your grayscale MRI images (1 channel) don't match ResNet's expected RGB input (3 channels)

Original ResNet: Expects [3, 224, 224] input
Modified ResNet: Works with [1, 224, 224] input



