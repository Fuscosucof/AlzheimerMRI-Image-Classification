import evaluate
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, create_optimizer
from preprocess import data_collator, labels, label2id, id2label, checkpoint, image_processor, aMRI_train, aMRI_val, aMRI_test 
accuracy = evaluate.load("accuracy")

PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
TOTAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
    

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
'''ResNet-50 model was trained on ImageNet (1000 classes) but you're trying to use it for Alzheimer's classification (4 classes).'''

training_args = TrainingArguments(
    output_dir="fusco_alzheimerMRI_model",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.002,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    num_train_epochs=20,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    seed=42,
    optim="adamw_torch",
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=aMRI_train,
    eval_dataset=aMRI_val,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()

