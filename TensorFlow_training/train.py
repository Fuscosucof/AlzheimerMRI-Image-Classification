import evaluate 
import numpy as np
from transformers import create_optimizer, TFAutoModelForImageClassification
from preprocess import data_collator, labels, label2id, id2label, checkpoint, image_processor, aMRI_train, aMRI_val, aMRI_test
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback


accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

PER_DEVICE_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
TOTAL_BATCH_SIZE = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS


num_epochs = 5
total_steps = len(aMRI_train) * num_epochs
warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
learning_rate = 0.002

# Create the main schedule (linear decay)
main_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=total_steps - warmup_steps,
    end_learning_rate=0.0,
    power=1.0  # Linear decay
)

# Wrap with warmup
lr_schedule = tf.keras.optimizers.schedules.LinearWarmup(
    after_warmup_lr_sched=main_schedule,
    warmup_steps=warmup_steps,
    warmup_learning_rate=0.0  # Start from 0
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate,  
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    
)



model = TFAutoModelForImageClassification.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# converting our train dataset to tf.data.Dataset
tf_train_dataset = aMRI_train.to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

# converting our test dataset to tf.data.Dataset
tf_eval_dataset = aMRI_eval.to_tf_dataset(
    columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
push_to_hub_callback = PushToHubCallback(
    output_dir="azheimer_img_classifier",
    tokenizer=image_processor,
    save_strategy="no",
)
callbacks = [metric_callback, push_to_hub_callback]

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)