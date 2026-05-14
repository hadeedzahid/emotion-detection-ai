import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    print("Loading dataset dair-ai/emotion...")
    # Load the emotion dataset from Hugging Face
    dataset = load_dataset("dair-ai/emotion", "split")
    
    # Take a very small subset for demonstration purposes (100 samples for train, 20 for eval)
    # This ensures the script runs quickly on a local machine without a GPU
    train_dataset = dataset["train"].select(range(100))
    eval_dataset = dataset["validation"].select(range(20))
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Evaluation set size: {len(eval_dataset)}")
    
    # Model and Tokenizer setup
    model_name = "distilbert-base-uncased"
    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # The dair-ai/emotion dataset has 6 labels: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./emotion_model_results",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1, # Just 1 epoch for demonstration
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="no", # Do not save intermediate checkpoints to save disk space
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating model...")
    results = trainer.evaluate()
    print(f"Evaluation results: {results}")
    
    # Save the final model
    save_path = "./emotion_model_final"
    print(f"Saving final model to {save_path}...")
    trainer.save_model(save_path)
    print("Fine-tuning script completed successfully!")

if __name__ == "__main__":
    main()
