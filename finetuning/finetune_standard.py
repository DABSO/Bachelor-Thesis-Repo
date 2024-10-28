
# fine_tune_decoder transformer model on a dataset

import os
import argparse
import logging
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from src.ConfigLoader import ConfigLoader
import pandas as pd

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ChatDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.labels = []

        for _, row in dataframe.iterrows():
            input_message = {
                "role": "user",
                "content": row["Input"]
            }
            output_message = {
                "role": "assistant",
                "content": row["Output"]
            }
            messages = [input_message, output_message]
            # Apply chat template using the tokenizer's custom method
            # Assuming apply_chat_template is a method of the tokenizer
            # If it's a standalone function, adjust accordingly
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                truncation=True,
                max_length=self.max_length,
                truncation_side="left",
                add_generation_prompt=True
            )
            self.inputs.append(tokenized['input_ids'])
            self.labels.append(tokenized['labels'])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Decoder Transformer model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to fine-tune."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset for fine-tuning."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=8192,
        help="Maximum sequence length."
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load model configuration
    logger.info("Loading model configuration...")
    model_config = ConfigLoader.get_model_config(args.model_name)
    model_repo = model_config["model_repo"]
    model_dir = os.getenv("MODEL_DIR")
    if model_dir is None:
        logger.error("MODEL_DIR environment variable is not set.")
        exit(1)
    model_path = os.path.join(model_dir, model_repo)

    # Load tokenizer and model
    logger.info(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Ensure tokenizer has the apply_chat_template method
    if not hasattr(tokenizer, 'apply_chat_template'):
        logger.error("Tokenizer does not have the method 'apply_chat_template'.")
        exit(1)
    
    logger.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    # get dataset path
    logger.info("getting dataset path...")
    dataset_path = ConfigLoader.get_fine_tuning_dataset_path(args.dataset_name, "predict_answer")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path {dataset_path} does not exist.")
        exit(1)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        exit(1)
    
    # Validate dataset columns
    if not {"Input", "Output"}.issubset(df.columns):
        logger.error("Dataset must contain 'Input' and 'Output' columns.")
        exit(1)
    
    # Create Dataset object
    logger.info("Processing dataset...")
    dataset = ChatDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    # Define training arguments
    output_dir = os.path.join(model_dir, args.model_name, args.dataset_name)
    logger.info(f"Training will be saved to {output_dir}...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    # Initialize data collator
    logger.info("Initializing data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    logger.info("Saving the fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete and model saved successfully.")

if __name__ == "__main__":
    main()
