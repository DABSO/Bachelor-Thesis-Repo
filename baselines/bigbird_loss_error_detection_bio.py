import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BigBirdTokenizerFast,
    BigBirdForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from src.ConfigLoader import ConfigLoader
from dotenv import load_dotenv

load_dotenv()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the DataFrame from the JSON file
training_dataset_path = ConfigLoader.get_input_dataset_path("human_annotated_train")
df = pd.read_json(training_dataset_path, orient='records', lines=True,dtype=False)  

# Ensure required columns are present
assert all(col in df.columns for col in ['question', 'context', 'label']), \
    "Dataframe must contain 'question', 'context', and 'label' columns"

# Initialize the tokenizer
tokenizer = BigBirdTokenizerFast.from_pretrained('google/bigbird-roberta-large')

# Function to align entities with tokens using BIO tagging
def align_labels_with_tokens(text, question, entities, tokenizer):
    tokenized_inputs = tokenizer(question, text, truncation=True, return_offsets_mapping=True)
    labels = [0] * len(tokenized_inputs['input_ids'])  # Initialize labels to 'O'
    offset_mapping = tokenized_inputs['offset_mapping']
    

    for entity in entities:
        entity = entity.strip()
        if not entity:
            continue
        start = 0
        while True:
            idx = text.find(entity, start)
            if idx == -1:
                break
            end_idx = idx + len(entity)
            start = end_idx
            for idx_token, (token_start, token_end) in enumerate(offset_mapping):
                if token_start is None or token_end is None:
                    continue
                if token_end <= idx:
                    continue
                elif token_start >= end_idx:
                    break
                else:
                    if token_start == idx:
                        labels[idx_token] = 1  # 'B-ENTITY'
                    else:
                        labels[idx_token] = 2  # 'I-ENTITY'
    return tokenized_inputs, labels

# Custom Dataset class
class CustomNERDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=4096):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['context']
        question = row['question']

        entities = [e.strip() for e in row['label'].split(';') if e.strip()]
        
        
        

        tokenized_inputs, labels = align_labels_with_tokens(text, question ,entities, self.tokenizer)
        # Truncate to max_length
        for key in tokenized_inputs:
            tokenized_inputs[key] = tokenized_inputs[key][:self.max_length]
        labels = labels[len():]

        # Return as tensors
        item = {key: torch.tensor(val) for key, val in tokenized_inputs.items() if key != 'offset_mapping'}
        item['labels'] = torch.tensor(labels)
        return item

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = CustomNERDataset(train_df, tokenizer)
test_dataset = CustomNERDataset(test_df, tokenizer)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# 2. Model Setup
model = BigBirdForTokenClassification.from_pretrained('google/bigbird-roberta-large', num_labels=3)
model.to(device)

# Define class weights to handle class imbalance
class_weights = torch.tensor([0.1, 1.0, 1.0]).to(device)  # Weights for 'O', 'B-ENTITY', 'I-ENTITY'
loss_fct = CrossEntropyLoss(weight=class_weights)

# Custom Trainer class to include custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Compute the loss with class weights
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# 3. Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    per_device_eval_batch_size=2,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

# 4. Train the model
trainer.train()

# 5. Evaluation Function to compute per-instance loss
def evaluate_per_instance(model, dataset, loss_fct):
    model.eval()
    losses = []
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=data_collator)
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            losses.append(loss.item())
    return losses

# Compute the loss per test instance
test_losses = evaluate_per_instance(model, test_dataset, loss_fct)
