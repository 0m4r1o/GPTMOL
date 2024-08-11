import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
import torch

# Step 1: Load and preprocess data
data = pd.read_csv('smiles.csv')
valid_smiles = [smile for smile in data['smiles'] if Chem.MolFromSmiles(smile) is not None]
train_smiles, test_smiles = train_test_split(valid_smiles, test_size=0.2)
# exit(1)

def tokenize_smiles(smiles):
    return list(smiles)

train_smiles_tokenized = [tokenize_smiles(smile) for smile in train_smiles]
test_smiles_tokenized = [tokenize_smiles(smile) for smile in test_smiles]

# Step 2: Define dataset class
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, tokenizer):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        input_ids = self.tokenizer.encode(smiles, return_tensors='pt').squeeze()
        return input_ids

def collate_fn(batch):
    # Pad sequences to the same length
    max_length = max([item.size(0) for item in batch])
    padded_batch = torch.stack([F.pad(item, (0, max_length - item.size(0)), value=tokenizer.pad_token_id) for item in batch])
    return padded_batch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = {'additional_special_tokens': ['<START>', '<END>']}
tokenizer.add_special_tokens(special_tokens)

train_dataset = SMILESDataset(train_smiles_tokenized, tokenizer)
test_dataset = SMILESDataset(test_smiles_tokenized, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Step 3: Fine-tune the model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

epochs = 1
learning_rate = 5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
# model.to('cuda')

model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

model.save_pretrained('fine_tuned_gpt2_smiles')
tokenizer.save_pretrained('fine_tuned_gpt2_smiles')

# Step 4: Generate new SMILES
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_smiles')
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_gpt2_smiles')
model.eval()
# model.to('cuda')

def generate_smiles(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors='pt')#.to('cuda')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    smiles = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return smiles

# Generate new SMILES strings
generated_smiles_list = []
for _ in range(100):  # Generate 100 new SMILES strings
    prompt = "<START>"
    generated_smiles = generate_smiles(prompt)
    if Chem.MolFromSmiles(generated_smiles):  # Validate SMILES
        generated_smiles_list.append(generated_smiles)

print(f"Generated {len(generated_smiles_list)} valid SMILES strings.")

# Step 5: Incorporate new molecules into training data
train_smiles.extend(generated_smiles_list)
train_smiles_tokenized = [tokenize_smiles(smile) for smile in train_smiles]

train_dataset = SMILESDataset(train_smiles_tokenized, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Step 6: Retrain the model
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2_smiles')
model.resize_token_embeddings(len(tokenizer))
# model.to('cuda')

epochs = 1 # Number of additional epochs
learning_rate = 5e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        # batch = batch.to('cuda')
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

model.save_pretrained('retrained_gpt2_smiles')
tokenizer.save_pretrained('retrained_gpt2_smiles')
