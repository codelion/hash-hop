import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import argparse
from hashhop.generate import MultiHopEval
import random
import os
import gc

class HashHopDataset(Dataset):
    def __init__(self, num_samples, n_chars_problem, num_queries, hops, hash_pair_str_length, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        for _ in range(num_samples):
            datapoint = MultiHopEval.make_one(
                n_chars_problem=n_chars_problem,
                num_queries=num_queries,
                hops=hops,
                hash_pair_str_length=hash_pair_str_length,
                chain_of_thought=True,
            )
            self.data.append((datapoint.prompt, datapoint.completion))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt, completion = self.data[idx]
        
        # Process completion to extract queries and answers
        lines = completion.split('\n')[5:]  # Skip the first 5 lines
        queries = []
        cot_answers = []
        final_answers = []
        for line in lines:
            # print(line)
            if '=' in line:
                query, answer = line.split(' = ', 1)
                cot, final = answer.strip("'").split(' = ')
                queries.append(query)
                cot_answers.append(cot)
                final_answers.append(final)
        
        inputs = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        queries_encoded = self.tokenizer.batch_encode_plus(
            queries,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        cot_answers_encoded = self.tokenizer.batch_encode_plus(
            cot_answers,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        final_answers_encoded = self.tokenizer.batch_encode_plus(
            final_answers,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'query_ids': queries_encoded['input_ids'].squeeze(),
            'query_mask': queries_encoded['attention_mask'].squeeze(),
            'cot_ids': cot_answers_encoded['input_ids'].squeeze(),
            'cot_mask': cot_answers_encoded['attention_mask'].squeeze(),
            'final_ids': final_answers_encoded['input_ids'].squeeze(),
            'final_mask': final_answers_encoded['attention_mask'].squeeze()
        }

class HashHopLearner(nn.Module):
    def __init__(self, encoder, hidden_size, vocab_size, num_attention_layers=3, num_reasoning_steps=3):
        super(HashHopLearner, self).__init__()
        self.encoder = encoder
        self.self_attention_layers = nn.ModuleList([nn.MultiheadAttention(hidden_size, num_heads=8) for _ in range(num_attention_layers)])
        self.reasoning_module = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_reasoning_steps)])
        self.cot_decoder = nn.Linear(hidden_size, vocab_size)
        self.final_decoder = nn.Linear(hidden_size, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids, attention_mask, query_ids):
        # Encode input
        encoded = self.encoder(input_ids, attention_mask=attention_mask)[0]
        encoded = self.layer_norm(encoded)
        
        # Self-attention
        for attention_layer in self.self_attention_layers:
            attended, _ = attention_layer(encoded, encoded, encoded)
            encoded = self.layer_norm(encoded + attended)  # Add residual connection and layer norm
        
        # Reasoning steps
        for reasoning_step in self.reasoning_module:
            reasoned = reasoning_step(encoded)
            encoded = self.layer_norm(encoded + F.relu(reasoned))  # Add residual connection, ReLU, and layer norm
        
        # Query-based attention
        batch_size, num_queries, query_length = query_ids.shape
        query_ids_flat = query_ids.view(-1, query_length)
        query_encoded = self.encoder(query_ids_flat)[0]
        query_encoded = query_encoded.view(batch_size, num_queries, -1, query_encoded.size(-1))
        query_encoded = self.layer_norm(query_encoded)
        
        # Perform attention for each query
        attended_outputs = []
        for i in range(num_queries):
            attention_weights = torch.bmm(query_encoded[:, i], encoded.transpose(1, 2))
            attention_weights = F.softmax(attention_weights, dim=-1)
            attended = torch.bmm(attention_weights, encoded)
            attended_outputs.append(attended)
        
        attended = torch.stack(attended_outputs, dim=1)
        attended = self.layer_norm(attended)
        
        # Decode output
        cot_output = self.cot_decoder(attended)
        final_output = self.final_decoder(attended)
        
        return cot_output, final_output

def train(model, train_loader, val_loader, device, num_epochs, learning_rate):
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.NLLLoss(ignore_index=0)  # Assuming 0 is the pad token
    
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            query_ids = batch['query_ids'].to(device)
            cot_ids = batch['cot_ids'].to(device)
            final_ids = batch['final_ids'].to(device)
            
            optimizer.zero_grad()
            
            cot_outputs, final_outputs = model(input_ids, attention_mask, query_ids)
            cot_loss = criterion(F.log_softmax(cot_outputs, dim=-1).view(-1, cot_outputs.size(-1)), cot_ids.view(-1))
            final_loss = criterion(F.log_softmax(final_outputs, dim=-1).view(-1, final_outputs.size(-1)), final_ids.view(-1))
            loss = cot_loss + final_loss
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
            else:
                print(f"Skipping batch due to invalid loss: {loss.item()}")
            
            # Clear unnecessary tensors from GPU memory
            # del input_ids, attention_mask, query_ids, cot_ids, final_ids, cot_outputs, final_outputs, loss
            # torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                query_ids = batch['query_ids'].to(device)
                cot_ids = batch['cot_ids'].to(device)
                final_ids = batch['final_ids'].to(device)
                
                cot_outputs, final_outputs = model(input_ids, attention_mask, query_ids)
                cot_loss = criterion(F.log_softmax(cot_outputs, dim=-1).view(-1, cot_outputs.size(-1)), cot_ids.view(-1))
                final_loss = criterion(F.log_softmax(final_outputs, dim=-1).view(-1, final_outputs.size(-1)), final_ids.view(-1))
                loss = cot_loss + final_loss
                val_loss += loss.item()

                # Clear unnecessary tensors from GPU memory
                # del input_ids, attention_mask, query_ids, cot_ids, final_ids, cot_outputs, final_outputs, loss
                # torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_hashhop_learner.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
def reinforce_fine_tune(model, dataloader, device, num_episodes, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for episode in range(num_episodes):
        total_reward = 0
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"RL Episode {episode+1}/{num_episodes}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            query_ids = batch['query_ids'].to(device)
            final_ids = batch['final_ids'].to(device)
            
            # Forward pass
            _, logits = model(input_ids, attention_mask, query_ids)
            
            # Reshape logits and final_ids to match
            batch_size, num_queries, seq_length, vocab_size = logits.shape
            logits = logits.view(-1, seq_length, vocab_size)
            final_ids = final_ids.view(-1, seq_length)
            
            # Sample actions (predictions) from the model's output distribution
            probs = F.softmax(logits, dim=-1)
            actions = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)
            
            # Compute rewards (accuracy of final answer)
            rewards = compute_accuracy(actions, final_ids)
            
            # Debug output
            # print(f"Batch {batch_idx}, Rewards: min={rewards.min().item():.4f}, max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")
            
            total_reward += rewards.mean().item()
            
            # Compute log probabilities of chosen actions
            log_probs = torch.log(probs.gather(2, actions.unsqueeze(2)).squeeze(2) + 1e-8)
            
            # Debug output
            # print(f"Batch {batch_idx}, Log Probs: min={log_probs.min().item():.4f}, max={log_probs.max().item():.4f}, mean={log_probs.mean().item():.4f}")
            
            # Compute loss
            loss = -(log_probs * rewards.unsqueeze(1)).mean()
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected in batch {batch_idx}")
                print(f"Rewards: {rewards}")
                print(f"Log Probs: {log_probs}")
                continue  # Skip this batch
            
            total_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        avg_reward = total_reward / len(dataloader)
        avg_loss = total_loss / len(dataloader)
        # print(f"RL Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
        
        # Early stopping if rewards are consistently zero
        if episode > 3 and avg_reward < 1e-6:
            print("Early stopping due to consistently low rewards")
            break

def compute_accuracy(predicted, target):
    # Compute accuracy only for non-padding tokens
    mask = (target != 0)  # Assuming 0 is the padding token
    correct = ((predicted == target) * mask).sum(dim=1)
    total = mask.sum(dim=1)
    accuracy = (correct.float() / total.float()).clamp(min=1e-8, max=1.0)  # Avoid division by zero and cap at 1.0
    
    # Debug output
    # print(f"Accuracy: min={accuracy.min().item():.4f}, max={accuracy.max().item():.4f}, mean={accuracy.mean().item():.4f}")
    
    return accuracy

def evaluate_model(model, dataloader, device):
    model.eval()
    total_reward = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            query_ids = batch['query_ids'].to(device)
            final_ids = batch['final_ids'].to(device)
            
            # Forward pass
            _, logits = model(input_ids, attention_mask, query_ids)
            
            # Reshape logits and final_ids to match
            batch_size, num_queries, seq_length, vocab_size = logits.shape
            logits = logits.view(-1, seq_length, vocab_size)
            final_ids = final_ids.view(-1, seq_length)
            
            # Use argmax for evaluation instead of sampling
            actions = logits.argmax(dim=-1)
            
            # Compute rewards (accuracy of final answer)
            rewards = compute_accuracy(actions, final_ids)
            
            total_reward += rewards.mean().item()
            num_batches += 1
    
    avg_reward = total_reward / num_batches
    return avg_reward

def analyze_dataset(dataloader):
    total_samples = 0
    total_queries = 0
    total_tokens = 0
    unique_hashes = set()
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        final_ids = batch['final_ids']
        
        total_samples += input_ids.size(0)
        total_queries += final_ids.size(1)
        total_tokens += (input_ids != 0).sum().item()
        
        # Assuming the first token of each query in final_ids is the hash
        hashes = final_ids[:, :, 0].view(-1)
        unique_hashes.update(hashes.tolist())
    
    print(f"Dataset Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Total queries: {total_queries}")
    print(f"Average queries per sample: {total_queries / total_samples:.2f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per sample: {total_tokens / total_samples:.2f}")
    print(f"Unique hashes: {len(unique_hashes)}")

def main():
    parser = argparse.ArgumentParser(description="Train HashHop Learner model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--train_samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=10, help="Number of validation samples")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--rl_episodes", type=int, default=10, help="Number of RL fine-tuning episodes")
    parser.add_argument("--rl_lr", type=float, default=1e-5, help="Learning rate for RL fine-tuning")
    args = parser.parse_args()
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Initialize tokenizer and encoder
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = BertModel.from_pretrained('bert-base-uncased')

    # Create datasets and dataloaders
    train_dataset = HashHopDataset(args.train_samples, 1000, 5, 2, 16, tokenizer)
    val_dataset = HashHopDataset(args.val_samples, 1000, 5, 2, 16, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Initialize model
    model = HashHopLearner(encoder, hidden_size=768, vocab_size=tokenizer.vocab_size).to(device)

    if args.resume and os.path.exists('best_hashhop_learner.pt'):
        print("Loading checkpoint...")
        checkpoint = torch.load('best_hashhop_learner.pt', map_location=device)
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded. Continuing training.")
    elif args.resume:
        print("No checkpoint found. Starting from scratch.")

    # Train model
    train(model, train_loader, val_loader, device, args.num_epochs, args.learning_rate)
    # analyze_dataset(train_loader)
    # RL fine-tuning
    print("Starting RL fine-tuning...")
    reinforce_fine_tune(model, train_loader, device, args.rl_episodes, args.rl_lr)

    # Final evaluation
    final_accuracy = evaluate_model(model, val_loader, device)
    print(f"Final accuracy after RL fine-tuning: {final_accuracy:.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'final_hashhop_learner_with_rl.pt')

if __name__ == "__main__":
    main()