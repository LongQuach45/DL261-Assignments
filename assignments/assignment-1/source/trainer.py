import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score

def train_and_validate(model, model_name, train_loader, val_loader, device, epochs=10, lr=1e-3, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_val_f1 = -1.0
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_name.lower()}_best.pth")
    total_train_time = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.perf_counter()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_time = time.perf_counter() - start_time
        total_train_time += epoch_train_time
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = np.mean(np.array(all_preds) == np.array(all_targets))
        epoch_val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), best_ckpt_path)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | "
              f"Val F1: {epoch_val_f1:.4f} | "
              f"Time: {epoch_train_time:.2f}s")
        
    return history, total_train_time, best_ckpt_path

def evaluate_test_set(model, ckpt_path, test_loader, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    collected_images = []
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs_dev = inputs.to(device)
            outputs = model(inputs_dev)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            if len(collected_images) < 256:
                collected_images.extend(inputs.numpy())
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets))
    test_f1 = f1_score(all_targets, all_preds, average='macro')
    
    return test_acc, test_f1, inference_time, np.array(all_targets), np.array(all_preds), np.array(collected_images)