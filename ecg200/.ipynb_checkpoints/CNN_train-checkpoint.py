import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from ecg200.CNN_utils import TimeSeriesImageDatasetMC
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, balanced_accuracy_score
from imblearn.metrics import specificity_score
from sklearn.model_selection import StratifiedShuffleSplit
torch.backends.cudnn.enabled = False


from sklearn.model_selection import StratifiedShuffleSplit


def stratified_split(test_dataset, y_test, val_size=50):
    y_test = np.array(y_test)
    unique, counts = np.unique(y_test, return_counts=True)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    indices = list(range(len(test_dataset)))
    
    for test_idx, val_idx in sss.split(indices, y_test):
        val_dataset = torch.utils.data.Subset(test_dataset, val_idx)  
        test_dataset = torch.utils.data.Subset(test_dataset, test_idx) 
        
        val_labels = [y_test[i] for i in val_idx]
        test_labels = [y_test[i] for i in test_idx]
        
        val_unique, val_counts = np.unique(val_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)
        
    return val_dataset, test_dataset
        

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    chart_types = ['area', 'line', 'bar', 'scatter']
    color_modes = ['color', 'monochrome']
    label_modes = ['with_label', 'without_label']
    scatter_modes = ['plain']
    bar_modes = ['fill', 'border']

    dataloaders = {}
    
    for chart_type in chart_types:
        for color_mode in color_modes:
            for label_mode in label_modes:
                if chart_type == 'scatter':
                    for scatter_mode in scatter_modes:
                        # Log current configuration
                        print(f"Creating dataloaders for: chart_type={chart_type}, scatter_mode={scatter_mode}, color_mode={color_mode}, label_mode={label_mode}")
                        
                        try:
                            # Create dataset and loaders for scatter mode
                            train_dataset = TimeSeriesImageDatasetMC(
                                X_train, y_train, split='train', transform=transform,
                                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, scatter_mode=scatter_mode
                            )
                            test_dataset = TimeSeriesImageDatasetMC(
                                X_test, y_test, split='test', transform=transform,
                                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, scatter_mode=scatter_mode
                            )

                            val_dataset, test_dataset = stratified_split(test_dataset, y_test, val_size=50)

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                            combo_key = f"{chart_type}_{scatter_mode}_{color_mode}_{label_mode}"
                            dataloaders[combo_key] = (train_loader, val_loader, test_loader)
                        except Exception as e:
                            print(f"Error occurred while creating dataloaders for combination: {chart_type}_{scatter_mode}_{color_mode}_{label_mode}")
                            print(f"Error: {e}")

                elif chart_type == 'bar':
                    for bar_mode in bar_modes:
                        # Log current configuration
                        print(f"Creating dataloaders for: chart_type={chart_type}, bar_mode={bar_mode}, color_mode={color_mode}, label_mode={label_mode}")
                        
                        try:
                            train_dataset = TimeSeriesImageDatasetMC(
                                X_train, y_train, split='train', transform=transform,
                                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, bar_mode=bar_mode
                            )
                            test_dataset = TimeSeriesImageDatasetMC(
                                X_test, y_test, split='test', transform=transform,
                                chart_type=chart_type, color_mode=color_mode, label_mode=label_mode, bar_mode=bar_mode
                            )

                            val_dataset, test_dataset = stratified_split(test_dataset, y_test, val_size=50)

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                            combo_key = f"{chart_type}_{bar_mode}_{color_mode}_{label_mode}"
                            dataloaders[combo_key] = (train_loader, val_loader, test_loader)
                        except Exception as e:
                            print(f"Error occurred while creating dataloaders for combination: {chart_type}_{bar_mode}_{color_mode}_{label_mode}")
                            print(f"Error: {e}")

                else:
                    # Log current configuration
                    print(f"Creating dataloaders for: chart_type={chart_type}, color_mode={color_mode}, label_mode={label_mode}")
                    
                    try:
                        # General case for other chart types
                        train_dataset = TimeSeriesImageDatasetMC(
                            X_train, y_train, split='train', transform=transform,
                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode
                        )
                        test_dataset = TimeSeriesImageDatasetMC(
                            X_test, y_test, split='test', transform=transform,
                            chart_type=chart_type, color_mode=color_mode, label_mode=label_mode
                        )

                        # Perform stratified split
                        val_dataset, test_dataset = stratified_split(test_dataset, y_test, val_size=50)

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                        combo_key = f"{chart_type}_{color_mode}_{label_mode}"
                        dataloaders[combo_key] = (train_loader, val_loader, test_loader)
                    except Exception as e:
                        print(f"Error occurred while creating dataloaders for combination: {chart_type}_{color_mode}_{label_mode}")
                        print(f"Error: {e}")

    print("helo")
    return dataloaders

def train_model(model, train_loader, val_loader, num_epochs, patience=10, optimizer=None, scheduler=None):

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    if scheduler is None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # class_weights = torch.tensor([1.0, 1.2, 3.0, 4.5, 5.0]).to(device)  
    # criterion = nn.CrossEntropyLoss(weight = class_weights)
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0
    trigger_times = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        scheduler.step(val_loss)

        final_val_loss = val_loss

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
    print(f'Validation Loss: {final_val_loss:.4f}')  

    return final_val_loss, best_val_accuracy


def evaluate_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    y_true = []
    y_pred = []
    y_probs = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    specificity_per_class = []
    balanced_acc_per_class = []
    recall_per_class = []

    conf_matrix = confusion_matrix(y_true, y_pred)

    for i in range(len(conf_matrix)):
        recall_i = conf_matrix[i, i] / conf_matrix[i, :].sum() if conf_matrix[i, :].sum() > 0 else 0
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0
        recall_per_class.append(recall_i)
        specificity_per_class.append(specificity_i)

    # Average recall and specificity
    avg_recall = np.mean(recall_per_class)
    avg_specificity = np.mean(specificity_per_class)

    # Balanced accuracy
    balanced_acc = (avg_recall + avg_specificity) / 2

    recall = recall_score(y_true, y_pred, average = "weighted")
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    auc = roc_auc_score(y_true, y_probs, multi_class='ovr') 
    
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')
    print(f'AUC: {auc:.2f}')
    print(f'Specificity: {avg_specificity:.2f}')
    print(f'Balanced Accuracy: {balanced_acc*100:.2f}%')
    print('Confusion Matrix:')
    print(conf_matrix)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,  
        'f1_score': f1,
        'auc': auc,
        'specificity': avg_specificity,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': conf_matrix
    }
