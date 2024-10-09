import numpy as np
import pandas as pd
import torch
import traceback
from collections import Counter
from ecg200.data_utils import read_ucr, normalize_data, to_torch_tensors
from ecg200.models.SimpleCNN import Simple2DCNN
from ecg200.models.DeepCNN import Deep2DCNN
from ecg200.CNN_train import create_dataloaders, train_model, evaluate_model
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_file = 'data/ECG200/ECG200_TRAIN.ts'
test_file = 'data/ECG200/ECG200_TEST.ts'

x_train, y_train = read_ucr(train_file)
x_test, y_test = read_ucr(test_file)


unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

nb_classes = len(unique_labels)

x_train, x_test = normalize_data(x_train, x_test)

model_configurations = {
    'Simple2DCNN': Simple2DCNN,
    'Deep2DCNN': Deep2DCNN
}

excel_file_path = 'base_model.xlsx'

def write_results_to_excel(file_path, results, version):
    # Load the workbook or create a new one if it doesn't exist
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    results_df = pd.DataFrame([results])

    results_df['Version'] = version

    # Concatenate the new results with the existing DataFrame
    df = pd.concat([df, results_df], ignore_index=True)

    # Save the DataFrame back to the Excel file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)


def calculate_and_save_averages(file_path, model_name, combo_key):
    print(f"\nCalculating averages for model '{model_name}' and combination '{combo_key}'...\n")
    
    df = pd.read_excel(file_path)
    
    # Filter for the current set of results to calculate the average
    average_results = df.mean(numeric_only=True).to_dict()
    average_results['Model'] = model_name
    average_results['Combination'] = combo_key
    average_results['Version'] = 'Average'

    # Append the average results to the existing DataFrame
    df = pd.concat([df, pd.DataFrame([average_results])], ignore_index=True)
    df.to_excel(file_path, index=False)
    
    print(f"Averages saved for model '{model_name}' and combination '{combo_key}' to file: {file_path}")

# Loop through model configurations and dataloaders
for model_name, model_class in model_configurations.items():
    for combo_key, loaders in create_dataloaders(X_train, y_train, X_test, y_test).items():
        for iteration in range(10):
            print(f"--------------------- Running iteration {iteration + 1} with architecture: {model_name}, combination: {combo_key}  --------------")

            # Unpacking the dataloaders
            train_loader, val_loader, test_loader = loaders

            # Initialize the model
            model = model_class(3, nb_classes).to(device)  

            # Train the model
            num_epochs = 100
            best_val_loss, best_val_accuracy = train_model(model, train_loader, val_loader, num_epochs)

            # Evaluate the model
            metrics = evaluate_model(model, test_loader)

            # Collect the results
            results = {
                'Model': model_name,
                'Combination': combo_key,
                'Best Validation Loss': best_val_loss,
                'Best Validation Accuracy': best_val_accuracy,
                'Test Loss': metrics['test_loss'],
                'Test Accuracy': metrics['test_accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'AUC': metrics['auc'],
                'Balanced Accuracy': metrics['balanced_accuracy'],
                'Specificity': metrics['specificity']
            }

            # Save the results to Excel
            write_results_to_excel(excel_file_path, results, f'v{iteration + 1}')

            print(f"\nFinished iteration {iteration + 1} for combination: {combo_key} with model: {model_name}\n" + "="*50 + "\n")

        # Calculate and save averages for the model and combination after all iterations
        calculate_and_save_averages(excel_file_path, model_name, combo_key)