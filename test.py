import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import os
import subprocess
from chemprop.train import run_training
from chemprop.data import get_data
from chemprop.args import TrainArgs, PredictArgs



# prefixes = ['temp', 'pred']
# for prefix in prefixes:
#     files_in_directory = os.listdir('/')
#     files_to_delete = [file for file in files_in_directory if file.startswith(prefix)]
#     for file in files_to_delete:
#         file_path = os.path.join(directory, file)
#         os.remove(file_path)
#         print(f'Deleted: {file_path}')

def save_temp_data(data, file_path, target_column):
    # Ensure SMILES strings are correctly formatted
    df = pd.DataFrame([(d.smiles[0].strip('[]').strip("'"), d.targets[0]) for d in data], columns=['smiles', target_column])
    df.to_csv(file_path, index=False)

def debug_csv(file_path):
    # Print out the contents of the CSV file for debugging
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # print(f"Contents of {file_path}:")
        # print(df.head())
        # print(f"Number of rows: {len(df)}")
    else:
        print(f"File {file_path} does not exist.")

def train_chemprop(train_data_path, save_dir, target_column, num_folds=5, num_trials=50):
    metrics = {'auc': [], 'f1': [], 'accuracy': []}

    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}...")
        
        # Load the dataset
        data = get_data(path=train_data_path, skip_none_targets=True)

        # Extract features and targets
        targets = np.array([d.targets for d in data]).flatten()
        features = [d.smiles for d in data]

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=trial)

        fold_metrics = {'auc': [], 'f1': [], 'accuracy': []}

        for fold, (train_idx, val_idx) in enumerate(kf.split(features)):
            print(f"  Starting fold {fold+1}/{num_folds}...")

            # Split data into training and validation sets
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            # Save temporary train and validation data
            temp_train_path = f'temp_train_trial_{trial}_fold_{fold}.csv'
            temp_val_path = f'temp_val_trial_{trial}_fold_{fold}.csv'

            save_temp_data(train_data, temp_train_path, target_column)
            save_temp_data(val_data, temp_val_path, target_column)

            # Debug CSV files
            debug_csv(temp_train_path)
            debug_csv(temp_val_path)

            # Define the command to run training
            command = [
                'chemprop_train',
                '--data_path', temp_train_path,
                '--dataset_type', 'classification',
                '--save_dir', os.path.join(save_dir, f'trial_{trial}_fold_{fold}'),
                '--target_columns', target_column,
                '--epochs', '35',
                '--batch_size', '10',
            ]

            try:
                # Run the training command
                subprocess.run(command, check=True)

                # Define the command to run prediction
                predict_command = [
                    'chemprop_predict',
                    '--test_path', temp_val_path,
                    '--checkpoint_dir', os.path.join(save_dir, f'trial_{trial}_fold_{fold}'),
                    '--preds_path', f'preds_trial_{trial}_fold_{fold}.csv'
                ]

                # Run the prediction command
                subprocess.run(predict_command, check=True)

                # Debug prediction file
                debug_csv(f'preds_trial_{trial}_fold_{fold}.csv')

                # Calculate metrics
                preds_df = pd.read_csv(f'preds_trial_{trial}_fold_{fold}.csv')
                # Check the column names and choose the correct one for predictions
                print(f"Prediction file columns: {preds_df.columns}")

                # Replace 'preds' with the actual column name for predictions
                pred_column_name = 'preds'  # Change if necessary based on actual column name
                if pred_column_name not in preds_df.columns:
                    raise KeyError(f"Column '{pred_column_name}' not found in predictions file.")

                # Convert predictions to float
                preds = preds_df[pred_column_name].astype(float)
                true_labels = [d.targets[0] for d in val_data]
                
                # Binarize predictions
                threshold = 0.5
                binary_preds = [1 if x > threshold else 0 for x in preds]

                auc = roc_auc_score(true_labels, preds)
                f1 = f1_score(true_labels, binary_preds)
                accuracy = accuracy_score(true_labels, binary_preds)
                
                fold_metrics['auc'].append(auc)
                fold_metrics['f1'].append(f1)
                fold_metrics['accuracy'].append(accuracy)

            except subprocess.CalledProcessError as e:
                print(f"Error during training or prediction for trial {trial} fold {fold}: {e}")
                continue
            except KeyError as e:
                print(f"KeyError: {e}")
                continue
            except TypeError as e:
                print(f"TypeError: {e}")
                continue

            # Clean up temporary files
            os.remove(temp_train_path)
            os.remove(temp_val_path)
            os.remove(f'preds_trial_{trial}_fold_{fold}.csv')

        # Record the average metrics for this trial
        metrics['auc'].append(np.mean(fold_metrics['auc']))
        metrics['f1'].append(np.mean(fold_metrics['f1']))
        metrics['accuracy'].append(np.mean(fold_metrics['accuracy']))

    # Calculate the average of each metric over all trials
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}

    return avg_metrics

def main():
    data_path = 'data/TG2_cleaned_smiles_activity.csv'  # Update this with your data path
    save_dir = 'acheckp'  # Update this with your checkpoint directory
    target_column = 'Activity'  # Update this with your target column name

    metrics = train_chemprop(data_path, save_dir, target_column)

    print('Average metrics over 50 trials of 5-fold cross-validation:')
    print(f"AUC: {metrics['auc']}")
    print(f"F1 Score: {metrics['f1']}")
    print(f"Accuracy: {metrics['accuracy']}")



import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import numpy as np

def evaluate_predictions(predictions_path, test_path):
    # Load predictions and testing data
    preds_df = pd.read_csv(predictions_path)
    test_df = pd.read_csv(test_path)
    # preds_df["Activity2"] = preds_df["Activity"]
    # preds_df.drop(["Activity"], axis = 1)
    preds_df.rename(columns={'Activity': 'Activity2'}, inplace=True)

    

    # Ensure the data has 'smiles' and 'preds' columns for predictions and 'smiles' and 'Activity' for testing
    if 'smiles' not in preds_df.columns or 'Activity2' not in preds_df.columns:
        raise ValueError("Prediction file must contain 'smiles' and 'Activity2' columns.")
    if 'smiles' not in test_df.columns or 'Activity' not in test_df.columns:
        raise ValueError("Testing file must contain 'smiles' and 'Activity' columns.")

    # Filter out rows with 'Invalid Smiles' in predictions
    preds_df = preds_df[preds_df['Activity2'] != 'Invalid Smiles']
    
    # Convert valid predictions to numeric
    preds_df['Activity2'] = pd.to_numeric(preds_df['Activity2'], errors='coerce')

    # Merge predictions with the testing data on 'smiles'
    merged_df = pd.merge(test_df, preds_df, on='smiles', how='inner')

    # Check if we have any valid data after merging
    if merged_df.empty:
        raise ValueError("No valid predictions found after merging.")

    # Extract true labels and predicted values
    true_labels = merged_df['Activity']
    preds = merged_df['Activity2'].astype(float)

    # Binarize predictions
    threshold = 0.5
    binary_preds = (preds > threshold).astype(int)

    # Calculate metrics
    auc = roc_auc_score(true_labels, binary_preds)
    f1 = f1_score(true_labels, binary_preds)
    accuracy = accuracy_score(true_labels, binary_preds)

    return auc, f1, accuracy

def main2():
    results = []

    # Loop through trials and folds
    num_trials = 50
    num_folds = 5

    for trial in range(num_trials):
        for fold in range(num_folds):
            pred_file = f'preds_trial_{trial}_fold_{fold}.csv'
            test_file = f'temp_val_trial_{trial}_fold_{fold}.csv'

            if os.path.exists(pred_file) and os.path.exists(test_file):
                # print(f"Evaluating trial {trial}, fold {fold}...")

                auc, f1, accuracy = evaluate_predictions(pred_file, test_file)
                results.append({
                        'trial': trial,
                        'fold': fold,
                        'AUC': auc,
                        'F1': f1,
                        'Accuracy': accuracy
                })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('full_result_metrics.csv2', index=False)
    
    # Calculate average metrics
    avg_results = results_df.groupby(['trial']).mean().reset_index()
    avg_results = avg_results.drop(["fold"], axis = 1)
    std_metrics = results_df.groupby(['trial']).std().reset_index()
    std_metrics = std_metrics.drop(["fold"], axis = 1)
    avg_results.to_csv('average_metrics2.csv', index=False)
    std_metrics.to_csv('avg_std_metrics2.csv', index = False)
    print("Evaluation complete. Results saved to 'average_metrics.csv'.")

    avg_AUC = avg_results['AUC'].mean()
    avg_F1 = avg_results['F1'].mean()
    avg_acc = avg_results['Accuracy'].mean()

    std_avg_AUC = std_metrics['AUC'].mean()
    std_avg_F1 = std_metrics['F1'].mean()
    std_avg_acc = std_metrics['Accuracy'].mean()

    print("Average AUC: " + str(avg_AUC) + " +- " +  str(std_avg_AUC))
    print("Average F1: " + str(avg_F1) + " +- " +  str(std_avg_F1))
    print("Average Accuracy: " + str(avg_acc) + " +- " +  str(std_avg_acc))


if __name__ == "__main__":
    main()
    main2()
