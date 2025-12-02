import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
import performance_measures.classification_measures as cm
import performance_measures.regression_measures as rm
import models.MLP.MLP as mlp
import models.AutoEncoders.AutoEncoders as AE
import models.knn.knn as knn
import wandb

print('Let us start working on Assignment 3')

print('Enter 1.If you want to go through classification using MLP')
print('Enter 2.If you want to go through regression using MLP')
print('Enter 3.If you want to go through Autoencoders')

choice = int(input())

if choice == 1:
    print('Enter yes/no.If you want to do preprocessing for WineQT dataset')
    command = input()
    if command == 'yes':
        df = pd.read_csv('../../data/external/WineQT.csv')

        # Describe the data
        for column in df.columns:
            print('Description of column:', column)
            print(df[column].describe())

        df = df.drop(columns=['Id'])

        # Distribution of various columns
        for x in df.columns:
            data = df[x]
            plt.hist(data, bins=10)
            plt.title(x)
            plt.savefig('figures/histogram_'+x+'.png')
            plt.close()

        # Normalize the data
        for x in df.columns:
            if x != 'quality':
                df[x] = (df[x] - df[x].mean())/df[x].std()

        # One hot encoding the quality column
        df = pd.get_dummies(df, columns=['quality'])
        df.replace({True: 1, False: 0}, inplace=True)

        # Divide the data into train,validation and test data 
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        # splitting the data into train, validation and test set
        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        train_set = df[:train_len]
        valid_set = df[train_len:train_len+valid_len]
        test_set = df[train_len+valid_len:]

        # Save the data
        train_set.to_csv('../../data/interim/3/WineQT_train.csv', index=False)
        valid_set.to_csv('../../data/interim/3/WineQT_valid.csv', index=False)
        test_set.to_csv('../../data/interim/3/WineQT_test.csv', index=False)

    print('Enter yes/no. If you want to see whether the MLP is working fine')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/WineQT_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/WineQT_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/WineQT_test.csv')

        X_train = train_set.iloc[:, :-6].values
        y_train = train_set.iloc[:, -6:].values

        X_valid = valid_set.iloc[:, :-6].values
        y_valid = valid_set.iloc[:, -6:].values

        X_test = test_set.iloc[:, :-6].values
        y_test = test_set.iloc[:, -6:].values

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        # Train the model
        active_func = ['relu','softmax']
        neurons = [64]
        model = mlp.MLP(1, neurons, active_func)
        model.fit(X_train, y_train, 'mini-batch', 'cross_entropy', 0.1, 1, 32, X_valid, y_valid)
        model.check_gradients(X_train, y_train, epsilon=1e-7)

        model.fit(X_train, y_train, 'mini-batch', 'cross_entropy', 0.1, 100, 32, X_valid, y_valid)

        y_pred = model.predict(X_valid)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test_class = np.argmax(y_valid, axis=1)

        accuracy = np.mean(y_pred_class == y_test_class)
        print(accuracy)

    print('Enter yes/no. If you want to do hyperparameter tuning')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/WineQT_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/WineQT_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/WineQT_test.csv')

        X_train = train_set.iloc[:, :-6].values
        y_train = train_set.iloc[:, -6:].values

        X_valid = valid_set.iloc[:, :-6].values
        y_valid = valid_set.iloc[:, -6:].values

        X_test = test_set.iloc[:, :-6].values
        y_test = test_set.iloc[:, -6:].values

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        best_acc = 0

        def train_model():
            global best_acc
            wandb.init()
            hyperparameters = wandb.config
            
            print(wandb.config)

            activations = [hyperparameters['activations']] * (len(hyperparameters['hidden_layer_config']))
            activations.append('softmax')
            
            print(len(hyperparameters["hidden_layer_config"]), hyperparameters["hidden_layer_config"],activations)
            model = mlp.MLP(len(hyperparameters["hidden_layer_config"]), hyperparameters["hidden_layer_config"],activations)
            
            train_loss, train_acc, train_recall, train_precision, train_f1, valid_loss, valid_acc, valid_recall, valid_precision, valid_f1 = model.fit(X_train, y_train, hyperparameters['optimizer'],'cross_entropy', hyperparameters['lr'], hyperparameters['epochs'], 32, X_valid, y_valid)        
            
            for i in range(0, hyperparameters['epochs']):
                wandb.log({"train_loss": train_loss[i],
                            "val_loss": valid_loss[i],
                        "train_accuracy": train_acc[i],
                        "val_accuracy": valid_acc[i],
                            "train_precision": train_precision[i],
                            "val_precision": valid_precision[i],
                            "train_recall": train_recall[i],
                            "val_recall": valid_recall[i],
                            "train_f1": train_f1[i],
                            "val_f1": valid_f1[i]})


        sweep_config = {
            "method": "grid",
            "metric": {
                "goal": "maximize",
                "name": "val_accuracy"
            },
            "parameters": {
                "hidden_layer_config": {
                    "values": [[32], [32, 32], [64], [64, 64], [64, 32]]
                },
                "activations": {
                    "values": ["relu", "sigmoid", "tanh"]
                },
                "epochs": {
                    "values": [100, 500, 300]
                },
                "lr": {
                    "values": [0.001, 0.01, 0.1]
                },
                "optimizer": {
                    "values": ["stochastic", "batch", "mini-batch"]
                }   
            }
            
        }


        sweep_id = wandb.sweep(sweep_config, project="smai_a3")
        wandb.agent(sweep_id, function=train_model)

        api = wandb.Api()
        sweep_id = f"keshava-kishora-iiit-hyderabad/smai_a3/{sweep_id}"

        sweep = api.sweep(sweep_id)
        runs = sweep.runs

        best_run = sorted(runs, key=lambda run: run.summary.get('val_accuracy'), reverse=True)[0]

        best_hyperparameters = best_run.config
        print("Best Hyperparameters:", best_hyperparameters)

    print('Enter yes/no. If you want to do testing on the best model')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/WineQT_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/WineQT_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/WineQT_test.csv')

        X_train = train_set.iloc[:, :-6].values
        y_train = train_set.iloc[:, -6:].values

        X_valid = valid_set.iloc[:, :-6].values
        y_valid = valid_set.iloc[:, -6:].values

        X_test = test_set.iloc[:, :-6].values
        y_test = test_set.iloc[:, -6:].values

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        # Best model on test set
        best_lr = 0.1
        best_epochs = 500
        best_hidden_layer_config = [64]
        best_optimizer = 'mini-batch'
        activations = ['relu', 'softmax']

        model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations)

        train_loss, train_acc, train_recall, train_precision, train_f1, valid_loss, valid_acc, valid_recall, valid_precision, valid_f1 = model.fit(X_train, y_train, best_optimizer,'cross_entropy', best_lr, best_epochs, 32, X_valid, y_valid)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test_class = np.argmax(y_test, axis=1)

        metrics = cm.Measures(y_test_class, y_pred_class)

        print("Accuracy for best model: ", metrics.accuracy())
        print("Precision for best model: ", metrics.precision_macro())
        print("Recall for best model: ", metrics.recall_macro())
        print("F1 Score for best model: ", metrics.f1_score_macro())

    print('Enter yes/no. If you want to do hyperparameter Analysis')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/WineQT_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/WineQT_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/WineQT_test.csv')

        X_train = train_set.iloc[:, :-6].values
        y_train = train_set.iloc[:, -6:].values

        X_valid = valid_set.iloc[:, :-6].values
        y_valid = valid_set.iloc[:, -6:].values

        X_test = test_set.iloc[:, :-6].values
        y_test = test_set.iloc[:, -6:].values

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)


        best_lr = 0.01
        best_epochs = 100
        best_hidden_layer_config = [64, 32]
        best_optimizer = 'batch'
        activations = ['relu', 'tanh', 'sigmoid']

        #1. Create subplots (1 row, 3 columns for each activation)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        for i, activation in enumerate(activations):
            activations_config = [activation] * len(best_hidden_layer_config)
            activations_config.append('softmax')
            
            model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations_config)
            train_loss, train_acc, train_recall, train_precision, train_f1, valid_loss,valid_acc, valid_recall, valid_precision, valid_f1 = model.fit(
                X_train, y_train, best_optimizer, 'cross_entropy', best_lr, best_epochs, 32, X_valid, y_valid)
            
            axs[i].plot(range(1, best_epochs + 1), valid_loss, label=f'{activation} loss')
            axs[i].set_title(f'Training Loss ({activation})')
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel('Loss')
            axs[i].legend()

        plt.tight_layout()

        plt.savefig('figures/Classification_activation_loss.png')
        plt.close()

        #2. Create subplots (1 row 3 columns for each learning rate)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        for i, lr in enumerate([0.001, 0.01, 0.1]):
            activations_config = ['relu'] * len(best_hidden_layer_config)
            activations_config.append('softmax')
            
            model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations_config)
            train_loss, train_acc, train_recall, train_precision, train_f1, valid_loss,valid_acc, valid_recall, valid_precision, valid_f1 = model.fit(
                X_train, y_train, best_optimizer, 'cross_entropy', lr, best_epochs, 32, X_valid, y_valid)
            
            axs[i].plot(range(1, best_epochs + 1), valid_loss, label=f'{lr} loss')
            axs[i].set_title(f'Training Loss (lr={lr})')
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel('Loss')
            axs[i].legend()

        plt.tight_layout()

        plt.savefig('figures/Classification_lr_loss.png')
        plt.close()

        #3. Create subplots (1 row 3 columns for each batch size)
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        for i, batch_size in enumerate([16, 32, 64]):
            activations_config = ['relu'] * len(best_hidden_layer_config)
            activations_config.append('softmax')
            
            model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations_config)
            train_loss, train_acc, train_recall, train_precision, train_f1, valid_loss,valid_acc, valid_recall, valid_precision, valid_f1 = model.fit(
                X_train, y_train, best_optimizer, 'cross_entropy', best_lr, best_epochs, batch_size, X_valid, y_valid)
            
            axs[i].plot(range(1, best_epochs + 1), valid_loss, label=f'{batch_size} loss')
            axs[i].set_title(f'Training Loss (batch_size={batch_size})')
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel('Loss')
            axs[i].legend()

        plt.tight_layout()

        plt.savefig('figures/Classification_batchsize_loss.png')
        plt.close()

    print('Enter yes/no. If you want to do multi-label classification using MLP')
    command = input()
    if command == 'yes':
        df = pd.read_csv('../../data/external/advertisement.csv')

        df = df.drop(columns=['city'])

        # One-hot encode categorical columns
        categorical_cols = ['gender', 'education', 'occupation', 'most bought item']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Min-max normalize numerical columns
        def min_max_normalize(column):
            return (column - column.min()) / (column.max() - column.min())

        df['age'] = min_max_normalize(df['age'])
        df['income'] = min_max_normalize(df['income'])
        df['purchase_amount'] = min_max_normalize(df['purchase_amount'])
        df['children'] = min_max_normalize(df['children'])

        # One-hot encode the labels (multi-label space-separated)
        # Split the 'labels' column into a list of labels
        df['labels'] = df['labels'].apply(lambda x: x.split(' '))

        # Get all unique labels
        unique_labels = sorted(set(label for sublist in df['labels'] for label in sublist))

        # Create one-hot columns for each label
        for label in unique_labels:
            df[label] = df['labels'].apply(lambda x: 1 if label in x else 0)

        # Drop the original 'labels' column
        df = df.drop(columns=['labels'])

        # convert boolean columns to integer
        df.replace({True: 1, False: 0}, inplace=True)

        # Divide the data into train,validation and test data 
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        # splitting the data into train, validation and test set
        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        train_set = df[:train_len]
        valid_set = df[train_len:train_len+valid_len]
        test_set = df[train_len+valid_len:]

        # Save the data
        train_set.to_csv('../../data/interim/3/advertisement_train.csv', index=False)
        valid_set.to_csv('../../data/interim/3/advertisement_valid.csv', index=False)
        test_set.to_csv('../../data/interim/3/advertisement_test.csv', index=False)

        # Load the data
        train_set = pd.read_csv('../../data/interim/3/advertisement_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/advertisement_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/advertisement_test.csv')

        X_train = train_set.iloc[:, :-8].values
        y_train = train_set.iloc[:, -8:].values

        X_valid = valid_set.iloc[:, :-8].values
        y_valid = valid_set.iloc[:, -8:].values

        X_test = test_set.iloc[:, :-8].values
        y_test = test_set.iloc[:, -8:].values

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)


        active_func = ['relu','relu','sigmoid']
        neurons = [64, 32]
        model = mlp.MLP(2, neurons, active_func)
        model.fit(X_train, y_train, 'batch', 'binary_crossentropy', 0.1, 1, 32, X_valid, y_valid)
        model.check_gradients(X_train, y_train, epsilon=1e-7)

        Hamming_loss,loss_train_epochs, accuracy_train_epochs, recall_train_epochs, precision_train_epochs, loss_valid_epochs, accuracy_valid_epochs, recall_valid_epochs, precision_valid_epochs = model.fit(X_train, y_train, 'batch', 'binary_crossentropy', 0.1, 500, 32, X_valid, y_valid)

        print("Hamming Loss: ", np.mean(Hamming_loss))
        print("Accuracy: ", np.mean(accuracy_valid_epochs))
        print("Precision: ", np.mean(precision_valid_epochs))
        print("Recall: ", np.mean(recall_valid_epochs))
        print("Loss: ", np.mean(loss_valid_epochs))

    print('Enter yes/no. If you want to do hyperparameter tuning for multi-label classification')
    command = input()
    if command == 'yes':
        # Hyperparameter tuning

        best_acc = 0

        def train_model():
            global best_acc
            wandb.init()
            hyperparameters = wandb.config
            
            print(wandb.config)

            activations = [hyperparameters['activations']] * (len(hyperparameters['hidden_layer_config']))
            activations.append('sigmoid')
            
            print(len(hyperparameters["hidden_layer_config"]), hyperparameters["hidden_layer_config"],activations)
            model = mlp.MLP(len(hyperparameters["hidden_layer_config"]), hyperparameters["hidden_layer_config"],activations)
            
            Hamming_loss,loss_train_epochs, accuracy_train_epochs, recall_train_epochs, precision_train_epochs, loss_valid_epochs, accuracy_valid_epochs, recall_valid_epochs, precision_valid_epochs = model.fit(X_train, y_train, hyperparameters['optimizer'],'binary_crossentropy', hyperparameters['lr'], hyperparameters['epochs'], 32, X_valid, y_valid)        
            
            for i in range(0, hyperparameters['epochs']):
                wandb.log({"Hamming_loss" : Hamming_loss[i],
                            "train_loss": loss_train_epochs[i],
                            "val_loss": loss_valid_epochs[i],
                        "train_accuracy": accuracy_train_epochs[i],
                        "val_accuracy": accuracy_valid_epochs[i],
                            "train_precision": precision_train_epochs[i],
                            "val_precision": precision_valid_epochs[i],
                            "train_recall": recall_train_epochs[i],
                            "val_recall": recall_valid_epochs[i]})


        sweep_config = {
            "method": "grid",
            "metric": {
                "goal": "maximize",
                "name": "val_accuracy"
            },
            "parameters": {
                "hidden_layer_config": {
                    "values": [[32], [32, 32], [64], [64, 64], [64, 32]]
                },
                "activations": {
                    "values": ["relu", "sigmoid", "tanh"]
                },
                "epochs": {
                    "values": [100, 500, 300]
                },
                "lr": {
                    "values": [0.001, 0.01, 0.1]
                },
                "optimizer": {
                    "values": ["stochastic", "batch", "mini-batch"]
                }   
            }
            
        }


        sweep_id = wandb.sweep(sweep_config, project="smai_a3")
        wandb.agent(sweep_id, function=train_model)

        api = wandb.Api()
        sweep_id = f"keshava-kishora-iiit-hyderabad/smai_a3/{sweep_id}"

        sweep = api.sweep(sweep_id)
        runs = sweep.runs

        best_run = sorted(runs, key=lambda run: run.summary.get('val_accuracy'), reverse=True)[0]

        best_hyperparameters = best_run.config
        print("Best Hyperparameters:", best_hyperparameters)

    print('Enter yes/no. If you want to see what is class supported by classifier')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/WineQT_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/WineQT_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/WineQT_test.csv')

        X_train = train_set.iloc[:, :-6].values
        y_train = train_set.iloc[:, -6:].values

        X_valid = valid_set.iloc[:, :-6].values
        y_valid = valid_set.iloc[:, -6:].values

        X_test = test_set.iloc[:, :-6].values
        y_test = test_set.iloc[:, -6:].values

        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        # Best model on test set
        best_lr = 0.1
        best_epochs = 500
        best_hidden_layer_config = [64]
        best_optimizer = 'mini-batch'
        activations = ['relu', 'softmax']

        model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations)

        train_loss, train_acc, train_recall, train_precision, train_f1, valid_loss, valid_acc, valid_recall, valid_precision, valid_f1 = model.fit(X_train, y_train, best_optimizer,'cross_entropy', best_lr, best_epochs, 32, X_valid, y_valid)

        y_pred = model.predict(X_test)
        y_pred_class = np.argmax(y_pred, axis=1)
        y_test_class = np.argmax(y_test, axis=1)

        print(y_pred_class)

        class_bias = {0: 0, 1: 0, 2: 0, 3:0, 4: 0, 5:0}

        for i in range(len(y_pred_class)):
            if y_pred_class[i] == y_test_class[i]:
                class_bias[y_pred_class[i]] += 1

        print(class_bias)


if choice == 2:
    print('Enter yes/no. If you want to do preprocessing for Housing dataset')
    command = input()
    if command == 'yes':

        df = pd.read_csv('../../data/external/HousingData.csv')

        # Describe the data
        for column in df.columns:
            print('Description of column:', column)
            print(df[column].describe())

        # Distribution of various columns
        for x in df.columns:
            data = df[x]
            plt.hist(data, bins=10)
            plt.title(x)
            plt.savefig('figures/histogram_'+x+'.png')
            plt.close()

        # Rows with null values are 112. So, we cannot rows with null values
        df.isnull().values.any()
        df.isnull().any(axis=1)

        # Colums with null values are 6(all are float type). So, we can replace the null values with the mean of the column
        df.isnull().sum()

        # Reaplce the null values with the mean of the column
        df.fillna(df.mean(), inplace=True)

        # Normalize and standardize the data except MEDV column without using sklearn 
        for x in df.columns:
            if x != 'MEDV':
                df[x] = (df[x] - df[x].mean())/df[x].std()


        # Divide the data into train,validation and test data 
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        # splitting the data into train, validation and test set
        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        train_set = df[:train_len]
        valid_set = df[train_len:train_len+valid_len]
        test_set = df[train_len+valid_len:]

        # Save the data
        train_set.to_csv('../../data/interim/3/HousingData_train.csv', index=False)
        valid_set.to_csv('../../data/interim/3/HousingData_valid.csv', index=False)
        test_set.to_csv('../../data/interim/3/HousingData_test.csv', index=False)

    print('Enter yes/no. If you want to see whether the MLP is working fine for regressioin')
    command = input()
    if command == 'yes':
        # Load the data
        train_set = pd.read_csv('../../data/interim/3/HousingData_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/HousingData_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/HousingData_test.csv')

        X_train = train_set.iloc[:, :-1].values
        y_train = train_set.iloc[:, -1].values
        y_train = y_train.reshape(-1, 1)

        X_valid = valid_set.iloc[:, :-1].values
        y_valid = valid_set.iloc[:, -1].values
        y_valid = y_valid.reshape(-1, 1)

        X_test = test_set.iloc[:, :-1].values
        y_test = test_set.iloc[:, -1].values
        y_test = y_test.reshape(-1, 1)

        # Train the model
        active_func = ['sigmoid','sigmoid','linear']
        neurons = [64, 32]
        model = mlp.MLP(2, neurons, active_func)
        model.fit(X_train, y_train, 'batch', 'MSE', 0.001, 1, None, X_valid, y_valid)
        model.check_gradients(X_train, y_train, epsilon=1e-7)

        model.fit(X_train, y_train, 'batch', 'MSE', 0.001, 300, None, X_valid, y_valid)

        # Predict the values and calculate the mean squared error
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(-1, 1)

        mse = np.mean((y_pred - y_test)**2)
        print('Mean Squared Error:', mse)

    print('Enter yes/no. If you want to do hyperparameter tuning for regression')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/HousingData_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/HousingData_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/HousingData_test.csv')

        X_train = train_set.iloc[:, :-1].values
        y_train = train_set.iloc[:, -1].values
        y_train = y_train.reshape(-1, 1)

        X_valid = valid_set.iloc[:, :-1].values
        y_valid = valid_set.iloc[:, -1].values
        y_valid = y_valid.reshape(-1, 1)

        X_test = test_set.iloc[:, :-1].values
        y_test = test_set.iloc[:, -1].values
        y_test = y_test.reshape(-1, 1)

        best_acc = 0

        def train_model():
            global best_acc
            wandb.init()
            hyperparameters = wandb.config
            
            print(wandb.config)

            activations = [hyperparameters['activations']] * (len(hyperparameters['hidden_layer_config']))
            activations.append('linear')
            
            model = mlp.MLP(len(hyperparameters["hidden_layer_config"]), hyperparameters["hidden_layer_config"],activations)
            
            MSE_train_epoch, RMSE_train_epoch, R_squared_train_epoch, MSE_valid_epoch, RMSE_valid_epoch, R_squared_valid_epoch = model.fit(X_train, y_train, hyperparameters['optimizer'],'MSE', hyperparameters['lr'], hyperparameters['epochs'], 32, X_valid, y_valid)
            
            for i in range(0, hyperparameters['epochs']):
                wandb.log({"MSE_train": MSE_train_epoch[i], "RMSE_train": RMSE_train_epoch[i], "R_squared_train": R_squared_train_epoch[i], "MSE_valid": MSE_valid_epoch[i], "RMSE_valid": RMSE_valid_epoch[i], "R_squared_valid": R_squared_valid_epoch[i]})


        sweep_config = {
            "method": "grid",
            "metric": {
                "goal": "minimize",
                "name": "MSE_valid"
            },
            "parameters": {
                "hidden_layer_config": {
                    "values": [[32], [32, 32], [64], [64, 64], [64, 32]]
                },
                "activations": {
                    "values": ["relu", "sigmoid", "tanh"]
                },
                "epochs": {
                    "values": [100, 300, 500]
                },
                "lr": {
                    "values": [0.001, 0.01, 0.1]
                },
                "optimizer": {
                    "values": ["stochastic", "batch", "mini-batch"]
                }   
            }
            
        }


        sweep_id = wandb.sweep(sweep_config, project="smai_a3")
        wandb.agent(sweep_id, function=train_model)

        api = wandb.Api()
        sweep_id = f"keshava-kishora-iiit-hyderabad/smai_a3/{sweep_id}"

        sweep = api.sweep(sweep_id)
        runs = sweep.runs

        best_run = sorted(runs, key=lambda run: run.summary.get('MSE_valid'), reverse=True)[0]

        best_hyperparameters = best_run.config
        print("Best Hyperparameters:", best_hyperparameters)

    print('Enter yes/no. If you want to do testing on the best model')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/HousingData_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/HousingData_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/HousingData_test.csv')

        X_train = train_set.iloc[:, :-1].values
        y_train = train_set.iloc[:, -1].values
        y_train = y_train.reshape(-1, 1)

        X_valid = valid_set.iloc[:, :-1].values
        y_valid = valid_set.iloc[:, -1].values
        y_valid = y_valid.reshape(-1, 1)

        X_test = test_set.iloc[:, :-1].values
        y_test = test_set.iloc[:, -1].values
        y_test = y_test.reshape(-1, 1)

        # Best model on test set
        best_lr = 0.01
        best_epochs = 500
        best_hidden_layer_config = [64, 32]
        best_optimizer = 'batch'
        activations = ['tanh', 'tanh', 'linear']

        model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations)
        MSE_train_epoch, RMSE_train_epoch, R_squared_train_epoch, MSE_valid_epoch, RMSE_valid_epoch, R_squared_valid_epoch = model.fit(X_train, y_train, best_optimizer,'MSE', best_lr, best_epochs, 32, X_valid, y_valid)

        y_pred = model.predict(X_test)
        metrics = rm.RegressionMeasures()

        print("Mean Squared Error:", metrics.MSE(y_test, y_pred))
        print("Root Mean Squared Error:", metrics.RMSE(y_test, y_pred))
        print("R Squared:", metrics.R_squared(y_test, y_pred))

    print('Enter yes/no. If you want to do preprocessing for diabetes dataset')
    command = input()
    if command == 'yes':
        df = pd.read_csv('../../data/external/diabetes.csv')

        # Normalize and standardize the data except Outcome column without using sklearn
        for x in df.columns:
            if x != 'Outcome':
                df[x] = (df[x] - df[x].mean())/df[x].std()

        # Divide the data into train,validation and test data
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)

        # splitting the data into train, validation and test set
        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        train_set = df[:train_len]
        valid_set = df[train_len:train_len+valid_len]
        test_set = df[train_len+valid_len:]

        # Save the data
        train_set.to_csv('../../data/interim/3/diabetes_train.csv', index=False)
        valid_set.to_csv('../../data/interim/3/diabetes_valid.csv', index=False)
        test_set.to_csv('../../data/interim/3/diabetes_test.csv', index=False)

    print('Enter yes/no.If you want to see the BCE loss vs MSE loss for diabetes dataset with logistic regression')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/diabetes_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/diabetes_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/diabetes_test.csv')

        X_train = train_set.iloc[:, :-1].values
        y_train = train_set.iloc[:, -1].values
        y_train = y_train.reshape(-1, 1)

        X_valid = valid_set.iloc[:, :-1].values
        y_valid = valid_set.iloc[:, -1].values
        y_valid = y_valid.reshape(-1, 1)

        X_test = test_set.iloc[:, :-1].values
        y_test = test_set.iloc[:, -1].values
        y_test = y_test.reshape(-1, 1)

        print(y_train.shape)

        # Train the model
        active_func = ['sigmoid']
        neurons = []
        model = mlp.MLP(0, neurons, active_func)
        model.fit(X_train, y_train, 'batch', 'MSE', 0.01, 1, None, X_valid, y_valid)
        model.check_gradients(X_train, y_train, epsilon=1e-7)

        MSE_train_epoch, RMSE_train_epoch, R_squared_train_epoch, MSE_valid_epoch, RMSE_valid_epoch, R_squared_valid_epoch = model.fit(X_train, y_train, 'batch', 'MSE', 0.01, 1000, None, X_valid, y_valid)

        active_func = ['sigmoid']
        neurons = []
        model = mlp.MLP(0, neurons, active_func)
        model.fit(X_train, y_train, 'batch', 'binary_crossentropy', 0.01, 1, None, X_valid, y_valid)
        model.check_gradients(X_train, y_train, epsilon=1e-7)

        Hamming_loss,loss_train_epochs, accuracy_train_epochs, recall_train_epochs, precision_train_epochs, loss_valid_epochs, accuracy_valid_epochs, recall_valid_epochs, precision_valid_epochs = model.fit(X_train, y_train, 'batch', 'binary_crossentropy', 0.01, 1000, 32, X_valid, y_valid)


        # Plot the graphs
        import matplotlib.pyplot as plt

        # draw two subplots between mse loss and bce loss in the same figure

        fig, axs = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle('MSE vs BCE')
        axs[0].plot(MSE_valid_epoch, label='MSE')
        axs[0].set_title('MSE')
        axs[1].plot(loss_valid_epochs, label='BCE')
        axs[1].set_title('BCE')
        plt.savefig('figures/MSE_vs_BCE.png')
        plt.show()
        plt.close()

    print('Enter yes/no. If you want see MSE loss for each datapoint in advertisement dataset')
    command = input()
    if command == 'yes':
        train_set = pd.read_csv('../../data/interim/3/HousingData_train.csv')
        valid_set = pd.read_csv('../../data/interim/3/HousingData_valid.csv')
        test_set = pd.read_csv('../../data/interim/3/HousingData_test.csv')

        X_train = train_set.iloc[:, :-1].values
        y_train = train_set.iloc[:, -1].values
        y_train = y_train.reshape(-1, 1)

        X_valid = valid_set.iloc[:, :-1].values
        y_valid = valid_set.iloc[:, -1].values
        y_valid = y_valid.reshape(-1, 1)

        X_test = test_set.iloc[:, :-1].values
        y_test = test_set.iloc[:, -1].values
        y_test = y_test.reshape(-1, 1)

        # Best model on test set
        best_lr = 0.01
        best_epochs = 100
        best_hidden_layer_config = [64, 32]
        best_optimizer = 'batch'
        activations = ['tanh', 'tanh', 'linear']

        model = mlp.MLP(len(best_hidden_layer_config), best_hidden_layer_config, activations)
        MSE_train_epoch, RMSE_train_epoch, R_squared_train_epoch, MSE_valid_epoch, RMSE_valid_epoch, R_squared_valid_epoch = model.fit(X_train, y_train, best_optimizer,'MSE', best_lr, best_epochs, 32, X_valid, y_valid)

        y_pred = model.predict(X_test)

        loss_batch = []
        for i in range(len(y_test)):
            loss_batch.append(np.mean((y_pred[i] - y_test[i])**2))

        plt.plot(range(1, len(y_test)+1), loss_batch)
        plt.title('MSE loss for each datapoint')
        plt.xlabel('Datapoint')
        plt.ylabel('MSE loss')
        plt.savefig('figures/MSE_loss_each_datapoint.png')
        plt.show()
        plt.close()

        # print the indices with high MSE and Low MSE
        high_mse = np.argsort(loss_batch)[-10:]
        low_mse = np.argsort(loss_batch)[:10]

        print('Indices with high MSE:', high_mse)
        print('Indices with low MSE:', low_mse)


if choice == 3:
    print('Enter yes/no. If you want to see the autoencoder working and reduced data from it')
    command = input()
    if command == 'yes':
        df = pd.read_csv('../../data/interim/1/spotify_normalized.csv')

        # jumble the data
        df = df.sample(frac=1, random_state=0, replace=False)

        X = df.drop(columns=['track_genre'])
        Y = df['track_genre']

        X = X.to_numpy()
        Y = Y.to_numpy()

        # split the data into training and validation

        neurons = [16]
        active_functions = ['tanh', 'linear']

        autoencoder = AE.Autoencoders(1, neurons, active_functions)
        autoencoder.fit(X, X, 'batch', 'MSE', 0.01, 100, 32)
        X_reduced = autoencoder.get_latent(X)

    print('Enter yes/no. If you want to do clustering on the reduced data using KNN')
    command = input()
    if command == 'yes':
        df = pd.read_csv('../../data/interim/1/spotify_normalized.csv')

        # jumble the data
        df = df.sample(frac=1, random_state=0, replace=False)

        X = df.drop(columns=['track_genre'])
        Y = df['track_genre']

        X = X.to_numpy()
        Y = Y.to_numpy()

        neurons = [16]
        active_functions = ['tanh', 'linear']

        autoencoder = AE.Autoencoders(1, neurons, active_functions)
        autoencoder.fit(X, X, 'batch', 'MSE', 0.01, 100, 32)
        X_reduced = autoencoder.get_latent(X)

        print(X_reduced.shape)
        print(Y.shape)

        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        x_train = X_reduced[:train_len]
        y_train = Y[:train_len]
        y_train = y_train

        x_valid = X_reduced[train_len:train_len+valid_len]
        y_valid = Y[train_len:train_len+valid_len]
        y_valid = y_valid

        x_test = X_reduced[train_len+valid_len:]
        y_test = Y[train_len+valid_len:]
        y_test = y_test

        Knn_classifier = knn.KNN_optim(16, 'manhattan')
        Knn_classifier.train_model(x_train, y_train)

        y_pred = Knn_classifier.predict(x_test)

        performance = cm.Measures(y_test, y_pred)

        print(f'accuracy : {performance.accuracy()}')
        print(f'precision_macro : {performance.precision_macro()}')
        print(f'recall_macro : {performance.recall_macro()}')
        print(f'f1_macro : {performance.f1_score_macro()}')

        print(f'precision_micro : {performance.precision_micro()}')
        print(f'recall_micro : {performance.recall_micro()}')
        print(f'f1_micro : {performance.f1_score_micro()}')

    print('Enter yes/no. If you want to do clustering on the reduced data using MLP')
    command = input()
    if command == 'yes':
        df = pd.read_csv('../../data/interim/1/spotify_normalized.csv')

        # jumble the data
        df = df.sample(frac=1, random_state=0, replace=False)

        X = df.drop(columns=['track_genre'])
        Y = df['track_genre']

        # one hot encoding for Y
        Y = pd.get_dummies(Y)

        X = X.to_numpy()
        Y = Y.to_numpy()

        # Finf reduced data from the autoencoder
        neurons = [16]
        active_functions = ['tanh', 'linear']

        autoencoder = AE.Autoencoders(1, neurons, active_functions)
        autoencoder.fit(X, X, 'batch', 'MSE', 0.01, 100, 32)
        X_reduced = autoencoder.get_latent(X)

        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        x_train = X_reduced[:train_len]
        y_train = Y[:train_len]
        y_train = y_train

        x_valid = X_reduced[train_len:train_len+valid_len]
        y_valid = Y[train_len:train_len+valid_len]
        y_valid = y_valid

        x_test = X_reduced[train_len+valid_len:]
        y_test = Y[train_len+valid_len:]
        y_test = y_test

        # Train MLP for classification on the reduced data
        active_func = ['relu','softmax']
        neurons = [120]
        model = mlp.MLP(1, neurons, active_func)
        model.fit(x_train, y_train, 'batch', 'cross_entropy', 0.1, 1, 32, x_valid, y_valid)
        model.fit(x_train, y_train, 'batch', 'cross_entropy', 0.1, 100, 32, x_valid, y_valid)

        y_pred = model.predict(x_test)

        y_pred_class = np.argmax(y_pred, axis=1)
        y_test_class = np.argmax(y_test, axis=1)

        performance = cm.Measures(y_test_class, y_pred_class)

        print(f'accuracy : {performance.accuracy()}')
        print(f'precision_macro : {performance.precision_macro()}')
        print(f'recall_macro : {performance.recall_macro()}')
        print(f'f1_macro : {performance.f1_score_macro()}')

        print(f'precision_micro : {performance.precision_micro()}')
        print(f'recall_micro : {performance.recall_micro()}')
        print(f'f1_micro : {performance.f1_score_micro()}')


