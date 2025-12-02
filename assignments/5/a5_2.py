import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
import random

print("This is the solution for question 2 of assignment 5")

print(" we will do following steps to solve the problem")

print("1. Load the data and visualize the mfcc features of the data")
print("2. Training the HMM model")
print("3. Testing the model on real-time data for generalization")

task = input("Enter the task you want to perform: ")

def extract_features(data_dir, max_length=100):
    features = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for i in range(10):
        for filename in os.listdir(data_dir):
            if filename.startswith(str(i)):
                y, sr = librosa.load(data_dir + filename)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features[i].append(mfcc.T)

    return features

# Dividing the data into training and testing sets without using sklearn
def split_data(features):
    train_features = {i: [] for i in range(10)}
    test_features = {i: [] for i in range(10)}
    
    for i in range(10):
        random.shuffle(features[i])
        split_idx = int(len(features[i]) * 0.8)
        train_features[i] = features[i][:split_idx]
        test_features[i] = features[i][split_idx:]
    
    return train_features, test_features


# Visualiznig one of the MFCC features of one of the recordings with a heatmap
def visualize_mfcc(mfcc):
    if len(mfcc.shape) == 1:
        mfcc = mfcc.reshape(-1, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(mfcc, aspect='auto', cmap='coolwarm', origin='lower')
    plt.colorbar(label='MFCC Coefficient Value')
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.title('MFCC Heatmap')
    plt.savefig("figures/MFCC_Heatmap.png")
    plt.close()


# Evaluating the model on the test data
def test_model(models, test_data):
    correct = 0
    total = 0
    for i in range(10):
        for data in test_data[i]:
            scores = [model.score(data) for model in models.values()]
            prediction = np.argmax(scores)
            if prediction == i:
                correct += 1
            total += 1
    return correct / total


if task == "1":
    # Extracting the MFCC features and visualizing them
    data = extract_features('../../data/external/recordings/')
    train_data, test_data = split_data(data)

    visualize_mfcc(train_data[0][0].T)
    print('Image saved in figures folder')

elif task == "2":
    data = extract_features('../../data/external/recordings/')
    train_data, test_data = split_data(data)

    # Train the model
    models = {0 : None, 1 : None, 2 : None, 3 : None, 4 : None, 5 : None, 6 : None, 7 : None, 8 : None, 9 : None}
    for i in range(10):
        X = np.concatenate(train_data[i])
        lengths = [len(x) for x in train_data[i]]
        model = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000)
        models[i] = model.fit(X, lengths)

    accuracy = test_model(models, test_data)

    print('Accuracy: ', accuracy)

elif task == "3":
    real_data =extract_features('../../data/external/keshava_audio/')

    models = {0 : None, 1 : None, 2 : None, 3 : None, 4 : None, 5 : None, 6 : None, 7 : None, 8 : None, 9 : None}

    data = extract_features('../../data/external/recordings/')
    train_data, test_data = split_data(data)

    for i in range(10):
        X = np.concatenate(train_data[i])
        lengths = [len(x) for x in train_data[i]]
        model = GaussianHMM(n_components=10, covariance_type="diag", n_iter=1000)
        models[i] = model.fit(X, lengths)

    for i in range(10):
        for data in real_data[i]:
            scores = [model.score(data) for model in models.values()]
            prediction = np.argmax(scores)
            print(f"Predicted: {prediction}, Actual: {i}")
    
    accuracy = test_model(models, real_data)
    print('Accuracy: ', accuracy)
