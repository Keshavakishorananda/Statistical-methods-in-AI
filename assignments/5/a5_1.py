import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../')

from models.kde.kde import KDE
from models.gmm.gmm import GMM

np.random.seed(42)

print("This is the solution for question 1 of assignment 5")

print("The following tasks can be performed: ")
print("1. Generate synthetic data")
print("2. Fit a KDE model to the data")
print("3. Comparing the KDE and GMM")

choice = int(input("Enter the number of the task you want to perform: "))

if choice == 1:
    # Larger circle
    n_points_large = 3000
    radius_large = 3

    angles_large = np.random.uniform(0, 2 * np.pi, n_points_large)
    radii_large = radius_large * np.sqrt(np.random.uniform(0, 1, n_points_large))
    x_large = radii_large * np.cos(angles_large)
    y_large = radii_large * np.sin(angles_large)

    # Smaller circle
    n_points_small = 500
    radius_small = 0.5

    angles_small = np.random.uniform(0, 2 * np.pi, n_points_small)
    radii_small = radius_small * np.sqrt(np.random.uniform(0, 1, n_points_small))
    x_small = radii_small * np.cos(angles_small) + 1
    y_small = radii_small * np.sin(angles_small) + 1

    x = np.concatenate([x_large, x_small])
    y = np.concatenate([y_large, y_small])

    data = np.stack([x, y], axis=1)
    np.savetxt("../../data/interim/5/Synthetic_data.csv", data, delimiter=",")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, alpha=0.6)
    plt.title("Original Data")
    plt.axis("equal")
    # save the plot
    plt.savefig("figures/Circular_data.png")
    plt.show()
    
if choice == 2:
    df = pd.read_csv('../../data/interim/5/Synthetic_data.csv', header=None)
    df.columns = ['x', 'y']
    df.head()

    data = df.values
    data = np.array(data, dtype='float32')

    kde = KDE('Gaussian', bandwidth=0.2)
    kde.fit(data)
    kde.visualize()

    kde = KDE('triangular', bandwidth=0.2)
    kde.fit(data)
    kde.visualize()

    kde = KDE('box', bandwidth=0.2)
    kde.fit(data)
    kde.visualize()

if choice == 3:
    df = pd.read_csv('../../data/interim/5/Synthetic_data.csv', header=None)
    df.columns = ['x', 'y']
    df.head()

    data = df.values
    data = np.array(data, dtype='float32')

    # For GMM
    for K in range(1, 4):
        gmm = GMM(K = K)
        gmm.fit(data)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = gmm.getprob_density(np.array([X.flatten(), Y.flatten()]).T)
        Z = Z.reshape(X.shape)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("GMM for, K = " + str(K))
        # save the plot
        plt.savefig("figures/GMM_K_" + str(K) + ".png")
        plt.close()


    # For KDE
    kde = KDE('Gaussian', bandwidth=0.2)
    kde.fit(data)
    kde.visualize()