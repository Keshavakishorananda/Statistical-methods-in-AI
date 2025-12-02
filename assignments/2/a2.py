import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster 
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
import time
import sys
sys.path.append('../../')

import models.k_means.k_means as K_means
import models.gmm.gmm as Gmm
import models.pca.pca as pca
import models.knn.knn as knn
import performance_measures.classification_measures as cm

def split_test_valid_train_sets(df):

    df = df.sample(frac=1, random_state=18).reset_index(drop=True)

    # splitting the data into train, validation and test set
    train_len = int(0.8*len(df))
    valid_len = int(0.1*len(df))
    test_len = len(df) - train_len - valid_len

    train_set = df[:train_len]
    valid_set = df[train_len:train_len+valid_len]
    test_set = df[train_len+valid_len:]


    x_train = train_set.drop('track_genre', axis=1)
    y_train = train_set['track_genre']

    x_valid = valid_set.drop('track_genre', axis=1)
    y_valid = valid_set['track_genre']

    x_test = test_set.drop('track_genre', axis=1)
    y_test = test_set['track_genre']


    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

df = pd.read_feather('../../data/external/word-embeddings.feather')

X_train = df['vit']
Words = df['words']

X_train = np.array([np.array(x) for x in X_train])

print('If you want to do clustering of data using K-means?Enter 1')
print('If you want to do clustering of data using GMMs?Enter 2')
print('If you want to do dimesionality reduction and visualization of data using PCA?Enter 3')
print('IF you want to see PCA+clustering? Enter 4')
print('If you want to do cluster Analysis? Enter 5')
print('If you want to do Hierarchical clustering? Enter 6')
print('If you want to do Nearest Neighbour clustering? Enter 7')
print('If you want to exit? Enter 8')

print('Enter your choice:')

question = int(input())
print('You have entered:', question)

k_kmeans1 = 7
k_gmm1 = 2
k2 = 3
k_kmeans3 = 4
k_gmm3 = 5
k_kmeans = k_kmeans3
k_gmm = k_gmm3
optimal_dimensions = 136

if question == 1:
    # Finding the optimal number of clusters using elbow method
    Elbow_Wcss = []
    k = range(1, 40)
    for K in k:
        model = K_means.Kmeans(K = K)
        model.fit(X_train)
        Elbow_Wcss.append(model.getCost())

    plt.plot(k, Elbow_Wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('figures/K_means_elbow.png')
    plt.close()

    # So, finally the optimal number of clusters k_kmeans1 = 10
    k_kmeans1 = 7

    # Training the model with optimal number of clusters
    model = K_means.Kmeans(K = k_kmeans1)
    model.fit(X_train)
    clusters_kmeans = model.predict(X_train)
    print('The clusters using kmeans are:', clusters_kmeans)

if question == 2:
    # Finding optimal number of clusters using AIC and BIC values
    AIC = []
    BIC = []
    for i in range(1,11):
        models = Gmm.GMM(i)
        models.fit(X_train)
        parameters = i*(X_train.shape[1] + (X_train.shape[1])*(X_train.shape[1] + 1)/2 + 1)
        aic = 2*parameters - 2*models.likelihood
        AIC.append(aic)

        bic = parameters*np.log(X_train.shape[0]) - 2*models.likelihood
        BIC.append(bic)


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1,11), AIC, label='AIC', marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('AIC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1,11), BIC, label='BIC', marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC')
    plt.legend()

    plt.savefig('figures/GMM_AIC_BIC.png')
    plt.close()

    # So, finally the optimal number of clusters k_gmm1 = 3
    k_gmm1 = 2

    # Training the model with optimal number of clusters
    model = Gmm.GMM(k_gmm1)
    model.fit(X_train)
    membership = model.getMembership(X_train)
    membership = np.argmax(membership, axis=1)
    print('The clusters using GMMs are:', membership)

if question == 3:
    # For 2 components
    model2 = pca.PCA(N_components=2)
    model2.fit(X_train)
    X_2 = model2.transform(X_train)

    plt.scatter(np.real(X_2[:, 0]), np.real(X_2[:, 1]))
    plt.title('2D plot')
    for i, word in enumerate(Words):
        plt.annotate(word, (X_2[i, 0], X_2[i, 1]), textcoords='offset points', xytext=(5, 5), ha='center')
    plt.savefig('figures/Word_embeddings_labels_2D_plot.png')
    plt.close()

    plt.scatter(np.real(X_2[:, 0]), np.real(X_2[:, 1]))
    plt.title('2D plot')
    plt.savefig('figures/Word_embeddings_2D_plot.png')
    plt.close()


    # For 3 components
    model3 = pca.PCA(N_components=3)
    model3.fit(X_train)
    X_3 = model3.transform(X_train)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.real(X_3[:, 0]), np.real(X_3[:, 1]), np.real(X_3[:, 2]))
    plt.title('3D plot')
    plt.savefig('figures/Word_embeddings_3D_plot.png')
    plt.close()

    # Verify the PCA implementation of 2D, 3D using checkPCA method : pending
    print('The checkPCA for 2D is:', model2.checkPCA(X_train, X_2))
    print('The checkPCA for 3D is:', model3.checkPCA(X_train, X_3))

    # identify what each of the new axes that are obtained from PCA represent? : pending

    #  Examine the 2D and 3D plots to identify any visible patterns or clusters. Without performing any clustering, estimate the approximate number of clusters based on the plots : pending
    # k2 = estimated number of clusters from 2D plot. So, k2 = 3

if question == 4:
    # Reduced data using PCA
    pca_model = pca.PCA(N_components=2)
    pca_model.fit(X_train)

    eigenvalues = pca_model.sort_eigen

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(eigenvalues) + 1), np.real(eigenvalues), marker='o', linestyle='--', color='b')
    threshold = 1
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.savefig('figures/screeplot_word_embeddings.png')
    plt.close()


    explained_variance = np.real(eigenvalues)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cummulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cummulative_variance) + 1), cummulative_variance, linestyle='--', color='b')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.title('Cummulative Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Cummulative Variance')
    plt.savefig('figures/cummulative_variance_word_embeddings.png')
    plt.close()


    optimal_dimensions = 136
    pca_model = pca.PCA(N_components=optimal_dimensions)
    pca_model.fit(X_train)
    X_reduced = pca_model.transform(X_train)


    #Part-1: kmeans clustering on 2D data
    k2 = 3
    pca_model = pca.PCA(N_components=2)
    pca_model.fit(X_train)
    X_2 = pca_model.transform(X_train)

    k_means_model = K_means.Kmeans(K=k2)
    k_means_model.fit(X_2)
    clusters_kmeans = k_means_model.predict(X_2)
    print('The clusters on 2D data using Kmeans are:', clusters_kmeans)

    plt.scatter(X_2[:, 0], X_2[:, 1], c=clusters_kmeans)
    plt.title('Kmeans clustering on 2D data')
    plt.savefig('figures/Kmeans_clustering_2D.png')
    plt.close()

    #PCA+kmeans clustering
    # calculate the optimal number of clusters for reduced data using elbow method for kmeans clustering
    Elbow_Wcss = []
    k = range(1, 40)
    for K in k:
        model = K_means.Kmeans(K = K)
        model.fit(X_reduced)
        Elbow_Wcss.append(model.getCost())

    plt.plot(k, Elbow_Wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('figures/K_means_elbow_reduceddata.png')
    plt.close()

    # Name the optimal number of clusters as k_kmeans3
    k_kmeans3 = 4

    # perform kmeans clustering on reduced data using the optimal number of clusters k_kmeans3
    k_means_model = K_means.Kmeans(K=k_kmeans3)
    k_means_model.fit(X_reduced)
    clusters_kmeans = k_means_model.predict(X_reduced)
    print('The clusters on reduced data using Kmeanns are:', clusters_kmeans)   


    #Part-2 : GMM clustering on 2D data
    gmm_model = Gmm.GMM(K=k2)
    gmm_model.fit(X_2)
    membership = gmm_model.getMembership(X_2)
    membership = np.argmax(membership, axis=1)
    print('The clusters on 2D data using GMMs are:', membership)

    plt.scatter(X_2[:, 0], X_2[:, 1], c=membership)
    plt.title('GMM clustering on 2D data')
    plt.savefig('figures/GMM_clustering_2D.png')
    plt.close()

    # PCA+GMM clustering
    # calculate the optimal number of clusters for reduced data using AIC and BIC values for GMM clustering
    AIC = []
    BIC = []
    for i in range(1,11):
        models = Gmm.GMM(i)
        models.fit(X_reduced)
        parameters = i*(X_reduced.shape[1] + (X_reduced.shape[1])*(X_reduced.shape[1] + 1)/2 + 1)
        aic = 2*parameters - 2*models.likelihood
        AIC.append(aic)

        bic = parameters*np.log(X_train.shape[0]) - 2*models.likelihood
        BIC.append(bic)

    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1,11), AIC, label='AIC', marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('AIC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1,11), BIC, label='BIC', marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC')
    plt.legend()

    plt.savefig('figures/GMM_AIC_BIC_reduceddata.png')
    plt.close()

    # Name the optimal number of clusters as k_gmm3
    k_gmm3 = 5

    # perform GMM clustering on reduced data using the optimal number of clusters k_gmm3
    gmm_model = Gmm.GMM(K = k_gmm3)
    gmm_model.fit(X_reduced)
    membership = gmm_model.getMembership(X_reduced)
    membership = np.argmax(membership, axis=1)
    print('The clusters on reduced data using GMMs are:', membership)


if question == 5:
    print('If you want to do cluster analysis using K-means? Enter yes')
    if input() == 'yes':
        # K-means cluseting Ananlysis
        model1 = K_means.Kmeans(K=k_kmeans1)
        model1.fit(X_train)

        model2 = K_means.Kmeans(K=k2)
        model2.fit(X_train)


        model3 = K_means.Kmeans(K=k_kmeans3)
        pca_model_reduced = pca.PCA(N_components=optimal_dimensions)
        pca_model_reduced.fit(X_train)
        X_reduced = pca_model_reduced.transform(X_train)
        model3.fit(X_reduced)

        clusters1 = model1.predict(X_train)
        clusters2 = model2.predict(X_train)
        clusters3 = model3.predict(X_reduced)

        cluster1 = {}
        cluster2 = {}
        cluster3 = {}

        for i in range(len(clusters1)):
            if clusters1[i] not in cluster1:
                cluster1[clusters1[i]] = []
            cluster1[clusters1[i]].append(Words[i])

        for i in range(len(clusters2)):
            if clusters2[i] not in cluster2:
                cluster2[clusters2[i]] = []
            cluster2[clusters2[i]].append(Words[i])

        for i in range(len(clusters3)):
            if clusters3[i] not in cluster3:
                cluster3[clusters3[i]] = []
            cluster3[clusters3[i]].append(Words[i])
        
        with open('figures/Kmeans_clusters.txt', 'w') as f:
            f.write('The clusters using k_kmeans1 are:\n')
            for key in cluster1:
                f.write(str(key) + ' ' + str(cluster1[key]) + '\n')
            f.write('----------------------------------------------------------------\n')
            f.write('The clusters using k2 are:\n')
            for key in cluster2:
                f.write(str(key) + ' ' + str(cluster2[key]) + '\n')
            f.write('----------------------------------------------------------------\n')
            f.write('The clusters using k_kmeans3 are:\n')
            for key in cluster3:
                f.write(str(key) + ' ' + str(cluster3[key]) + '\n')
            f.write('----------------------------------------------------------------\n')

    print('If you want to do cluster analysis using GMMs? Enter yes')
    if input() == 'yes':
        # GMM cluseting Ananlysis
        model1 = Gmm.GMM(K=k_gmm1)
        model1.fit(X_train)

        model2 = Gmm.GMM(K=k2)
        model2.fit(X_train)

        model3 = Gmm.GMM(K=k_gmm3)
        pca_model_reduced = pca.PCA(N_components=optimal_dimensions)
        pca_model_reduced.fit(X_train)
        X_reduced = pca_model_reduced.transform(X_train)
        model3.fit(X_reduced)

        membership1 = model1.getMembership(X_train)
        membership1 = np.argmax(membership1, axis=1)

        membership2 = model2.getMembership(X_train)
        membership2 = np.argmax(membership2, axis=1)

        membership3 = model3.getMembership(X_reduced)
        membership3 = np.argmax(membership3, axis=1)

        cluster1 = {}
        cluster2 = {}
        cluster3 = {}

        for i in range(len(membership1)):
            if membership1[i] not in cluster1:
                cluster1[membership1[i]] = []
            cluster1[membership1[i]].append(Words[i])

        for i in range(len(membership2)):
            if membership2[i] not in cluster2:
                cluster2[membership2[i]] = []
            cluster2[membership2[i]].append(Words[i])

        for i in range(len(membership3)):
            if membership3[i] not in cluster3:
                cluster3[membership3[i]] = []
            cluster3[membership3[i]].append(Words[i])


        with open('figures/Gmm_clusters.txt', 'w') as f:
            f.write('The clusters using k_gmm1 are:\n')
            for key in cluster1:
                f.write(str(key) + ' ' + str(cluster1[key]) + '\n')
            f.write('----------------------------------------------------------------\n')
            f.write('The clusters using k2 are:\n')
            for key in cluster2:
                f.write(str(key) + ' ' + str(cluster2[key]) + '\n')
            f.write('----------------------------------------------------------------\n')
            f.write('The clusters using k_gmm3 are:\n')
            for key in cluster3:
                f.write(str(key) + ' ' + str(cluster3[key]) + '\n')
            f.write('----------------------------------------------------------------\n')

    print("If you want to do cluster analysis among best of K-means and GMMs? Enter yes")
    if input() == 'yes':
        pca_model_reduced = pca.PCA(N_components=optimal_dimensions)
        pca_model_reduced.fit(X_train)
        X_reduced = pca_model_reduced.transform(X_train)

        k_means_model = K_means.Kmeans(K=k_kmeans)
        k_means_model.fit(X_reduced)
        clusters_kmeans = k_means_model.predict(X_reduced)


        gmm_model = Gmm.GMM(K=k_gmm)
        gmm_model.fit(X_reduced)
        membership = gmm_model.getMembership(X_reduced)
        membership = np.argmax(membership, axis=1)

        print('The clusters using kmeans are:', clusters_kmeans)
        print('The clusters using GMMs are:', membership)

        cluster_kmeans = {}
        cluster_gmm = {}

        for i in range(len(clusters_kmeans)):
            if clusters_kmeans[i] not in cluster_kmeans:
                cluster_kmeans[clusters_kmeans[i]] = []
            cluster_kmeans[clusters_kmeans[i]].append(Words[i])

        for i in range(len(membership)):
            if membership[i] not in cluster_gmm:
                cluster_gmm[membership[i]] = []
            cluster_gmm[membership[i]].append(Words[i])

        with open('figures/Best_clusters.txt', 'w') as f:
            f.write('The clusters using kmeans are:\n')
            for key in cluster_kmeans:
                f.write(str(key) + ' ' + str(cluster_kmeans[key]) + '\n')
            f.write('----------------------------------------------------------------\n')
            f.write('The clusters using GMMs are:\n')
            for key in cluster_gmm:
                f.write(str(key) + ' ' + str(cluster_gmm[key]) + '\n')
            f.write('----------------------------------------------------------------\n')


    
if question == 6:
    # Part-1:
    euclidean_dist = pdist(X_train, metric='euclidean')
    cosine_dist = pdist(X_train, metric='cosine')

    # Method = 'complete'
    plt.figure(figsize=(12, 6))
    plt.title('Euclidean_complete')
    dendrogram(linkage(euclidean_dist, method='complete'))
    plt.savefig('figures/Hierarchical_Clustering_complete_euclidean.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.title('Cosine_complete')
    dendrogram(linkage(cosine_dist, method='complete'))
    plt.savefig('figures/Hierarchical_Clustering_complete_cosine.png')
    plt.close()

    # Method = 'single'
    plt.figure(figsize=(12, 6))
    plt.title("Euclidean_single")
    dendrogram(linkage(euclidean_dist, method='single'))
    plt.savefig('figures/Hierarchical_Clustering_single_euclidean.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.title('Cosine_single')
    dendrogram(linkage(cosine_dist, method='single'))
    plt.savefig('figures/Hierarchical_Clustering_single_cosine.png')
    plt.close()

    # Method = 'average'
    plt.figure(figsize=(12, 6))
    plt.title('Euclidean_average')
    dendrogram(linkage(euclidean_dist, method='average'))
    plt.savefig('figures/Hierarchical_Clustering_average_euclidean.png')
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.title('Cosine_average')
    dendrogram(linkage(cosine_dist, method='average'))
    plt.savefig('figures/Hierarchical_Clustering_average_cosine.png')
    plt.close()

    # Method = 'ward'
    plt.figure(figsize=(12, 6))
    plt.title('Euclidean_ward')
    dendrogram(linkage(euclidean_dist, method='ward'))
    plt.savefig('figures/Hierarchical_Clustering_ward_euclidean.png')
    plt.close()


    plt.figure(figsize=(12, 6))
    plt.title('Cosine_ward')
    dendrogram(linkage(cosine_dist, method='ward'))
    plt.savefig('figures/Hierarchical_Clustering_ward_cosine.png')
    plt.close()

    # write a report analysing the results of hierarchical clustering using different distance metrics and linkage methods : pending


    # Part-2:
    best_method = 'ward'
    k_best1 = k_kmeans3
    k_best2 = k_gmm3

    pca_model = pca.PCA(N_components=optimal_dimensions)
    pca_model.fit(X_train)
    X_reduced = pca_model.transform(X_train)

    # For K-means
    linkage_matrix = linkage(X_train, method=best_method)
    cluster1 = fcluster(linkage_matrix, k_best1, criterion='maxclust')
    cluster1 = cluster1 - 1

    model = K_means.Kmeans(K=k_best1)
    model.fit(X_reduced)
    cluster_kmeans = model.predict(X_reduced)

    print('The clusters using hierarchical clustering are:', cluster1)
    print('The clusters using kmeans are:', cluster_kmeans)
    ari_score = adjusted_rand_score(cluster_kmeans, cluster1)
    print('when we compare hierarchical and kmeans the ARI score is:', ari_score)

    # For GMM
    linkage_matrix = linkage(X_train, method=best_method)
    cluster2 = fcluster(linkage_matrix, k_best2, criterion='maxclust')
    cluster2 = cluster2 - 1

    gmm_model = Gmm.GMM(K=k_best2)
    gmm_model.fit(X_reduced)
    membership = gmm_model.getMembership(X_reduced)
    membership = np.argmax(membership, axis=1)
    cluster_gmm = membership

    print('The clusters using hierarchical clustering are:', cluster2)
    print('The clusters using kmeans are:', cluster_gmm)
    ari_score = adjusted_rand_score(cluster_gmm, cluster2)
    print('Whwn we compare hierarchical and Gmm the ARI score is:', ari_score)

    
if question == 7:
    # import data from the folder interim
    spotify_data = pd.read_csv('../../data/interim/spotify_normalized.csv')

    x_train, y_train, x_valid, y_valid, x_test, y_test = split_test_valid_train_sets(spotify_data)

    # Optimal dimenions using PCA
    pca_model = pca.PCA(N_components=5)
    pca_model.fit(x_train)
    eigenvalues = pca_model.sort_eigen

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='--', color='b')
    threshold = 1
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.savefig('figures/PCA_spotify.png')
    plt.close()

    explained_variance = eigenvalues
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    cummulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(cummulative_variance) + 1), cummulative_variance, linestyle='--', color='b')
    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.title('Cummulative Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Cummulative Variance')
    plt.savefig('figures/Cummulative_variance_spotify.png')
    plt.close()

    optimal_dimensions = 16
    pca_model = pca.PCA(N_components=optimal_dimensions)
    pca_model.fit(x_train)
    x_reduced_train = pca_model.transform(x_train)

    optimal_dimensions = 16
    pca_model = pca.PCA(N_components=optimal_dimensions)
    pca_model.fit(x_test)
    x_reduced_test = pca_model.transform(x_test)

    print('Do you apply KNN on the reduced data and comapre the results with original data? Enter yes')
    if input() == 'yes':
        time_list = []
        # Apply KNN on the reduced data
        model = knn.KNN_optim(k=16, dist_metric='manhattan')
        model.train_model(x_reduced_train, y_train)
        start = time.time()
        y_pred = model.predict(x_reduced_test)
        end = time.time()
        time_list.append(end-start)
        results = cm.Measures(y_test, y_pred)

        print('Metrics for KNN on reduced data:')
        print('The accuracy of the model is:', results.accuracy())
        print('The micro precision of the model is:', results.precision_micro())
        print('The macro precision of the model is:', results.precision_macro())
        print('The micro recall of the model is:', results.recall_micro())
        print('The macro recall of the model is:', results.recall_macro())
        print('The F1 micro score of the model is:', results.f1_score_micro())
        print('The F1 macro score of the model is:', results.f1_score_macro())

        # Apply KNN on the original data
        model = knn.KNN_optim(k=16, dist_metric='manhattan')
        model.train_model(x_train, y_train)
        start = time.time()
        y_pred = model.predict(x_test)
        end = time.time()
        time_list.append(end-start)
        results = cm.Measures(y_test, y_pred)

        print('Metrics for KNN on original data:')
        print('The accuracy of the model is:', results.accuracy())
        print('The micro precision of the model is:', results.precision_micro())
        print('The macro precision of the model is:', results.precision_macro())
        print('The micro recall of the model is:', results.recall_micro())
        print('The macro recall of the model is:', results.recall_macro())
        print('The F1 micro score of the model is:', results.f1_score_micro())
        print('The F1 macro score of the model is:', results.f1_score_macro())

        # Compute and plot the inference times for KNN with PCA and without PCA : pending
        plt.bar(['PCA', 'Without PCA'], time_list)
        plt.title('Inference times for KNN with PCA and without PCA')
        plt.xlabel('Method')
        plt.ylabel('Time')
        plt.savefig('figures/Inference_times_KNN.png')
        plt.close()

if question == 8:
    quit()