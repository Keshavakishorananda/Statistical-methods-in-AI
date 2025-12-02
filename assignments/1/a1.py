import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import time
import sys
sys.path.append('../../')
import models.knn.knn
import models.linear_regression.linear_regression as lr
import performance_measures.classification_measures as performance

import performance_measures.regression_measures
rm = performance_measures.regression_measures.RegressionMeasures()

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

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


def drop_features(path, features):
    df = pd.read_csv(path)
    df = df.drop(features, axis=1)
    return df

def preprocessing(path):
    df = pd.read_csv(path)

    # Drop the duplicates and unwanted columns
    df = df.drop('Unnamed: 0', axis=1)

    df = df.dropna()

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset='track_id')
    df = df.drop('track_id', axis=1)

    cat_colums = df.select_dtypes(include='object').columns.to_list()
    num_colums = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
    bool_colums = df.select_dtypes(include='bool').columns.to_list()

    # Encoding
    for colums in cat_colums:
        if colums != 'track_genre':
            frequency = df[colums].value_counts()
            df[colums] = df[colums].map(frequency)
            df[colums] = df[colums].astype('float64')

    df.replace({True: 1, False: 0}, inplace=True)
    df['explicit'] = df['explicit'].astype('float64')

    df['track_genre'] = df['track_genre'].astype('category')
    df['track_genre'] = df['track_genre'].cat.codes
    df['track_genre'] = df['track_genre'].astype('int64')


    # Normalization
    for colums in num_colums:
        df[colums] = (df[colums] - df[colums].mean()) / df[colums].std()

    for colums in cat_colums:
        if colums != 'track_genre':
            df[colums] = (df[colums] - df[colums].mean()) / df[colums].std()

    return df




print(f'Welcome to the Assignment 1')
print(f'Which Question do you want to do? Enter 1, 2 or 3')
question = input()

if question == '1':
    # Task 1
    print(f'Do you want to do the TASK-1? Enter yes or no')
    task1 = input()

    if task1 == 'yes':
        
        df = pd.read_csv('../../data/external/spotify.csv')

        # Drop the duplicates and unwanted columns
        df = df.drop('Unnamed: 0', axis=1)

        df = df.dropna()

        df = df.drop_duplicates()
        df = df.drop_duplicates(subset='track_id')
        df = df.drop('track_id', axis=1)

        df.to_csv('../../data/interim/spotify_drop_duplicates.csv', index=False)

        cat_colums = df.select_dtypes(include='object').columns.to_list()
        num_colums = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
        bool_colums = df.select_dtypes(include='bool').columns.to_list()

        #Encoding
        for colums in cat_colums:
            if colums != 'track_genre':
                frequency = df[colums].value_counts()
                df[colums] = df[colums].map(frequency)
                df[colums] = df[colums].astype('float64')

        df.replace({True: 1, False: 0}, inplace=True)
        df['explicit'] = df['explicit'].astype('float64')

        # LLM prompt : How to do label encoding for the target variable
        # start code 
        df['track_genre'] = df['track_genre'].astype('category')
        df['track_genre'] = df['track_genre'].cat.codes
        df['track_genre'] = df['track_genre'].astype('int64')
        # end code

        df.to_csv('../../data/interim/spotify_encoded.csv', index=False)

        # Normalization
        for colums in num_colums:
            df[colums] = (df[colums] - df[colums].mean()) / df[colums].std()

        for colums in cat_colums:
            if colums != 'track_genre':
                df[colums] = (df[colums] - df[colums].mean()) / df[colums].std()

        
        df.to_csv('../../data/interim/spotify_normalized.csv', index=False)


        # 1. Find the distribution of the features
        print('Do you want to find the distribution of the features and comment on skewed data and outliers? Enter yes or no')
        distribution = input()

        if distribution == 'yes':
            df = pd.read_csv('../../data/interim/spotify_drop_duplicates.csv')

            cat_colums = df.select_dtypes(include='object').columns.to_list()
            num_colums = df.select_dtypes(include=['float64', 'int64']).columns.to_list()
            bool_colums = df.select_dtypes(include='bool').columns.to_list()


            # Show the ditrsibution of both categorical and numerical features

            # LLM prompt : How to plot the bar plot for the categorical features
            # start code
            for cate in cat_colums:
                x = df[cate].value_counts().index
                y = df[cate].value_counts().values
                plt.figure(figsize=(10,5))
                sns.barplot(x=x[:30], y=y[:30])
                plt.xticks(rotation=90)
                plt.savefig(f'figures/distribution_spotify/{cate}_distribution.png')
                plt.close()
            # end code

            # LLM prompt : How to plot the histogram and boxplot for the numerical features
            # start code
            for x in num_colums:
                data = df[x]
                data = np.array(data)

                plt.subplot(1,2,1)
                sns.histplot(data, bins=30, kde=True, color='lightgreen', edgecolor='red')
                plt.xlabel(x)
                plt.ylabel('frequency')
                plt.title('Histogram with density plot')

                plt.subplot(1,2,2)
                sns.boxplot(data)
                plt.xlabel(x)
                plt.title('boxplot')

                plt.tight_layout()

                plt.savefig(f'figures/distribution_spotify/{x}_distribution.png')
                plt.close()
            # end code


            ## Comment on the skewed data by observing the plots
            mean_list = []
            median_list =  []

            for colums in num_colums:
                mean_list.append(df[colums].mean())
                median_list.append(df[colums].median())
            
            plt.figure(figsize=(10,5))

            plt.subplot(1,2,1)
            sns.barplot(x=num_colums, y=mean_list)
            plt.xlabel('Features')
            plt.ylabel('Mean')


            plt.subplot(1,2,2)
            sns.barplot(x=num_colums, y=median_list)
            plt.xlabel('Features')
            plt.ylabel('Median')

            plt.tight_layout()
            plt.savefig('figures/distribution_spotify/mean_median.png')
            plt.close()

            for i in range(len(num_colums)):
                if mean_list[i] > median_list[i]:
                    print(f'{num_colums[i]} is right skewed')
                elif mean_list[i] < median_list[i]:
                    print(f'{num_colums[i]} is left skewed')
                else:
                    print(f'{num_colums[i]} is normally distributed')

            
            ## comment on outliers by observing the boxplot
            # LLM prompt : How to find the outliers
            # start code
            for colums in num_colums:
                q1 = df[colums].quantile(0.25)
                q3 = df[colums].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - 1.5*iqr
                upper_bound = q3 + 1.5*iqr

                outliers = df[(df[colums] < lower_bound) | (df[colums] > upper_bound)]
                print(f'Number of outliers in {colums} is {len(outliers)}')
            # end code



        # 2. Find the correlation between the features and labelled data
        print('Do you want to find the correlation between the features and labelled data? Enter yes or no')
        correlation = input()

        if correlation == 'yes':
            df = pd.read_csv('../../data/interim/spotify_drop_duplicates.csv')
            cat_colums = df.select_dtypes(include='object').columns.to_list()
            num_colums = df.select_dtypes(include=['float64', 'int64']).columns.to_list()

            for colums in cat_colums:
                if colums != 'track_genre':
                    x = df[colums].unique().tolist()
                    y = df.groupby(colums)['track_genre'].unique()

                    len_list = []
                    for i in x:
                        len_list.append(len(y.loc[i]))

                    plt.figure(figsize=(10,5))
                    sns.barplot(x=x[:30], y=len_list[:30])
                    plt.xticks(rotation=45)
                    plt.savefig(f'figures/correlation_spotify/{colums}_vs_track_genre.png')
                    plt.close()


            df = pd.read_csv('../../data/interim/spotify_normalized.csv')

            feature_corr = {}
            for colums in num_colums:
                correlation = df['track_genre'].corr(df[colums])

                if correlation < 0:
                    feature_corr[colums] = -correlation
                else:
                    feature_corr[colums] = correlation

                plt.figure(figsize=(10, 6))
                plt.scatter(df[colums], df['track_genre'])
                plt.title(f'{colums} vs track_genre')
                plt.xticks(rotation=45)
                plt.savefig(f'figures/correlation_spotify/{colums}_vs_track_genre.png')
                plt.close()
            
            # LLM prompt : How to sort the dictionary based on values
            # start code
            sorted_feature_corr = dict(sorted(feature_corr.items(), key=lambda item: item[1], reverse=True))
            # end code

            print(sorted_feature_corr)



    # Task 2
    print(f'Do you want to do Task 2? ')
    task2 = input()

    if task2 == 'yes':
        df = pd.read_csv('../../data/interim/spotify_normalized.csv')

        x_train, y_train, x_valid, y_valid, x_test, y_test = split_test_valid_train_sets(df)

        # import KNN model
        knn_optim = models.knn.knn.KNN_optim(16, 'manhattan')
        knn_optim.train_model(x_train, y_train)

        y_pred = []

        y_pred = knn_optim.predict(x_test)

        y_pred = np.array(y_pred)

        performance_measures1 = performance.Measures(y_test, y_pred)
        print(f'accuracy : {performance_measures1.accuracy()}')

        print(f'macro precision : {performance_measures1.precision_macro()}')
        print(f'macro recall : {performance_measures1.recall_macro()}')
        print(f'macro f1_score : {performance_measures1.f1_score_macro()}')

        print(f'micro precision : {performance_measures1.precision_micro()}')
        print(f'micro recall : {performance_measures1.recall_micro()}')
        print(f'micro f1_score : {performance_measures1.f1_score_micro()}')


    # Task 3
    print(f'Do you want to do Task 3? Enter yes or no')
    task3 = input()
    best_model = []

    if task3 == 'yes':
        df = pd.read_csv('../../data/interim/spotify_normalized.csv')

        x_train, y_train, x_valid, y_valid, x_test, y_test = split_test_valid_train_sets(df)

        best_model_list = []
        accuracy_euclidean = []
        accuracy_manhattan = []
        accuracy_cosine = []
        k_range = 31

        # import KNN model
        knn_optim_euclidean = []

        for i in range(1, k_range):
            knn_optim_euclidean.append(models.knn.knn.KNN_optim(i, 'euclidean'))
            knn_optim_euclidean[i-1].train_model(x_train, y_train)

        for i in range(1, k_range):
            y_pred = knn_optim_euclidean[i-1].predict(x_valid)
            accur = performance.Measures(y_valid, y_pred).accuracy()
            accuracy_euclidean.append(accur)
            info_model = ['euclidean', i, accur]
            best_model_list.append(info_model)
            print(f'Accuracy for k={i} and euclidean distance metric is {accur}')


        knn_optim_manhattan = []

        for i in range(1, k_range):
            knn_optim_manhattan.append(models.knn.knn.KNN_optim(i, 'manhattan'))
            knn_optim_manhattan[i-1].train_model(x_train, y_train)

        for i in range(1, k_range):
            y_pred = knn_optim_manhattan[i-1].predict(x_valid)
            accur = performance.Measures(y_valid, y_pred).accuracy()
            accuracy_manhattan.append(accur)
            info_model = ['manhattan', i, accur]
            best_model_list.append(info_model)
            print(f'Accuracy for k={i} and manhattan distance metric is {accur}')



        knn_optim_cosine = []

        for i in range(1, k_range):
            knn_optim_cosine.append(models.knn.knn.KNN_optim(i, 'cosine'))
            knn_optim_cosine[i-1].train_model(x_train, y_train)

        for i in range(1, k_range):
            y_pred = knn_optim_cosine[i-1].predict(x_valid)
            accur = performance.Measures(y_valid, y_pred).accuracy()
            accuracy_cosine.append(accur)
            info_model = ['cosine', i, accur]
            best_model_list.append(info_model)
            print(f'Accuracy for k={i} and cosine distance metric is {accur}')


        best_model_list = sorted(best_model_list, key=lambda x: x[2], reverse=True)

        # 1. Find the best model with the best hyperparameters
        best_model = best_model_list[0]
        print(f'best model with more accuracy has distance metric {best_model[0]}, k={best_model[1]} and accuracy of {best_model[2]}')

        # 2. print the top 10 models

        print(f'The best model is with distance metric {best_model_list[0][0]}, k={best_model_list[0][1]} and accuracy of {best_model_list[0][2]}')

        for i in range(10):
            print(f'The model with distance metric {best_model_list[i][0]}, k={best_model_list[i][1]} and accuracy of {best_model_list[i][2]}')


        # 3. plot the graph for k vs accuracy for Eulidean distance metric
        k = [i for i in range(1, 31)]
        plt.plot(k, accuracy_euclidean)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title('k vs accuracy for Eulidean distance metric')
        plt.savefig('figures/knn_figures/k_vs_accuracy_euclidean.png')
        plt.close()

        plt.plot(k, accuracy_manhattan)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title('k vs accuracy for Manhattan distance metric')
        plt.savefig('figures/knn_figures/k_vs_accuracy_manhattan.png')
        plt.close()

        plt.plot(k, accuracy_cosine)
        plt.xlabel('k')
        plt.ylabel('accuracy')
        plt.title('k vs accuracy for Cosine distance metric')
        plt.savefig('figures/knn_figures/k_vs_accuracy_cosine.png')
        plt.close()

    print(f'Do you want to accuracy by dropping each feature once? Enter yes or no')
    inteim_task = input()

    if inteim_task == 'yes':
        # 4. Drop features and check the accuracy
        df_init = pd.read_csv('../../data/interim/spotify_normalized.csv')

        for col in df_init.columns:
            if col != 'track_genre':
                df = drop_features('../../data/interim/spotify_normalized.csv', col)
                x_train, y_train, x_valid, y_valid, x_test, y_test = split_test_valid_train_sets(df)

                knn_feature = models.knn.knn.KNN_optim(best_model[1], best_model[0])
                knn_feature.train_model(x_train, y_train)

                y_pred = knn_feature.predict(x_test)

                y_pred = np.array(y_pred)

                performance_measures1 = performance.Measures(y_test, y_pred)
                accur = performance_measures1.accuracy()
                print(f'accuracy for dropping {col} : {performance_measures1.accuracy()}')

                # write the above print statement to a file
                with open('figures/knn_figures/document.txt', 'a') as f:
                    f.write(f'accuracy for dropping {col} : {accur}\n')


    # Task 4
    print(f'Do you want to do Task 4? Enter yes or no')
    task4 = input()

    if task4 == 'yes':

        print('Do you want to plot time vs training data points for all models? Enter yes or no')
        inff_time = input()

        if inff_time == 'yes':
            init_knn = models.knn.knn.KNN(5, 'manhattan')
            best_knn = models.knn.knn.KNN(16, 'manhattan')
            optimized_knn = models.knn.knn.KNN_optim(16, 'manhattan')
            sklearn_knn = KNeighborsClassifier(n_neighbors=16, metric='manhattan')

            time_list = []

            df = pd.read_csv('../../data/interim/spotify_normalized.csv')
            x_train, y_train, x_valid, y_valid, x_test, y_test = split_test_valid_train_sets(df)

            # LLM prompt : How to use sklearn KNN model
            # start code
            init_knn.train_model(x_train, y_train)
            best_knn.train_model(x_train, y_train)
            optimized_knn.train_model(x_train, y_train)
            sklearn_knn.fit(x_train, y_train)
            # end code

            # LLM prompt : How to record the time taken for the prediction
            # start code
            time_start = time.time()
            init_knn.predict(x_valid[0])
            time_end = time.time()
            print(f'init_knn time is {(time_end-time_start)*8974}')
            time_list.append((time_end-time_start)*8974)
            # end code

            time_start = time.time()
            best_knn.predict(x_valid[0])
            time_end = time.time()
            print(f'best_knn time is {(time_end-time_start)*8974}')
            time_list.append((time_end-time_start)*8974)

            time_start = time.time()
            optimized_knn.predict(x_valid)
            time_end = time.time()
            print(f'optimized_knn time is {time_end-time_start}')
            time_list.append(time_end-time_start)

            time_start = time.time()
            sklearn_knn.predict(x_valid)
            time_end = time.time()
            print(f'sklearn_knn time is {time_end-time_start}')
            time_list.append(time_end-time_start)


            plt.bar(['init_knn', 'best_knn', 'optimized_knn', 'sklearn_knn'], time_list)
            plt.xlabel('Models')
            plt.ylabel('Time taken')
            plt.savefig('figures/knn_figures/time_vs_training_data_points.png')
            plt.close()



        print('Do you want to plot inference time vs training data points? Enter yes or no')
        inference_time = input()

        if inference_time == 'yes':
            init_knn = models.knn.knn.KNN(5, 'manhattan')
            best_knn = models.knn.knn.KNN(16, 'manhattan')
            optimized_knn = models.knn.knn.KNN_optim(16, 'manhattan')
            sklearn_knn = KNeighborsClassifier(n_neighbors=16, metric='manhattan')

            init_knn_time = []
            best_knn_time = []
            optimized_knn_time = []
            sklearn_knn_time = []

            df = pd.read_csv('../../data/interim/spotify_normalized.csv')
            x_train, y_train, x_valid, y_valid, x_test, y_test = split_test_valid_train_sets(df)

            for i in range(1000, 10001, 1000):
                init_knn.train_model(x_train[:i], y_train[:i])
                best_knn.train_model(x_train[:i], y_train[:i])
                optimized_knn.train_model(x_train[:i], y_train[:i])
                sklearn_knn.fit(x_train[:i], y_train[:i])
                


                start_time = time.time()
                init_knn.predict(x_valid[0])
                end_time = time.time()
                print(f'init_knn time for {i} data points is {(end_time-start_time)*8974}')
                init_knn_time.append((end_time-start_time)*8974)

                start_time = time.time()
                best_knn.predict(x_valid[0])
                end_time = time.time()
                print(f'best_knn time for {i} data points is {(end_time-start_time)*8974}')
                best_knn_time.append((end_time-start_time)*8974)

                start_time = time.time()
                optimized_knn.predict(x_valid)
                end_time = time.time()
                print(f'optimized_knn time for {i} data points is {end_time-start_time}')
                optimized_knn_time.append(end_time-start_time)
                
                start_time = time.time()
                sklearn_knn.predict(x_valid)
                end_time = time.time()
                print(f'sklearn_knn time for {i} data points is {end_time-start_time}')
                sklearn_knn_time.append(end_time-start_time)

            plt.plot([i for i in range(1000, 10001, 1000)], init_knn_time, label='init_knn')
            plt.plot([i for i in range(1000, 10001, 1000)], best_knn_time, label='best_knn')
            plt.plot([i for i in range(1000, 10001, 1000)], optimized_knn_time, label='optimized_knn')
            plt.plot([i for i in range(1000, 10001, 1000)], sklearn_knn_time, label='sklearn_knn')

            plt.xlabel('Number of data points')
            plt.ylabel('Time taken')

            plt.legend()
            plt.savefig('figures/knn_figures/time_vs_data_points.png')
            plt.close()
            

    # Task 5
    print(f'Do you want to do Task 5? Enter yes or no')
    task5 = input()

    if task5 == 'yes':
        
        train_set = preprocessing('../../data/external/spotify-2/train.csv')
        test_set = preprocessing('../../data/external/spotify-2/test.csv')
        valid_set = preprocessing('../../data/external/spotify-2/validate.csv')

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

        
        best_model = models.knn.knn.KNN_optim(16, 'manhattan')
        best_model.train_model(x_train, y_train)

        y_pred = best_model.predict(x_test)

        y_pred = np.array(y_pred)

        performance_measures1 = performance.Measures(y_test, y_pred)
        print(f'accuracy : {performance_measures1.accuracy()}')
        print(f'macro precision : {performance_measures1.precision_macro()}')
        print(f'macro recall : {performance_measures1.recall_macro()}')
        print(f'macro f1_score : {performance_measures1.f1_score_macro()}')
        print(f'micro precision : {performance_measures1.precision_micro()}')
        print(f'micro recall : {performance_measures1.recall_micro()}')
        print(f'micro f1_score : {performance_measures1.f1_score_micro()}')


if question == '2':
    
    # Task 0
    print(f'Do you want to do Task 0? Enter yes or no')
    task0 = input()

    if task0 == 'yes':

        df = pd.read_csv('../../data/external/linreg.csv')

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_len = int(0.8*len(df))
        valid_len = int(0.1*len(df))
        test_len = len(df) - train_len - valid_len

        train_set = df[:train_len]
        valid_set = df[train_len:train_len+valid_len]
        test_set = df[train_len+valid_len:]

        train_set.to_csv('../../data/interim/linreg_train_set.csv', index=False)
        valid_set.to_csv('../../data/interim/linreg_valid_set.csv', index=False)
        test_set.to_csv('../../data/interim/linreg_test_set.csv', index=False)

        x_train = train_set['x']
        y_train = train_set['y']

        x_valid = valid_set['x']
        y_valid = valid_set['y']

        x_test = test_set['x']
        y_test = test_set['y']

        print("Fot Train set:")
        print(f'Means for train set: {x_train.mean()}, {y_train.mean()}')
        print(f'variances for train set: {x_train.var()}, {y_train.var()}')
        print(f'Stds for train set: {x_train.std()}, {y_train.std()}')

        print("For Validation set:")
        print(f'Means for validation set: {x_valid.mean()}, {y_valid.mean()}')
        print(f'variances for validation set: {x_valid.var()}, {y_valid.var()}')
        print(f'Stds for validation set: {x_valid.std()}, {y_valid.std()}')

        print("For Test set:")
        print(f'Means for test set: {x_test.mean()}, {y_test.mean()}')
        print(f'variances for test set: {x_test.var()}, {y_test.var()}')
        print(f'Stds for test set: {x_test.std()}, {y_test.std()}')

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        plt.scatter(x_train, y_train)
        plt.scatter(x_valid, y_valid)
        plt.scatter(x_test, y_test)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Scatter plot of the train, validation and test set')
        plt.savefig('figures/linreg_fig/linreg_data.png')
        plt.close()

    
    # Task 1
    print(f'Do you want to do Task 1? Enter yes or no')
    task1 = input()

    if task1 == 'yes':

        train_set = pd.read_csv('../../data/interim/linreg_train_set.csv')
        valid_set = pd.read_csv('../../data/interim/linreg_valid_set.csv')
        test_set = pd.read_csv('../../data/interim/linreg_test_set.csv')

        x_train = train_set['x']
        y_train = train_set['y']

        x_valid = valid_set['x']
        y_valid = valid_set['y']

        x_test = test_set['x']
        y_test = test_set['y']

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Hyperparameter tuning for the learning rate
        print('Hyperparameter tuning for the learning rate')
        lin_reg1 = lr.linear_regression(1,0.1, 18)
        lin_reg2 = lr.linear_regression(1,0.2, 18)
        lin_reg3 = lr.linear_regression(1,0.3, 18)
        lin_reg4 = lr.linear_regression(1,0.01, 18)
        lin_reg5 = lr.linear_regression(1,0.001, 18)

        lin_reg1.train(x_train, y_train)
        lin_reg2.train(x_train, y_train)
        lin_reg3.train(x_train, y_train)
        lin_reg4.train(x_train, y_train)
        lin_reg5.train(x_train, y_train)

        for i in range(100):
            lin_reg1.update_parameters()
            lin_reg2.update_parameters()
            lin_reg3.update_parameters()
            lin_reg4.update_parameters()
            lin_reg5.update_parameters()



        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = lin_reg1.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.1')



        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = lin_reg2.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.2')


        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = lin_reg3.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.3')



        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = lin_reg4.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.01')


        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = lin_reg5.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.001')

        print('So, finally we can say that n = 0.1 is the best learning rate for this model')


        # Find the MSE, standard deviation and variance for the test set
        print('Now test the model with the test set')
        y_pred = []
        for x,y in zip(x_test, y_test):
            pred = lin_reg1.predict(x)
            y_pred.append(pred)

        print("For Test set:")
        print(f'MSE: {rm.MSE(y_test, y_pred)}')
        print(f'Std: {rm.standard_deviation(y_pred)}')
        print(f'Var: {rm.variance(y_pred)}')

        y_pred = []
        for x,y in zip(x_train, y_train):
            pred = lin_reg1.predict(x)
            y_pred.append(pred)

        print("For Train set:")
        print(f'MSE: {rm.MSE(y_train, y_pred)}')
        print(f'Std: {rm.standard_deviation(y_pred)}')
        print(f'Var: {rm.variance(y_pred)}')

        # plot the line with training points
        plt.scatter(x_train, y_train, color='blue')
        plt.plot(x_train, [lin_reg1.predict(x) for x in x_train], color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Line with training points')
        plt.savefig('figures/linreg_fig/lin_plot.png')
        plt.close()

    
    # Task 2
    print(f'Do you want to do Task 2? Enter yes or no')
    task2 = input()

    if task2 == 'yes':
        train_set = pd.read_csv('../../data/interim/linreg_train_set.csv')
        valid_set = pd.read_csv('../../data/interim/linreg_valid_set.csv')
        test_set = pd.read_csv('../../data/interim/linreg_test_set.csv')

        x_train = train_set['x']
        y_train = train_set['y']

        x_valid = valid_set['x']
        y_valid = valid_set['y']

        x_test = test_set['x']
        y_test = test_set['y']

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_test = np.array(x_test)
        y_test = np.array(y_test)


        # Hyperparameter tuning for the learning rate
        print('Hyperparameter tuning for the learning rate')
        poly_reg1 = lr.linear_regression(5,0.1, 18)
        poly_reg2 = lr.linear_regression(5,0.01, 18)
        poly_reg3 = lr.linear_regression(5,0.001, 18)

        poly_reg1.train(x_train, y_train)
        poly_reg2.train(x_train, y_train)
        poly_reg3.train(x_train, y_train)

        for epoch in range(100):
            poly_reg1.update_parameters()

        for epoch in range(100):
            poly_reg2.update_parameters()

        for epoch in range(100):
            poly_reg3.update_parameters()


        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = poly_reg1.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.1')


        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = poly_reg2.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.01')


        y_pred = []
        for x,y in zip(x_valid, y_valid):
            pred = poly_reg3.predict(x)
            y_pred.append(pred)

        print(rm.MSE(y_valid, y_pred), 'when n = 0.001')

        print('So, finally we can say that n = 0.1 is the best learning rate for this model')



        # Finding the best degree for the polynomial regression
        poly_reg = []

        for i in range(1,26):
            poly_reg.append(lr.linear_regression(i,0.1, 18))

        for i in range(1,26):
            poly_reg[i-1].train(x_train, y_train)

        for i in range(1,26):
            for epoch in range(100):
                poly_reg[i-1].update_parameters()

        y_pred_test_MSE = []
        y_pred_test_var = []
        y_pred_test_std = []

        for i in range(1,26):
            y_pred = []
            for x,y in zip(x_test, y_test):
                pred = poly_reg[i-1].predict(x)
                y_pred.append(pred)
                
            y_pred_test_MSE.append(rm.MSE(y_test, y_pred))
            y_pred_test_var.append(rm.variance(y_pred))
            y_pred_test_std.append(rm.standard_deviation(y_pred))

        
        y_pred_train_MSE = []
        y_pred_train_var = []
        y_pred_train_std = []


        for i in range(1,26):
            y_pred = []
            for x,y in zip(x_train, y_train):
                pred = poly_reg[i-1].predict(x)
                y_pred.append(pred)
                
            y_pred_train_MSE.append(rm.MSE(y_train, y_pred))
            y_pred_train_var.append(rm.variance(y_pred))
            y_pred_train_std.append(rm.standard_deviation(y_pred))

        print('For MSE:')
        for i in range(25):
            print(f'MSE on test set for degree {i+1} is : {y_pred_test_MSE[i]}')
            print(f'MSE on train set for degree {i+1} is : {y_pred_train_MSE[i]}')

        print('For Variance:')
        for i in range(25):
            print(f'Variance on test set for degree {i+1} is : {y_pred_test_var[i]}')
            print(f'Variance on train set for degree {i+1} is : {y_pred_train_var[i]}')

        print('For Standard Deviation:')
        for i in range(25):
            print(f'Standard Deviation on test set for degree {i+1} is : {y_pred_test_std[i]}')
            print(f'Standard Deviation on train set for degree {i+1} is : {y_pred_train_std[i]}')


        print('So, the k value that gives the least MSE for test set is 15')
        
        # write the above print statement to a file
        with open('figures/linreg_fig/document.txt', 'w') as f:
            f.write(f'{poly_reg[15].parameters}\n')

    # Task 3
    print(f'Do you want to do Task 3? Enter yes or no')
    task3 = input()

    if task3 == 'yes':
        train_set = pd.read_csv('../../data/interim/linreg_train_set.csv')
        valid_set = pd.read_csv('../../data/interim/linreg_valid_set.csv')
        test_set = pd.read_csv('../../data/interim/linreg_test_set.csv')

        x_train = train_set['x']
        y_train = train_set['y']

        x_valid = valid_set['x']
        y_valid = valid_set['y']

        x_test = test_set['x']
        y_test = test_set['y']

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # For k = 1, n = 0.1 model
        reg1 = lr.linear_regression(1,0.1, 18)
        reg1.train(x_train, y_train)
        MSE_linreg1 = []
        var_linreg1 = []
        std_linreg1 = []

        # For k = 5, n = 0.1 model
        reg2 = lr.linear_regression(5,0.1, 18)
        reg2.train(x_train, y_train)
        MSE_linreg2 = []
        var_linreg2 = []
        std_linreg2 = []


        # For k = 10, n = 0.1 model
        reg3 = lr.linear_regression(10,0.1, 18)
        reg3.train(x_train, y_train)
        MSE_linreg3 = []
        var_linreg3 = []
        std_linreg3 = []

        # For k = 15, n = 0.1 model
        reg4 = lr.linear_regression(15,0.1, 18)
        reg4.train(x_train, y_train)
        MSE_linreg4 = []
        var_linreg4 = []
        std_linreg4 = []

        # For k = 19, n = 0.1 model
        reg5 = lr.linear_regression(19,0.1, 18)
        reg5.train(x_train, y_train)
        MSE_linreg5 = []
        var_linreg5 = []
        std_linreg5 = []

        # For k=15, n=0.1, seed = 7
        reg6 = lr.linear_regression(15,0.1, 7)
        reg6.train(x_train, y_train)
        MSE_linreg6 = []
        var_linreg6 = []
        std_linreg6 = []

        reg_list = []
        reg_list.append(reg1)
        reg_list.append(reg2)
        reg_list.append(reg3)
        reg_list.append(reg4)
        reg_list.append(reg5)
        reg_list.append(reg6)

        MSE_list = []
        MSE_list.append(MSE_linreg1)
        MSE_list.append(MSE_linreg2)
        MSE_list.append(MSE_linreg3)
        MSE_list.append(MSE_linreg4)
        MSE_list.append(MSE_linreg5)
        MSE_list.append(MSE_linreg6)

        var_list = []
        var_list.append(var_linreg1)
        var_list.append(var_linreg2)
        var_list.append(var_linreg3)
        var_list.append(var_linreg4)
        var_list.append(var_linreg5)
        var_list.append(var_linreg6)

        std_list = []
        std_list.append(std_linreg1)
        std_list.append(std_linreg2)
        std_list.append(std_linreg3)
        std_list.append(std_linreg4)
        std_list.append(std_linreg5)
        std_list.append(std_linreg6)

        true_var = y_train.var()
        true_std = y_train.std()


        # Save the graph for GIF
        for i in range(6):
            for epoch in range(101):
                reg_list[i].update_parameters()
                if epoch % 10 == 0:
                    y_pred = []
                    for x,y in zip(x_train, y_train):
                        pred = reg_list[i].predict(x)
                        y_pred.append(pred)

                    mean_squared_error = rm.MSE(y_train, y_pred)
                    variance = rm.variance(y_pred)
                    standard_deviation = rm.standard_deviation(y_pred)
                    MSE_list[i].append(mean_squared_error)
                    var_list[i].append(variance)
                    std_list[i].append(standard_deviation)

                    sorted_data = sorted(zip(x_train, y_pred), key=lambda x: x[0])
                    x_sorted, y_pred_sorted = zip(*sorted_data)

                    fig, axes = plt.subplots(2,2, figsize=(20,10))

                    axes[0, 0].scatter(x_train, y_train)
                    axes[0, 0].plot(x_sorted, y_pred_sorted, color='red')
                    axes[0, 0].set_title(f'Epoch: {epoch}')
                    axes[0, 0].set_xlabel('x')
                    axes[0, 0].set_ylabel('y')

                    index = np.arange(len(MSE_list[i]))
                    axes[0, 1].scatter(index, MSE_list[i])
                    axes[0, 1].set_title('MSE')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('MSE')


                    axes[1, 0].scatter(index, var_list[i])
                    axes[1, 0].axhline(y=true_var, color='r', linestyle='--', label='y=0.5')
                    axes[1, 0].set_title('Variance')
                    axes[1, 0].set_xlabel('Epoch')
                    axes[1, 0].set_ylabel('Variance')



                    axes[1, 1].scatter(index, std_list[i])
                    axes[1, 1].axhline(y=true_std, color='r', linestyle='--', label='y=0.5')
                    axes[1, 1].set_title('Standard Deviation')
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Standard Deviation')
                    

                    plt.tight_layout()
                    if i == 0:
                        plt.savefig(f'figures/linreg_k=1/epoch_{epoch}.jpg')
                        plt.close()
                    elif i == 1:
                        plt.savefig(f'figures/linreg_k=5/epoch_{epoch}.jpg')
                        plt.close()
                    elif i == 2:
                        plt.savefig(f'figures/linreg_k=10/epoch_{epoch}.jpg')
                        plt.close()
                    elif i == 3:
                        plt.savefig(f'figures/linreg_k=15/epoch_{epoch}.jpg')
                        plt.close()
                    elif i == 4:
                        plt.savefig(f'figures/linreg_k=19/epoch_{epoch}.jpg')
                        plt.close()
                    else:
                        plt.savefig(f'figures/linreg_k=15_seed=7/epoch_{epoch}.jpg')
                        plt.close()


        # Now create the GIF
        # Generate the animated gif for all the models
        from PIL import Image, ImageDraw, ImageFont
        import glob
        import cv2
        import numpy as np

        def generate_gif(image_folder, output_gif_path, duration_per_frame):
            # Collect all image paths
            image_paths = glob.glob(f"{image_folder}*.jpg")

            # Sort the images based on epoch number
            image_paths = sorted(image_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

            # Initialize an empty list to store the images
            frames = []

            # Loop through each image file to add text and append to frames
            for image_path in image_paths:
                img = Image.open(image_path)

                # Reduce the frame size by 50%
                img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))

                # Create a new draw object after resizing
                draw = ImageDraw.Draw(img)

                # Text to display at top-left and bottom-right corners
                top_left_text = image_path.split("/")[-1]
                bottom_right_text = "Add your test here to be displayed on Images"

                # Draw top-left text
                draw.text((10, 10), top_left_text, fill=(255, 255, 255))

                # Calculate x, y position of the bottom-right text
                text_width, text_height = draw.textsize(bottom_right_text)
                x = img.width - text_width - 10  # 10 pixels from the right edge
                y = img.height - text_height - 10  # 10 pixels from the bottom edge

                # Draw bottom-right text
                draw.text((x, y), bottom_right_text, fill=(255, 255, 255))

                frames.append(img)


            # Save frames as an animated GIF
            frames[0].save(output_gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        duration=duration_per_frame,
                        loop=0,
                        optimize=True)


        # Generate the animated gif for all the models

        generate_gif("figures/linreg_k=1/", "figures/linreg_k=1/animated_presentation.gif", 250)
        generate_gif("figures/linreg_k=5/", "figures/linreg_k=5/animated_presentation.gif", 250)
        generate_gif("figures/linreg_k=10/", "figures/linreg_k=10/animated_presentation.gif", 250)
        generate_gif("figures/linreg_k=15/", "figures/linreg_k=15/animated_presentation.gif", 250)
        generate_gif("figures/linreg_k=19/", "figures/linreg_k=19/animated_presentation.gif", 250)
        generate_gif("figures/linreg_k=15_seed=7/", "figures/linreg_k=15_seed=7/animated_presentation.gif", 250)

elif question == '3':

    # Task 0
    print(f'Do you want to do Task 0? Enter yes or no')
    task0 = input()

    if task0 == 'yes':
        # Load the data
        df = pd.read_csv('../../data/external/regularisation.csv')

        df = df.sample(frac=1, random_state=18).reset_index(drop=True)

        # Split the data into train, validation and test set
        train_len = int(0.8 * len(df))
        valid_len = int(0.1 * len(df))
        test_len = len(df) - train_len - valid_len

        train_set = df[:train_len]
        valid_set = df[train_len:train_len+valid_len]
        test_set = df[train_len+valid_len:]

        train_set.to_csv('../../data/interim/regularisation_train_set.csv', index=False)
        valid_set.to_csv('../../data/interim/regularisation_valid_set.csv', index=False)
        test_set.to_csv('../../data/interim/regularisation_test_set.csv', index=False)

        x_train = train_set['x']
        y_train = train_set['y']


        x_valid = valid_set['x']
        y_valid = valid_set['y']

        x_test = test_set['x']
        y_test = test_set['y']

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print('For train set')
        print(f'Mean of train set:{np.mean(y_train)}')
        print(f'Standard deviation of train set:{np.std(y_train)}')
        print(f'variance of train set:{np.var(y_train)}')


        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)
        print('For valid set')
        print(f'Mean of valid set:{np.mean(y_valid)}')
        print(f'Standard deviation of valid set:{np.std(y_valid)}')
        print(f'variance of valid set:{np.var(y_valid)}')

        x_test = np.array(x_test)
        y_test = np.array(y_test)
        print('For test set')
        print(f'Mean of test set:{np.mean(y_test)}')
        print(f'Standard deviation of test set:{np.std(y_test)}')
        print(f'variance of test set:{np.var(y_test)}')


        plt.scatter(x_train, y_train, color = 'blue')
        plt.xlabel('x_train')
        plt.ylabel('y_train')
        plt.title('Scatter plot of train set')  
        plt.savefig('figures/regularisation_fig/train_set.png')
        plt.close()
        


    # Task 1
    print(f'Do you want to do Task 1? Enter yes or no')
    task1 = input()

    if task1 == 'yes':
        # Load the data
        train_set = pd.read_csv('../../data/interim/regularisation_train_set.csv')
        valid_set = pd.read_csv('../../data/interim/regularisation_valid_set.csv')
        test_set = pd.read_csv('../../data/interim/regularisation_test_set.csv')

        x_train = train_set['x']
        y_train = train_set['y']


        x_valid = valid_set['x']
        y_valid = valid_set['y']

        x_test = test_set['x']
        y_test = test_set['y']

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        poly_reg = []

        for i in range(1, 17):
            poly_reg.append(lr.linear_regression(i+4,0.1, 18))

        for i in range(16):
            poly_reg[i].train(x_train, y_train)

        for i in range(16):
            for epoch in range(500+i*100):
                poly_reg[i].update_parameters()


        poly_reg_l1 = []

        for i in range(1, 17):
            poly_reg_l1.append(lr.linear_regression(i+4,0.1, 18, r=0.1, regularizer='l1'))

        for i in range(16):
            poly_reg_l1[i].train(x_train, y_train)

        for i in range(16):
            for epoch in range(500+i*100):
                poly_reg_l1[i].update_parameters()



        poly_reg_l2 = []

        for i in range(1, 17):
            poly_reg_l2.append(lr.linear_regression(i+4,0.1, 18, r=0.1, regularizer='l2'))

        for i in range(16):
            poly_reg_l2[i].train(x_train, y_train)

        for i in range(16):
            for epoch in range(500+i*100):
                poly_reg_l2[i].update_parameters()



        print('For validation set')
        for i in range(16):
            y_pred = []
            y_pred_l1 = []
            y_pred_l2 = []
            for x, y in zip(x_valid, y_valid):
                y_pred.append(poly_reg[i].predict(x))
                y_pred_l1.append(poly_reg_l1[i].predict(x))
                y_pred_l2.append(poly_reg_l2[i].predict(x))


            print(f'MSE for without regularisation in {i+5}th degree polynomial: {rm.MSE(y_valid, y_pred)}')
            print(f'MSE for l1 regularisation in {i+5}th degree polynomial: {rm.MSE(y_valid, y_pred_l1)}')
            print(f'MSE for l2 regularisation in {i+5}th degree polynomial: {rm.MSE(y_valid, y_pred_l2)}')

            print(f'variance for without regularisation in {i+5}th degree polynomial: {rm.variance(y_pred)}')
            print(f'variance for l1 regularisation in {i+5}th degree polynomial: {rm.variance(y_pred_l1)}')
            print(f'variance for l2 regularisation in {i+5}th degree polynomial: {rm.variance(y_pred_l2)}')

            print(f'standard deviation for without regularisation in {i+5}th degree polynomial: {rm.standard_deviation(y_pred)}')
            print(f'standard deviation for l1 regularisation in {i+5}th degree polynomial: {rm.standard_deviation(y_pred_l1)}')
            print(f'standard deviation for l2 regularisation in {i+5}th degree polynomial: {rm.standard_deviation(y_pred_l2)}')


        print('For test set')
        for i in range(16):
            y_pred = []
            y_pred_l1 = []
            y_pred_l2 = []
            for x, y in zip(x_test, y_test):
                y_pred.append(poly_reg[i].predict(x))
                y_pred_l1.append(poly_reg_l1[i].predict(x))
                y_pred_l2.append(poly_reg_l2[i].predict(x))


            print(f'MSE for without regularisation in {i+5}th degree polynomial: {rm.MSE(y_test, y_pred)}')
            print(f'MSE for l1 regularisation in {i+5}th degree polynomial: {rm.MSE(y_test, y_pred_l1)}')
            print(f'MSE for l2 regularisation in {i+5}th degree polynomial: {rm.MSE(y_test, y_pred_l2)}')

            print(f'variance for without regularisation in {i+5}th degree polynomial: {rm.variance(y_pred)}')
            print(f'variance for l1 regularisation in {i+5}th degree polynomial: {rm.variance(y_pred_l1)}')
            print(f'variance for l2 regularisation in {i+5}th degree polynomial: {rm.variance(y_pred_l2)}')

            print(f'standard deviation for without regularisation in {i+5}th degree polynomial: {rm.standard_deviation(y_pred)}')
            print(f'standard deviation for l1 regularisation in {i+5}th degree polynomial: {rm.standard_deviation(y_pred_l1)}')
            print(f'standard deviation for l2 regularisation in {i+5}th degree polynomial: {rm.standard_deviation(y_pred_l2)}')

        
        for i in range(16):
            y_pred = []
            y_pred_l1 = []
            y_pred_l2 = []
            for x,y in zip(x_train, y_train):
                y_pred.append(poly_reg[i].predict(x))
                y_pred_l1.append(poly_reg_l1[i].predict(x))
                y_pred_l2.append(poly_reg_l2[i].predict(x))

                    
            mse = rm.MSE(y_train, y_pred)
            var = rm.variance(y_pred)
            std = rm.standard_deviation(y_pred)

            mse_l1 = rm.MSE(y_train, y_pred_l1)
            var_l1 = rm.variance(y_pred_l1)
            std_l1 = rm.standard_deviation(y_pred_l1)

            mse_l2 = rm.MSE(y_train, y_pred_l2)
            var_l2 = rm.variance(y_pred_l2)
            std_l2 = rm.standard_deviation(y_pred_l2)

            x_sorted, y_sorted = zip(*sorted(zip(x_train, y_pred)))
            x_sorted_l1, y_sorted_l1 = zip(*sorted(zip(x_train, y_pred_l1)))
            x_sorted_l2, y_sorted_l2 = zip(*sorted(zip(x_train, y_pred_l2)))

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)

            plt.scatter(x_train, y_train, color = 'red')
            plt.plot(x_sorted, y_sorted, color = 'blue')
            plt.text(0.5, -0.2, f'MSE: {mse:.3f}\nVariance: {var:.4f}\nSTD: {std:.1f}', 
                fontsize=10, ha='center', va='top', transform=plt.gca().transAxes)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Polynomial regression of degree'  + str(i+5) + ' without regularization')

            plt.subplot(1, 3, 2)
            plt.scatter(x_train, y_train, color = 'red')
            plt.plot(x_sorted_l1, y_sorted_l1, color = 'blue')
            plt.text(0.5, -0.2, f'MSE: {mse_l1:.3f}\nVariance: {var_l1:.4f}\nSTD: {std_l1:.1f}', 
                fontsize=10, ha='center', va='top', transform=plt.gca().transAxes)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Polynomial regression of degree' + str(i+5) + ' with l1 regularization')


            plt.subplot(1, 3, 3)
            plt.scatter(x_train, y_train, color = 'red')
            plt.plot(x_sorted_l2, y_sorted_l2, color = 'blue')
            plt.text(0.5, -0.2, f'MSE: {mse_l2:.3f}\nVariance: {var_l2:.4f}\nSTD: {std_l2:.1f}', 
                fontsize=10, ha='center', va='top', transform=plt.gca().transAxes)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Polynomial regression of degree' + str(i+5) + ' with l2 regularization' )

            plt.tight_layout()
            plt.savefig(f'figures/regularisation_fig/degree_{i+5}.png')
            plt.close()
