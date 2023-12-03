import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tensorflow import keras
import pickle


def load_data():
    return pd.read_csv(os.path.join("dataset", "heart.csv"))

def get_split_dataset():
    heart_data = load_data()

    X = heart_data.iloc[:,:-1].values
    y = heart_data.iloc[:,13].values

    # split dataset 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # saving the training and test dataset
    train_df = pd.DataFrame(X_train, columns=heart_data.columns[:-1])
    train_df['target'] = y_train  # Assuming 'target' is the column name for the target variable

    test_df = pd.DataFrame(X_test, columns=heart_data.columns[:-1])
    test_df['target'] = y_test  # Assuming 'target' is the column name for the target variable

    # Save DataFrames as CSV files
    train_df.to_csv('./dataset/train_data.csv', index=False)
    test_df.to_csv('./dataset/test_data.csv', index=False)


    return X_train, X_test, y_train, y_test


def get_trained_logistic_regression(X_train, y_train):
    model_filename = "./models/Logistic_Regression.pkl"

    print("Training Logistic Regression Model..")
    
    log_reg_model = LogisticRegression(max_iter=1000)
    log_reg_model.fit(X_train, y_train)

    with open(model_filename, 'wb') as file:
        pickle.dump(log_reg_model, file)

    print(f"Saving LogisticRegression Model in {model_filename}")
    return log_reg_model


def get_trained_knn_model(X_train, y_train):
    model_filename = "./models/knn.pkl"
    print("Training kNN Model..")
    knn_euclidean_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_euclidean_classifier.fit(X_train, y_train)

    with open(model_filename, 'wb') as file:
        pickle.dump(knn_euclidean_classifier, file)

    print(f"Saving kNN Model in {model_filename}")
    return knn_euclidean_classifier



def get_trained_naive_bayes_model(X_train, y_train):
    model_filename = "./models/naive_bayes.pkl"
    print("Training Naive Bayes Model..")
    naive_bayes_model = GaussianNB()

    naive_bayes_model.fit(X_train, y_train)

    with open(model_filename, 'wb') as file:
        pickle.dump(naive_bayes_model, file)

    print(f"Saving Naive Bayes Model in {model_filename}")
    return naive_bayes_model


def get_trained_neural_network_model(X_train, y_train):
    model_filename = "./models/neural_network.pkl"
    print("Training Neural Network Model..")
    num_features = X_train.shape[1]
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=20, batch_size=20, validation_split=0.1)

    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"Saving Neural Network Model in {model_filename}")
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_split_dataset()

    standard_scaler = StandardScaler()
    X_train_scaled = standard_scaler.fit_transform(X_train)
    X_test_scaled = standard_scaler.transform(X_test)

    # inputs need to be standardized
    knn_model = get_trained_knn_model(X_train_scaled, y_train)
    lr_model = get_trained_logistic_regression(X_train, y_train)
    
    # Normalized inputs
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)

    nb_model = get_trained_naive_bayes_model(X_train_normalized, y_train)
    
    nn_model = get_trained_neural_network_model(X_train_normalized, y_train)

    # Save the standard_scaler
    with open('./models/std_scaler.pkl', 'wb') as std_file:
        pickle.dump(standard_scaler, std_file)

    # Save the normalization scaler
    with open('./models/norm_scaler.pkl', 'wb') as norm_file:
        pickle.dump(scaler, norm_file)

    print("Saved scalers...")
    print("Training required models is Done!")


