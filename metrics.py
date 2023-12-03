import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from app import get_model


print("Loading Models and datasets...")
# load the Models:
lg_model = get_model("logisticRegression")
nn_model = get_model("simpleNeuralNetwork")
nb_model = get_model("naiveBayes")
knn_model = get_model("kNN")

# Retrieve Scalers
standard_scaler = get_model("std_scaler")
norm_scaler = get_model("norm_scaler")


# Load dataset
train_df = pd.read_csv("./dataset/train_data.csv")
test_df = pd.read_csv("./dataset/test_data.csv")

X_train = train_df.iloc[:,:-1].values
y_train = train_df.iloc[:,13].values

X_test = test_df.iloc[:,:-1].values
y_test = test_df.iloc[:,13].values

X_test_scaled = standard_scaler.transform(X_test)
X_test_normalized = norm_scaler.transform(X_test)

print("Finished loading models and dataset")

def print_report(y_test, y_preds):
    print("===============================================================")
    labels = ['without heart disease', 'with heart disease']
    print(classification_report(y_test, y_preds, target_names=labels, zero_division=1))


    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds)
    recall = recall_score(y_test, y_preds)
    class_names=[1,0]
    conf_matrix = confusion_matrix(y_test, y_preds, labels=class_names)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("===============================================================")



def main(user_input):
    # user_iput == 0 -> LogisticRegression lg_model
    if user_input == 0:
        y_preds =  lg_model.predict(X_test_scaled)
        print_report(y_test, y_preds)
    # user_iput == 1 -> Knn knn_model
    elif user_input == 2:
        y_preds =knn_model.predict(X_test_scaled)
        print_report(y_test, y_preds)
    
    # user_iput == 2 -> NeuralNetwork nn_model
    elif user_input == 2:
        y_preds = nn_model.predict(X_test_normalized)
        predicted_labels = (y_preds > 0.7).astype(int)
        print_report(y_test, predicted_labels)
    
    else:
        y_preds = nb_model.predict(X_test_normalized)
        print_report(y_test, y_preds)


if __name__ == "__main__":
    valid_inputs = [0, 1, 2, 3]
    
    while True:

        print("Classification Models: \n 0: Logistic Regression \n 1: K-nearest neighbor \n 2: Simple Neural Network \n 3: Naive Bayes")
        user_input = input("Please enter 0, 1, 2, 3, or 'Q' to quit: ")
        
        if user_input.upper() == 'Q':
            print("Exiting the program.")
            break
        
        try:
            user_input = int(user_input)
            
            if user_input in valid_inputs:
                main(user_input)
            else:
                print("--->>>Invalid input. Please enter 0, 1, 2, 3, or 'Q' to quit.")
        except ValueError:
            print("--->>>Invalid input. Please enter a number or 'Q' to quit.")
    
    if user_input.upper() != 'Q':
        # Now user_input contains a valid input (0, 1, 2, or 3)
        print("You entered:", user_input)
        main(user_input)
