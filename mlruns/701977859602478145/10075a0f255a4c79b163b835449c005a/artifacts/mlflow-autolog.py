import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment('MLOPS-exp-3-mlflow-autologging')

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 10
min_samples_leaf = 15 # defines the minimum number of samples required to be at a leaf node
max_features='sqrt' # the maximum number of features random forest considers to split a node.

# Mention your experiment below
mlflow.autolog()


with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, min_samples_leaf= min_samples_leaf, max_features= max_features, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    #Log Metrics
    # mlflow auto logs the metrics

    # Log Parameters
    # mlflow auto logs the Parameters

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # mlflow auto logs artifacts like Confusion-matrix.png
    
    # to save the code/ py script we need to mention it explicitly
    mlflow.log_artifact(__file__)

    # to save the tags we need to mention it explicitly
    mlflow.set_tags({"Author": 'Varshith', "Project": "Wine Classification"})

    print('accuracy -- ',accuracy)