import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Tells MLflow where to store & read experiment data
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("MLOPS-Exp-1")

# Load Wine dataset
# Built-in sklearn Wine Multi-class classification (3 wine types) dataset
wine = load_wine()
X = wine.data
y = wine.target


# Train test split
# 90% training, 10% testing random_state = 42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)


# Define the params for RF model
# experiment parameters Logged to MLflow so you can compare runs later 
# This is where MLflow shines when tuning models.
max_depth = 10    #t otal number of decision trees in the ensemble model.
n_estimators = 5  # Controls the maximum number of levels (splits) a single decision tree is allowed to grow


# Mention your experiment below
# experiment is a container for multiple runs
#mlflow.set_experiment('MLOPS-Exp-1')
#mlflow.set_tracking_uri("file:./mlruns")

mlflow.set_experiment('MLOPS-Exp-1')

# start an MLflow Run

with mlflow.start_run():

    # Model Training
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Model Evaluation 
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log Metrics
    mlflow.log_metric('accuracy', accuracy)

    # Log Parameters
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)


    # Creating a confusion matrix plot
    # Rows are Actual labels & Columns are Predicted labels
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot as Artifact
    plt.savefig('Confusion-matrix-1.png')

    # log artifacts using mlflow
    mlflow.log_artifact('Confusion-matrix-1.png')
    mlflow.log_artifact(__file__)

    # tags helps in filtering runs, organizing experiments, and adding business context
    mlflow.set_tags({"Author": 'Varshith', "Task": "Wine Classification"})

    # Log the model- Saves the trained model, and model metadata
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    
    # Just console output — not tracked, unless logged explicitly.
    print(accuracy)