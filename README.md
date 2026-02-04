# MLFLOW
MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. It also helps in MLOps by using tools like experiment tracking, model packaging, and serving &amp; deployment, making the ML process more reproducible


- MLflow Tracking: An API and UI for logging parameters, code versions, metrics, and output files (artifacts) when running your ML code.
- MLflow Models: A standard format for packaging models in "flavors" (e.g., scikit-learn, TensorFlow, PyTorch) that can be understood by diverse deployment tools.
- MLflow Model Registry: A centralized model store for collaborative lifecycle management, featuring version control, annotations, and stage transitions (e.g., Staging to Production).
- MLflow Projects: A format for packaging data science code in a reusable and reproducible way, often using a simple MLproject file to define dependencies
- Scalability: MLFlow works with libraries like TensorFlow and PyTorch. It supports large-scale tasks with distributed computing. It also integrates with cloud storage for added flexibility.

-----------------------------
## Setting Up MLFlow
 

Installation -- <br>
To get started, install MLFlow using pip:
~~~
pip install mlflow
~~~
<br>

Running the Tracking Server  -- <br>
To set up a centralized tracking server, run:
~~~
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
~~~ 
This command uses an SQLite database for metadata storage and saves artifacts in the mlruns directory.

<br>

Launching the MLFlow UI -- <br>
The MLFlow UI is a web-based tool for visualizing experiments and models. You can launch it locally with:
~~~
mlflow ui
~~~ 
By default, the UI is accessible at http://localhost:5000.

<br>

In case of any error run below command and then rerun mlfow ui 
~~~
pip install "jinja2<3.1" "flask<3.0"
~~~
Now you can see the browser running and accessible at http://localhost:5000

-----------------------------
# Key Components of MLFlow

### **1. MLFlow Tracking** <br>

Experiment tracking is at the heart of MLflow. It enables teams to log:<br>

Parameters: Hyperparameters used in each model training run.<br>
Metrics: Performance metrics such as accuracy, precision, recall, or loss values.<br>
Artifacts: Files generated during the experiment, such as models, datasets, and plots.<br>
Source Code: The exact code version used to produce the experiment results.<br>

~~~
import mlflow

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

    # Log artifacts
    with open("model_summary.txt", "w") as f:
        f.write("Model achieved 95% accuracy.")
    mlflow.log_artifact("model_summary.txt")
~~~

<br>

### **2. MLFlow Projects** <br>
MLflow Projects enable reproducibility and portability by standardizing the structure of ML code. A project contains:<br>

Source code: The Python scripts or notebooks for training and evaluation.<br>
Environment specifications: Dependencies specified using Conda, pip, or Docker.<br>
Entry points: Commands to run the project, such as train.py or evaluate.py.<br>

~~~
name: my_ml_project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      data_path: {type: str, default: "data.csv"}
      epochs: {type: int, default: 10}
    command: "python train.py --data_path {data_path} --epochs {epochs}"
~~~

<br>

### **3. MLFlow Models** <br>

MLFlow Models manage trained models. They prepare models for deployment. Each model is stored in a standard format. <br>
This format includes the model and its metadata. Metadata has the model's framework, version, and dependencies. <br>
MLFlow supports deployment on many platforms. This includes REST APIs, Docker, and Kubernetes. It also works with cloud services like AWS SageMaker.

~~~
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Train and save a model
model = RandomForestClassifier()
mlflow.sklearn.log_model(model, "random_forest_model")

# Load the model later for inference
loaded_model = mlflow.sklearn.load_model("runs://random_forest_model")
~~~

<br>

### **4. MLFlow Model Registry** <br>
The Model Registry tracks models through the following lifecycle stages: <br>

Staging: Models in testing and evaluation. <br>
Production: Models deployed and serving live traffic. <br>
Archived: Older models preserved for reference. <br>

The registry helps teams work together. It keeps track of different model versions. <br>
It also manages the approval process for moving models forward.
~~~
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register a new model
model_uri = "runs://random_forest_model"
client.create_registered_model("RandomForestClassifier")
client.create_model_version("RandomForestClassifier", model_uri, "Experiment1")

# Transition the model to production
client.transition_model_version_stage("RandomForestClassifier", version=1, stage="Production")
~~~

<br>

------------------

## Real-World Use Cases
 
1. Hyperparameter Tuning: Track hundreds of experiments with different hyperparameter configurations to identify the best-performing model.<br>
2. Collaborative Development: Teams can share experiments and models via the centralized MLflow tracking server.<br>
3. CI/CD for Machine Learning: Integrate MLflow with Jenkins or GitHub Actions to automate testing and deployment of ML models.<br>

## Best Practices for MLFlow

1. Centralize Experiment Tracking: Use a remote tracking server for team collaboration.<br>
2. Version Control: Maintain version control for code, data, and models.<br>
3. Standardize Workflows: Use MLFlow Projects to ensure reproducibility.<br>
4. Monitor Models: Continuously track performance metrics for production models.<br>
5. Document and Test: Keep thorough documentation and perform unit tests on ML workflows.<br>
