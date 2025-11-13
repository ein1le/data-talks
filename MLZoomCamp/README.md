# MLZoomCamp Midterm Project 

Tags:
__Multiclass Classification • XGBoost & Logistic Regression • Flask Web Service • Docker Deployment__

## Overview 

This project is the midterm homework submission for **Alexey Grigorev’s 2025 MLZoomCamp**, covering content from weeks 1–6.

It demonstrates a complete machine learning workflow for a small scale, static service, including:
- Training Logistic Regression and XGBoost multiclass classifiers 
- Validation using Cross-fold validation to inform model selection and hyperparameter tuning
- Manual preprocessing pipelines
- Prediction service served through Flask API
- Static HTML/CSS/JS Frontend supporting JSON POST requests
- Containerized deployment using Docker
- Reproducible environments

The underlying dataset is retrieved from Kaggle:
https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation
Which classifies customers into 4 types (A, B, C, D).

## Repository Structure
```bash
data/
  └── Train.csv                # Dataset downloaded from Kaggle

model_registry/
  └── *.pkl                    # Serialized trained models (logreg + xgb)

scripts/
  ├── helper.py                # Preprocessing helpers for both models
  ├── predict.py               # Loads model registry, transforms JSON → DF → predictions
  ├── train_logistic.py        # CLI training script for Logistic Regression
  └── train_xgb.py             # CLI training script for XGBoost

web_service/
  ├── app.py                   # Flask server (predict + health routes)
  ├── index.html               # Browser UI
  ├── styles.css               # Visual styling
  ├── app.js                   # Logic for sending prediction requests
  └── wsgi.py                  # Entry point for Waitress/Gunicorn

dockerfile                      # For reproducible deployment
.dockerignore                   # Exclude unnecessary files from image
.gitignore                      # Ignore venv, pycache, etc.
README.md                       # This document
requirements.txt                # Python dependencies
testing-notebook.ipynb         # EDA, experimentation, performance evaluation

```

The `testing-notebook.ipynb` is a Jupyter notebook used to experiment with the data and models. The content includes:
- Exploratory data analysis
- Distribution and correlation analysis
- Model training and evaluation
- Training and Validation performance analysis
- Heatmaps and hyperparameter visualizations

## Virtual Environment 

This workflow assumes you are using uv, the modern Python package + environment manager.

> `uv` is a fast, modern alternative to `pip` and `venv`, but any standard Python virtual environment also works.

**1. Create and activate virtual environment**

In the project root, run:
```bash
uv venv
```
To create a deddicated environment in `.venv`

Then activate the environment:
```bash
.venv\Scripts\activate.bat
```
Once activated, install dependencies from `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

## Training 

Training can be done via the command line, after cding to the `scripts` directory.

```bash
cd scripts

python train_logistic.py ../data/Train.csv #Train Logistic Regression
python train_xgb.py ../data/Train.csv #Train XGBoost
```
Both training scripts can be run using the command above, with a required dataset path as an argument. Running these scripts will follow this general process:

1. Load CSV
2. Split: 80% Train, 10% Validation, 10% Test
3. Fit preprocessing on training data
4. Run cross-validated hyperparameter search
5. Refit best model on Train + Validation
6. Save .pkl artifact to `model_registry/`

All trained artifacts are serialized by date and timestamp, and saved to the `model_registry/` directory.
`scripts/predict.py` automatically loads the newest `.pkl` artifact from `model_registry/` based on the timestamp embedded in the filename.

#### Logistic Regression Flags

| Flag                        | Description                                   | Default              |
|-----------------------------|-----------------------------------------------|----------------------|
| `--temp-size`               | Fraction of data reserved for val+test        | `0.2`                |
| `--val-ratio-within-temp`   | Split within temp between val/test            | `0.5`                |
| `--cv-folds`                | StratifiedKFold splits                        | `5`                  |
| `--random-state`            | Random seed                                   | `42`                 |
| `--c-min-exp`               | log10 lower bound of C                        | `-3`                 |
| `--c-max-exp`               | log10 upper bound of C                        | `2`                  |
| `--c-num`                   | Number of C values to test                    | `12`                 |
| `--solver`                  | Logistic regression solver                    | `lbfgs`              |
| `--penalty`                 | Regularization penalty                         | `l2`                 |
| `--max-iter`                | Maximum iterations                             | `1000`               |
| `--multi-class`             | Force multinomial mode                         | `None`               |
| `--scoring`                 | CV metric                                      | `balanced_accuracy`  |
| `--registry-dir`            | Output directory for model artifact            | `../model_registry`  |
| `--model-prefix`            | Prefix for saved model name                    | `logreg`             |


#### XGBoost Flags


| Flag                        | Description                                   | Default              |
|-----------------------------|-----------------------------------------------|----------------------|
| `--temp-size`               | Fraction of data reserved for val+test        | `0.2`                |
| `--val-ratio-within-temp`   | Validation ratio inside temp split            | `0.5`                |
| `--cv-folds`                | StratifiedKFold splits                        | `5`                  |
| `--random-state`            | Random seed                                   | `42`                 |
| `--scoring`                 | CV scoring metric                             | `accuracy`           |
| `--max-depths`              | List of tree depths                           | `3,4,5,6`            |
| `--min-child-weight`        | List of min_child_weight values               | `1,3,5`              |
| `--learning-rates`          | Learning rate search grid                     | `0.03,0.05,0.1`      |
| `--n-estimators`            | Tree counts                                   | `200,400,600`        |
| `--reg-lambda`              | L2 regularization grid                        | `1.0,2.0`            |
| `--reg-alpha`               | L1 regularization grid                        | `0.0,0.5`            |
| `--registry-dir`            | Output directory for model artifact            | `../model_registry`  |
| `--model-prefix`            | Prefix for saved model name                    | `xgb`                |


## Web Servicing

This project includes a lightweight web service that allows users to submit JSON data through a browser and receive predictions from either the Logistic Regression or XGBoost model trained earlier. 

The service is intentionally minimal and works fully locally, however supports portability via Docker.

### Architecture
**Backend**
Located in /web_service/app.py, this backend:

- Loads the most recent model artifacts from /model_registry/
- Accepts:
  - POST requests to /predict
  - GET requests to /health for diagnostics

- Performs preprocessing using logic from `scripts/predict.py`
- Runs inference and returns a JSON response

> `waitress` was used over `gunicorn` for simplicity and ability to deploy WSGI-grade applications on Windows.

**Frontend**
Located in /web_service/:
The Browser sends a JSON request to Flask, which calls the `/predict` endpoint, loading an appropriate model and performing inference.

`index.html` — user interface with a JSON input box and a model selector

`styles.css` — layout and presentation (modern card-style UI)

`app.js` — handles frontend logic:

- Reads JSON from the textarea
- Validates it client-side
- Sends a POST request to /predict
- Renders prediction results in formatted HTML

The result is an entirely self-contained UI that works in any browser with no frameworks.


#### Endpoints

- `/health` (GET) Checks registry and loaded models
- `/predict` (POST) Makes a prediction by calling a saved model
- `/` (GET) Root, serves `index.html`

#### Execution
**Option 1: Direct Flask Development Server**
Running the command from the terminal:
```bash
python web_service/app.py
```

Deploys a local web service that can be accessed at `http://localhost:8000`


**Option 2: Waitress Server**
Recommended for stable production-grade serving on Windows/Linux/MacOS:

```bash
python -m waitress --listen=0.0.0.0:8000 web_service.app:app
```
**Option 3: Docker**

Running the project on `Dockerfile` ensures consistent environments.

The Dockerfile copies the web service, prediction scripts, and model registry into the container, installs dependencies, and launches Waitress as the WSGI server.

Model registry remains mountable and all environments are reproducible and deployablel on cloud.

To get started, build the Docker Image using:
```bash
docker build -t ml-zoomcamp-midterm .
```

Then run the container using:
```bash
docker run -p 8000:8000 ml-zoomcamp-midterm
```

This exposes the app on `http://localhost:8000` or `http://0.0.0.0:8000` on WSL or VMs.


## Predictions 

The web interface allows you to manually enter a JSON object representing a customer, select a machine-learning model, and retrieve a predicted customer segment along with associated probabilities.

The model predicts customer segments as one of four classes, `A, B, C, D` which correpsond to the original dataset target variable.

#### How to use the Web Predictor
1. Open your browser and navigate to 
```
http://localhost:8000
```
2. In the JSON Input Box, provide a single JSON object representing the customer profile you want to predict.

3. Choose the model from the dropdown list
4. Click the "Predict" button


After the Prediction, the UI will display a predicted segment, along with associated probabilities for each segment.

**Example**
```
A: 0.70
B: 0.10
C: 0.05
D: 0.15
```
This means the model estimates the customer has a 70% likelihood of belonging to segment `A`, with other classes significantly less likely.

> Probabilities always sum to 1.00.

Example JSON Input:
```json
{
  "Gender": "Male",
  "Ever_Married": "Yes",
  "Graduated": "No",
  "Age": 35,
  "Work_Experience": 7,
  "Family_Size": 3,
  "Spending_Score": "Average",
  "Profession": "Artist",
  "Var_1": "Cat_3"
}
```

## License

This project is built solely for educational purposes as part of the MLZoomCamp 2025 midterm assignment.
