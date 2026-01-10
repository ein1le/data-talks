# MLZoomCamp – Capstone Project

This project is a final capstone submission for Alex Gregoriev's MLZoomCamp.
It uses a pre-trained [EfficientNetB0](https://arxiv.org/abs/1905.11946) model trained [Kaggle's Vegetable Image dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) to predict images of vegetables into 15 classes.  
EDA and accelerted GPU training was done in the Kaggle notebook and the output model was exported as a SavedModel and served via TensorFlow Serving. A Python gateway exposes a simple `/predict` HTTP endpoint and a minimal HTML page for image uplaods and returns a predicted vegetable class with probabilities.
Both Local and Cloud-native Kubernetes deployments are supported.

Model details:

- Base architecture: **EfficientNetB0**
- Training: transfer learning on a vegetable image dataset
- Output: probability vector over **15 classes** (Bean, Bitter_Gourd, Bottle_Gourd, Brinjal,
  Broccoli, Cabbage, Capsicum, Carrot, Cauliflower, Cucumber, Papaya, Potato, Pumpkin, Radish,
  Tomato)
- Export: TensorFlow **SavedModel** (under `model/v1`) with signature:
  - Input tensor: `image` – shape `(None, 224, 224, 3)`, `float32`
  - Output tensor: `output_0` – shape `(None, 15)`, `float32` probabilities

The `class_names.json` file holds the ordered list of vegetable labels that are aligned with
the `output_0` indices, so prediction index `i` is mapped to `class_names[i]`.

---

## Project Structure

Project root (this folder):

- `mlzoomcamp-capstone.ipynb` – original notebook for training, exporting, and inspecting the model
- `requirements.txt` – Python dependencies for local (non‑Docker) development
- `docker-compose.yml` – local multi‑container setup (gateway + TensorFlow Serving model)

Model artifacts:

- `model/` – model assets and metadata
  - `class_names.json` – ordered list of 15 vegetable class names
  - `v1/` – TensorFlow SavedModel export (version 1 of the model)
    - `saved_model.pb` – main SavedModel graph
    - `fingerprint.pb` – SavedModel fingerprint
    - `assets/` – auxiliary assets used by the SavedModel (if any)
    - `variables/` – trained weights for the model

Application code:

- `scripts/` – gateway application and client logic
  - `settings.py` – central configuration (paths, image size, TF Serving host/port, etc.)
  - `preprocessing.py` – image preprocessing helpers (resize + normalization to [0, 1])
  - `predict.py` – gRPC client for TensorFlow Serving; uses preprocessing helpers and maps outputs to class names
  - `gateway.py` – Flask HTTP API that serves the HTML page and exposes `/predict` for image uploads

Frontend:

- `frontend/` – static files for the simple web UI
  - `index.html` – single‑page interface with file upload and “Predict” button

Container and Kubernetes config:

- `config/` – Dockerfiles and Kubernetes manifests
  - `Dockerfile.model` – builds the TensorFlow Serving image with the SavedModel mounted at `/models/vegetable-model/1`
  - `Dockerfile.gateway` – builds the Flask gateway + gRPC client image and bundles the frontend
  - `eks-config.yaml` – `eksctl` cluster configuration (EKS cluster definition)
  - `model-deployment.yaml` – Kubernetes Deployment for the TensorFlow Serving model
  - `model-service.yaml` – Kubernetes Service (ClusterIP) for the model (gRPC on port 8500)
  - `gateway-deployment.yaml` – Kubernetes Deployment for the Flask gateway
  - `gateway-service.yaml` – Kubernetes Service (LoadBalancer) exposing the gateway over HTTP

---

## Local Development

You have three main ways to run this project locally:

1. Using Docker Compose (recommended for quick end‑to‑end tests)
2. Using a Python virtual environment (venv) and a separate TF Serving container
3. Using a local Kubernetes cluster (kind)

All commands below assume you are in the project root, e.g. (PowerShell on Windows):

```bash
cd "D:\Education\Alexy\data-talks\MLZoomCamp - Capstone"
```

### 1. Local with Docker Compose

Prerequisites:

- Docker (with Docker Compose support)

Start both the model and gateway containers:

```bash
docker-compose up --build
```

What this does:

- Builds `config/Dockerfile.model` as the `model` service, exposing gRPC on `localhost:8500`
- Builds `config/Dockerfile.gateway` as the `gateway` service, exposing HTTP on `localhost:5000`
- Configures the gateway to call the model via `TF_SERVING_HOST=model` and `TF_SERVING_PORT=8500`

Open the app:

- Navigate to `http://localhost:5000/` in your browser
- Upload an image of a vegetable and click **Predict**
- The page will show a “Top prediction” plus the list of top classes with probabilities

Stop the stack:

```bash
docker-compose down
```

### 2. Local with Python venv + TF Serving

This option runs the gateway directly on your host Python, while using a separate TF Serving
container for the model.

Prerequisites:

- Python 3.11+ (or compatible)
- Docker (for TF Serving)

#### 2.1 Create and activate a virtual environment

From the project root:

```bash
python -m venv .venv

# PowerShell (Windows):
.\.venv\Scripts\Activate.ps1

# or on Unix/macOS:
# source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 2.2 Run TensorFlow Serving for the model

Use the official TensorFlow Serving image and mount your SavedModel directory:

```bash
docker run --rm \
  -p 8500:8500 \
  -e MODEL_NAME=vegetable-model \
  -v "%cd%\model\v1:/models/vegetable-model/1" \
  tensorflow/serving:2.13.0
```

On Unix/macOS you’d use:

```bash
docker run --rm \
  -p 8500:8500 \
  -e MODEL_NAME=vegetable-model \
  -v "$(pwd)/model/v1:/models/vegetable-model/1" \
  tensorflow/serving:2.13.0
```

This exposes gRPC at `localhost:8500`.

#### 2.3 Run the gateway locally

With TF Serving running in one terminal and your venv active in another:

```bash
set TF_SERVING_HOST=localhost
set TF_SERVING_PORT=8500
set TF_SERVING_MODEL_NAME=vegetable-model
set GATEWAY_PORT=5000

python scripts/gateway.py
```

(On Unix/macOS use `export` instead of `set`.)

Then open `http://localhost:5000/` in your browser as before.

### 3. Local Kubernetes with kind

You can also simulate the Kubernetes deployment locally using [kind](https://kind.sigs.k8s.io/).

Prerequisites:

- `kind` installed
- `kubectl` configured to use kind
- Docker (for cluster nodes and images)

#### 3.1 Create a kind cluster

```bash
kind create cluster --name capstone
```

Verify:

```bash
kubectl cluster-info --context kind-capstone
```

#### 3.2 Build images and load them into kind

Since kind runs its own Docker daemon inside the Kubernetes nodes, you typically need to build
the images locally and then load them into the kind cluster:

```bash
# Build images locally
docker build -f config/Dockerfile.model -t mlzoomcamp-capstone-model:latest .
docker build -f config/Dockerfile.gateway -t mlzoomcamp-capstone-gateway:latest .

# Load them into the kind cluster
kind load docker-image mlzoomcamp-capstone-model:latest --name capstone
kind load docker-image mlzoomcamp-capstone-gateway:latest --name capstone
```

#### 3.3 Apply Kubernetes manifests

Apply the existing manifests in `config/`:

```bash
kubectl apply -f config/model-deployment.yaml
kubectl apply -f config/model-service.yaml
kubectl apply -f config/gateway-deployment.yaml
kubectl apply -f config/gateway-service.yaml
```

The `gateway-service.yaml` uses `type: LoadBalancer`, which kind does not implement as a real
cloud load balancer. Instead, use port‑forwarding for local access:

```bash
kubectl port-forward service/gateway 8080:80
```

Then open:

- `http://localhost:8080/` – frontend
- `http://localhost:8080/predict` – API endpoint (if calling directly)

When done, you can delete the cluster:

```bash
kind delete cluster --name capstone
```

---

## Cloud Deployment (AWS)

For cloud deployment on AWS, you typically use:

- **Amazon ECR** – container registry that stores your images
- **Amazon EKS** – managed Kubernetes cluster that runs your pods

Below is a high‑level workflow; adapt AWS account IDs, regions, and names as needed.

### 1. Set up ECR registries

Assuming region `eu-west-1` and account ID `123456789012`:

```bash
aws ecr create-repository --repository-name mlzoomcamp-capstone-model
aws ecr create-repository --repository-name mlzoomcamp-capstone-gateway
```

This gives you ECR URLs like:

- `123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-model`
- `123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-gateway`

Log Docker into ECR:

```bash
aws ecr get-login-password --region eu-west-1 |
  docker login --username AWS --password-stdin 123456789012.dkr.ecr.eu-west-1.amazonaws.com
```

### 2. Build and push images

```bash
# Build
docker build -f config/Dockerfile.model -t mlzoomcamp-capstone-model:latest .
docker build -f config/Dockerfile.gateway -t mlzoomcamp-capstone-gateway:latest .

# Tag
docker tag mlzoomcamp-capstone-model:latest \
  123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-model:latest

docker tag mlzoomcamp-capstone-gateway:latest \
  123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-gateway:latest

# Push
docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-model:latest
docker push 123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-gateway:latest
```

Update your Kubernetes Deployments to use these ECR images:

- In `config/model-deployment.yaml`:
  ```yaml
  image: 123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-model:latest
  ```
- In `config/gateway-deployment.yaml`:
  ```yaml
  image: 123456789012.dkr.ecr.eu-west-1.amazonaws.com/mlzoomcamp-capstone-gateway:latest
  ```

### 3. Create an EKS cluster (eksctl)

The file `config/eks-config.yaml` defines a simple EKS cluster configuration. Create it with:

```bash
eksctl create cluster -f config/eks-config.yaml
```

This will provision an EKS cluster (and node group) in your AWS account. Ensure the node group
role has permission to pull from ECR (e.g. attach `AmazonEC2ContainerRegistryReadOnly`).

### 4. Deploy to EKS

With your `kubectl` context pointing to the new EKS cluster:

```bash
kubectl apply -f config/model-deployment.yaml
kubectl apply -f config/model-service.yaml
kubectl apply -f config/gateway-deployment.yaml
kubectl apply -f config/gateway-service.yaml
```

The `gateway` Service is of type `LoadBalancer`, so AWS will provision an external load balancer.
Get the external hostname or IP:

```bash
kubectl get service gateway
```

Then access:

- `http://<EXTERNAL_HOSTNAME>/` – frontend
- `http://<EXTERNAL_HOSTNAME>/predict` – API endpoint

---

## Usage

Once the gateway is running (locally via venv, Docker Compose, kind, or on EKS), the basic usage
is the same.

### Web UI

1. Open the gateway URL in a browser:
   - Local venv or Docker Compose: `http://localhost:5000/`
   - kind port‑forward: `http://localhost:8080/`
   - EKS LoadBalancer: `http://<EXTERNAL_HOSTNAME>/`
2. Use the file picker to upload an image of a vegetable (JPEG/PNG, etc.).
3. Click **Predict**.
4. The page displays:
   - **Top prediction** – best matching vegetable class with probability
   - **Predictions list** – top classes and their probabilities

### HTTP API

You can also call the `/predict` endpoint directly from scripts or other services.

Example with `curl` (local gateway on port 5000):

```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@path/to/your/vegetable.jpg"
```

Example JSON response:

```json
{
  "top_prediction": {
    "class": "Tomato",
    "probability": 0.97
  },
  "predictions": [
    { "class": "Tomato", "probability": 0.97 },
    { "class": "Pumpkin", "probability": 0.02 },
    { "class": "Papaya", "probability": 0.01 }
  ]
}
```

Probabilities are returned as floats in `[0, 1]`; the frontend converts them to percentages.
