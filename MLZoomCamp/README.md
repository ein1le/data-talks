# MLZoomCamp Scaffold

This directory now contains a placeholder structure for a containerized machine learning service using Flask and a lightweight front end. Each artifact is empty by design so that actual code can be added later without conflicts.

## Included placeholders
- data/: sample CSV files describing the schema for training and inference data.
- notebooks/eda.ipynb: notebook shell reserved for exploratory data analysis.
- scripts/train.py and scripts/predict.py: outlines for training and inference workflows.
- model_registry/: canonical location for serialized models and metadata.
- web_service/: Flask plus Gunicorn application skeleton meant to wrap the model in an API.
- frontend/: stub for a minimal page that can POST feature values to the API.
- requirements.txt and Dockerfile: placeholders for dependency management and containerization.

## Next steps
1. Populate the notebook with EDA steps on the data in data/.
2. Implement the preprocessing pipeline and model training logic in scripts/train.py.
3. Save trained artifacts into model_registry/ and wire them into scripts/predict.py.
4. Build out the Flask routes inside web_service/app.py and connect the HTML form in frontend/index.html.
5. Complete the Dockerfile so the service can be deployed consistently.



Development and Notebook Interaction
.venv\Scripts\activate.bat


uv pip install -r requirements.txt


Deployment 
Docker
