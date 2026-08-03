"""
Central configuration — loads all secrets from GCP Secret Manager.
Only GOOGLE_APPLICATION_CREDENTIALS and GCP_PROJECT_ID come from .env
"""

import os
from google.cloud import secretmanager
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "safaricom-intelligence")

def get_secret(secret_id: str) -> str:
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# load all secrets once at import time
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GCS_BUCKET_NAME = get_secret("GCS_BUCKET_NAME")
BIGQUERY_DATASET = get_secret("BIGQUERY_DATASET")
BIGQUERY_MART_DATASET = get_secret("BIGQUERY_MART_DATASET")

# constants
EMBEDDING_MODEL_PATH = "models/Xenova/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NUM_RESULTS = 5
LLM_MODEL = LLM_MODEL = "openai/gpt-oss-120b"
GCP_PROJECT_ID = PROJECT_ID
