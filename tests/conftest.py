"""
tests/conftest.py

Prevents any test from hitting a live network dependency just to be
collected. config.py fetches every secret from GCP Secret Manager as soon
as it's imported, and nearly every module under retrieval/ and ingestion/
does `import config` at module level -- so without this, `uv run pytest`
would need real GCP credentials just to COLLECT tests, defeating the whole
point of a fast/free/deterministic suite, and failing outright in GitHub
Actions, which has no GCP credentials at all.

conftest.py is always imported before any test module in the same tree, so
injecting a fake `config` into sys.modules here means the real config.py
-- and its live get_secret() calls -- never executes during tests.
"""

import sys
from types import ModuleType

_fake_config = ModuleType("config")
_fake_config.PROJECT_ID = "test-project"
_fake_config.GCP_PROJECT_ID = "test-project"
_fake_config.GROQ_API_KEY = "test-groq-key"
_fake_config.GCS_BUCKET_NAME = "test-bucket"
_fake_config.BIGQUERY_DATASET = "test_dataset"
_fake_config.BIGQUERY_MART_DATASET = "test_mart_dataset"
_fake_config.EMBEDDING_MODEL_PATH = "models/Xenova/all-MiniLM-L6-v2"
_fake_config.CHUNK_SIZE = 500
_fake_config.CHUNK_OVERLAP = 50
_fake_config.NUM_RESULTS = 5
_fake_config.LLM_MODEL = "openai/gpt-oss-120b"

sys.modules["config"] = _fake_config