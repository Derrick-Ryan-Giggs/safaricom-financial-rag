"""
ingestion/upload_gcs.py

Uploads raw PDFs and processed JSONL chunks to the GCS bucket that backs
this project (gs://safaricom-rag/). Skips files that already exist in GCS
with a matching size, so re-running this after adding one new PDF doesn't
re-upload everything.

Usage:
    uv run python ingestion/upload_gcs.py --raw-dir raw --processed-dir processed
"""

import argparse
from pathlib import Path

from google.cloud import storage

import config


def get_bucket() -> storage.Bucket:
    client = storage.Client(project=config.GCP_PROJECT_ID)
    return client.bucket(config.GCS_BUCKET_NAME)


def upload_directory(bucket: storage.Bucket, local_dir: str, gcs_prefix: str, pattern: str) -> None:
    """
    Upload every file matching `pattern` in `local_dir` to gs://<bucket>/<gcs_prefix>/.
    Skips upload if a blob with the same name and size already exists.
    """
    local_dir_obj = Path(local_dir)
    if not local_dir_obj.exists():
        print(f"Skipping {local_dir} -- directory does not exist.")
        return

    files = sorted(local_dir_obj.glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' found in {local_dir}.")
        return

    for local_path in files:
        blob_name = f"{gcs_prefix}/{local_path.name}"
        blob = bucket.blob(blob_name)

        if blob.exists():
            blob.reload()
            if blob.size == local_path.stat().st_size:
                print(f"Skipping {local_path.name} (already uploaded, same size).")
                continue

        print(f"Uploading {local_path.name} -> gs://{bucket.name}/{blob_name}")
        blob.upload_from_filename(str(local_path))

    print(f"Done: {local_dir} -> gs://{bucket.name}/{gcs_prefix}/")


def main():
    parser = argparse.ArgumentParser(description="Upload raw PDFs and processed JSONL to GCS.")
    parser.add_argument("--raw-dir", default="raw", help="Local directory containing raw PDFs.")
    parser.add_argument("--processed-dir", default="processed", help="Local directory containing processed JSONL files.")
    args = parser.parse_args()

    bucket = get_bucket()
    upload_directory(bucket, args.raw_dir, "raw", "*.pdf")
    upload_directory(bucket, args.processed_dir, "processed", "*.jsonl")


if __name__ == "__main__":
    main()