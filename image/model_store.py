"""Manage models in the S3 model store: seed from Hugging Face, list, remove."""

import argparse
import os
import posixpath
import sys
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download

ALLOW_PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"]
# Uploaded after every other file, so an interrupted seed is never mistaken for a stored model
MARKER = ".complete"


def model_id_from_env(bucket: str) -> str:
    model = os.environ.get("MODEL", "")
    prefix = f"s3://{bucket}/"
    if not model.startswith(prefix):
        sys.exit(f"MODEL ({model or 'unset'}) is not under {prefix}; pass a Hugging Face model id instead")
    return model[len(prefix) :].rstrip("/")


def keys(s3, bucket: str, prefix: str = "") -> list[str]:
    try:
        pages = s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        return [obj["Key"] for page in pages for obj in page.get("Contents", [])]
    except s3.exceptions.NoSuchBucket:
        return []


def seed(s3, bucket: str, model_id: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        print(f"Creating bucket {bucket}")
        s3.create_bucket(Bucket=bucket)

    if f"{model_id}/{MARKER}" in keys(s3, bucket, f"{model_id}/"):
        print(f"s3://{bucket}/{model_id} is already in the model store")
        return

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading {model_id} from Hugging Face")
        local_dir = Path(snapshot_download(model_id, local_dir=tmp, allow_patterns=ALLOW_PATTERNS))

        files = [p for p in local_dir.rglob("*") if p.is_file() and ".cache" not in p.parts]
        if not any(p.suffix == ".safetensors" for p in files):
            sys.exit(f"error: {model_id} has no .safetensors weights, which the S3 streamer requires")

        for path in files:
            key = f"{model_id}/{path.relative_to(local_dir).as_posix()}"
            print(f"Uploading s3://{bucket}/{key}")
            s3.upload_file(str(path), bucket, key)
        s3.put_object(Bucket=bucket, Key=f"{model_id}/{MARKER}", Body=b"")

    print(f"Seeded s3://{bucket}/{model_id}")


def list_models(s3, bucket: str) -> None:
    models = {posixpath.dirname(k) for k in keys(s3, bucket) if posixpath.basename(k) == MARKER}
    for model_id in sorted(models):
        print(model_id)


def remove(s3, bucket: str, model_id: str) -> None:
    to_delete = keys(s3, bucket, f"{model_id.rstrip('/')}/")
    if not to_delete:
        sys.exit(f"error: s3://{bucket}/{model_id} is not in the model store")
    for i in range(0, len(to_delete), 1000):
        batch = [{"Key": k} for k in to_delete[i : i + 1000]]
        s3.delete_objects(Bucket=bucket, Delete={"Objects": batch})
    print(f"Removed s3://{bucket}/{model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seed_cmd = commands.add_parser("seed", help="copy a Hugging Face model into the store (default: MODEL)")
    seed_cmd.add_argument("model_id", nargs="?")
    commands.add_parser("list", help="list stored models")
    rm_cmd = commands.add_parser("rm", help="remove a stored model")
    rm_cmd.add_argument("model_id")
    args = parser.parse_args()

    bucket = os.environ.get("S3_BUCKET", "models")
    s3 = boto3.client("s3")

    if args.command == "seed":
        seed(s3, bucket, args.model_id or model_id_from_env(bucket))
    elif args.command == "list":
        list_models(s3, bucket)
    else:
        remove(s3, bucket, args.model_id)


if __name__ == "__main__":
    main()
