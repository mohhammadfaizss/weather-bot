import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

script_location = Path(__file__).resolve().parent.parent
local_folder = script_location / "Data"

def download_s3_bucket(bucket_name: str):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")
    
    for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = Path(local_folder) / key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {key}...")
            s3.download_file(bucket_name, key, str(local_path))
    
    print("Done.")

if __name__ == "__main__":
    # put the actual bucket name over here
    download_s3_bucket("thisismycustombucketformymospipeline")