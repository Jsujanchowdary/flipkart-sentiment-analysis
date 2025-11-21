import os
import json
import csv
import io
import uuid
import time
import requests
import boto3

# Env variables
EC2_URL = "http://18.61.25.80/predict"     
TABLE_NAME = "nikhil"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 50))

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)

def preprocess(text: str) -> str:
    import re
    text = re.sub('[^a-zA-Z0-9]', ' ', (text or ""))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def call_ec2(texts):
    """Send batch of texts to EC2 model API."""
    payload = {"texts": texts}
    resp = requests.post(EC2_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def lambda_handler(event, context):
    record = event["Records"][0]
    bucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    print(f"Triggered by S3 file: s3://{bucket}/{key}")

    # Read CSV from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    csv_data = obj["Body"].read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(csv_data))

    batch_texts = []
    batch_info = []
    total = 0

    with table.batch_writer() as writer:
        for idx, row in enumerate(reader):
            text = f"{row.get('Review','')} {row.get('Summary','')}"
            text_clean = preprocess(text)

            batch_texts.append(text_clean)
            batch_info.append(row)

            # send batch
            if len(batch_texts) == BATCH_SIZE:
                results = call_ec2(batch_texts)

                preds = results["predictions"]
                labels = results["labels"]

                for r, p, l in zip(batch_info, preds, labels):
                    writer.put_item(Item={
                        "id": str(uuid.uuid4()),
                        "product_name": r.get("ProductName", ""),
                        "price": r.get("Price", ""),
                        "rate": r.get("Rate", ""),
                        "review": r.get("Review", ""),
                        "summary": r.get("Summary", ""),
                        "sentiment": l,
                        "sentiment_code": int(p),
                        "timestamp": int(time.time())
                    })
                    total += 1

                batch_texts = []
                batch_info = []

        # Final leftover batch
        if batch_texts:
            results = call_ec2(batch_texts)
            preds = results["predictions"]
            labels = results["labels"]

            for r, p, l in zip(batch_info, preds, labels):
                writer.put_item(Item={
                    "id": str(uuid.uuid4()),
                    "product_name": r.get("ProductName", ""),
                    "price": r.get("Price", ""),
                    "rate": r.get("Rate", ""),
                    "review": r.get("Review", ""),
                    "summary": r.get("Summary", ""),
                    "sentiment": l,
                    "sentiment_code": int(p),
                    "timestamp": int(time.time())
                })
                total += 1

    return {
        "statusCode": 200,
        "body": json.dumps({"processed": total})
    }
