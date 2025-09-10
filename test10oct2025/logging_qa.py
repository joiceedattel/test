from app.config import settings
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, HttpResponseError
import logging
import json


credential = AzureNamedKeyCredential(
    settings.azure_storage_account_name, settings.azure_storage_account_key
)
blob_service = BlobServiceClient(
    account_url=settings.azure_blob_endpoint, credential=credential
)
container_client = blob_service.get_container_client(settings.blob_container_name)


# Append interaction to daily file in blob
def append_to_daily_blob(question: str, user_id: str, response: str, response_id: str):
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    blob_path = f"{user_id}/{today_str}.jsonl"

    # Format line to append
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "question": question,
        "response": response,
        "id": response_id,
    }
    log_line = json.dumps(log_entry) + "\n"

    blob_client = container_client.get_blob_client(blob_path)
    data_bytes = log_line.encode("utf-8")

    try:
        # Try to download existing content
        blob_client.append_block(data_bytes)
    except ResourceNotFoundError:
        # Blob doesn't exist — create it and retry append
        try:
            blob_client.create_append_blob()
            blob_client.append_block(data_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to create and append to blob: {e}") from e

    except ResourceExistsError as e:
        raise RuntimeError(f"Blob exists but could not be appended (may be sealed or corrupted): {e}") from e

    except HttpResponseError as e:
        raise RuntimeError(f"Append failed due to HTTP error: {e.message}") from e

    except Exception as e:
        raise RuntimeError(f"Unexpected error during append: {e}") from e


def log_low_quality_response(question, answer, faithfulness, answer_relevance, context):
    log_entry = {
        "event": "low_quality_response",
        "question": question,
        "answer": answer,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context": context,
    }
    logging.warning(json.dumps(log_entry))