import boto3
import pickle
from io import BytesIO
from botocore.exceptions import ClientError
from ragbot.config.settings import settings
from ragbot.utils.logger import get_logger

logger = get_logger("s3_storage")

def get_s3_client():
    """
    Create S3 client using AWS credential chain.
    Prefers IAM roles, then env vars, then ~/.aws/credentials
    """
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
def upload_chunks(chunks):
    """
    Upload chunks to S3 as pickled bytes.
    
    Args:
        chunks: List of document chunks to upload
    
    Raises:
        ClientError: If S3 upload fails
    """
    logger.info(f"Uploading {len(chunks)} chunks to S3")
    s3 = get_s3_client()
    buffer = BytesIO()
    
    try:
        pickle.dump(chunks, buffer)
        buffer.seek(0)
        
        s3.upload_fileobj(
            buffer,
            settings.s3_bucket,
            settings.s3_chunks_key
        )
        logger.info(f"Successfully uploaded to s3://{settings.s3_bucket}/{settings.s3_chunks_key}")
    finally:
        buffer.close()

def download_chunks():
    """
    Download chunks from S3.
    
    Returns:
        List of chunks if found, None if not found
    
    Raises:
        ClientError: If S3 error other than 404
        pickle.UnpicklingError: If data is corrupted
    """
    logger.info(f"Downloading chunks from S3")
    s3 = get_s3_client()
    buffer = BytesIO()
    
    try:
        s3.download_fileobj(
            settings.s3_bucket,
            settings.s3_chunks_key,
            buffer
        )
        buffer.seek(0)
        chunks = pickle.load(buffer)
        logger.info(f"Successfully downloaded {len(chunks)} chunks")
        return chunks
        
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"Chunks not found: s3://{settings.s3_bucket}/{settings.s3_chunks_key}")
            return None
        else:
            logger.error(f"S3 download failed: {e}")
            raise
            
    except pickle.UnpicklingError as e:
        logger.error(f"Failed to unpickle chunks: {e}")
        raise
        
    finally:
        buffer.close()        