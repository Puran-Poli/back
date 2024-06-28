import os
import sys
from functools import lru_cache

import boto3
import requests
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger
from pydantic import BaseSettings
from starlette.config import Config

load_dotenv()


class Settings(BaseSettings):
    VERSION: str = "v2.2.03"
    env: str = os.getenv("ENV", "")
    url: str = os.getenv("URL", "")
    pg_user: str = os.getenv("SQL_USER", "")
    pg_pass: str = os.getenv("SQL_PASS", "")
    pg_host: str = os.getenv("SQL_HOST", "")
    pg_database: str = os.getenv("SQL_DB", "")
    pg_port: str = os.getenv("SQL_PORT", "")
    asyncpg_url: (
        str
    ) = f"postgresql+asyncpg://{pg_user}:{pg_pass}@{pg_host}:5432/{pg_database}"
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", None)
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", None)
    API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
    SESSION_SECRET_KEY = os.environ.get("API_SECRET_KEY")
    FRONTEND_URL = os.environ.get("FRONTEND_URL")
    ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 2
    REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30
    SUPERUSER_EMAIL = os.getenv("SUPERUSER_EMAIL", "")
    SUPERUSER_NAME = os.getenv("SUPERUSER_NAME", "")


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()


def send_notification(message: str):
    logger.error(message)
    try:
        requests.post(
            os.getenv("SLACK_WEBHOOK_URL", ""),
            headers={"Content-Type": "application/json"},
            json={"channel": "Chakanya", "username": "Chakanya", "text": message},
            timeout=15,
        )
    except requests.Timeout:
        logger.error("Timeout occurred when trying to send message to Slack.")
    except requests.RequestException as e:
        logger.error(f"Error occurred when communicating with Slack: {e}.")


config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": (
                "<level>{level}:</level>     <level>{message}</level> <blue>{file}:{function}:"
                "{line}</blue> <green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            ),
            "backtrace": True,
            "diagnose": True,
            "enqueue": True,
        },
        {
            "sink": "logs/{time}.log",
            "format": (
                "<level>{level}:</level>     <level>{message}</level> <blue>{file}:{function}:"
                "{line}</blue> <green>{time:YYYY-MM-DD HH:mm:ss}</green>"
            ),
            # "serialize": True,
            "rotation": "00:00",
            "retention": "30 days",
            "compression": "zip",
            "enqueue": True,
        },
    ],
    # "extra": {"user": "someone"}
}

s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

oauth = OAuth(
    Config(
        environ={
            "GOOGLE_CLIENT_ID": settings.GOOGLE_CLIENT_ID,
            "GOOGLE_CLIENT_SECRET": settings.GOOGLE_CLIENT_SECRET,
        }  # type: ignore
    )
)
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/token")
CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)
