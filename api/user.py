from datetime import datetime

from authlib.integrations.starlette_client import OAuthError
from fastapi import APIRouter, Depends, Request
from sqlalchemy import insert, update

from api.helper import (
    create_token,
    get_user_from_email,
    get_user_from_token,
    refresh_helper,
)
from config import CREDENTIALS_EXCEPTION, logger, oauth, settings
from model import db
from model.db import get_session

router = APIRouter(prefix="/user")


@router.get("/login")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request, settings.FRONTEND_URL)


@router.get("/token")
async def authenticate_google_token(request: Request):
    if settings.env != "local":
        try:
            logger.info("Inside google auth")            
            resp = await oauth.google.authorize_access_token(request)
        except OAuthError as e:
            logger.info(f"log error {e}")
            raise CREDENTIALS_EXCEPTION

        user_data = {
            "email": resp["userinfo"]["email"],
            "name": resp["userinfo"]["name"],
            "image": resp["userinfo"].get("picture"),
        }
    else:
        user_data = {
            "email": settings.SUPERUSER_EMAIL,
            "name": settings.SUPERUSER_NAME,
            "image": "https://asdasd",
        }
    login_time = datetime.utcnow()
    user_data.update({"login_time": str(login_time)})

    user = await get_user_from_email(user_data["email"])
    user_data.update(user)

    logger.info(f"Creating session for {user_data['name']}")
    query = insert(db.Session).values({"user_id": user["id"], "login_time": login_time})
    async with get_session() as s:
        await s.execute(query)
    return {
        "access_token": create_token(user_data, login_time),
        "refresh_token": create_token(user_data, login_time, "refresh"),
    }


@router.post("/refresh")
async def refresh(user=Depends(refresh_helper)):
    login_time = datetime.utcnow()
    update_query = (
        update(db.Session)
        .where(
            db.Session.user_id == user["id"],
        )
        .values({"deleted": True})
    )
    insert_query = insert(db.Session).values(
        {"user_id": user["id"], "login_time": login_time}
    )

    async with get_session() as s:
        await s.execute(update_query)
        await s.execute(insert_query)

    logger.info(
        f"Expired all old sessions for the user and created a new session for {user['name']}"
    )
    user.pop("login_time")
    user.pop("exp")
    user.update({"login_time": str(login_time)})
    return {"access_token": create_token(user, login_time)}


@router.get("/logout")
async def logout(user=Depends(get_user_from_token)):
    query = (
        update(db.Session)
        .where(
            db.Session.user_id == user["id"],
        )
        .values({"deleted": True})
    )
    async with get_session() as s:
        await s.execute(query)
    logger.info(f"{user['name']} logged out")
    return {"msg": "Logged out successfully"}


@router.get("/")
def get_user(user=Depends(get_user_from_token)):
    logger.info(f"{user['name']} fetched own user details")
    return user
