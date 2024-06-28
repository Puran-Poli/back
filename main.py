from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.bulk_update import router as bulk_update_router
from api.export import router as export_router
from api.grading import router as grading_router
from api.products import router as products_router
from api.products_search import router as products_search_router
from api.sales import router as sales_router
from api.user import router as user_router
from config import logger, send_notification, settings
from model.db import get_session

app = FastAPI(
    title="Chanakya Backend", version=settings.VERSION, docs_url="", redoc_url=""
)

origins = ["*"]
# "http://localhost:3000",
# 'https://devtigc.techtact.co',
# 'https://tigc.techtact.co',
# postman?


async def global_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except DBAPIError as e:
        if settings.env == "prod":
            send_notification(f"[{settings.env}] DB Timeout: {str(e)}")
        logger.exception("DB Timeout")
        return Response("DB Timeout", 503)
    except Exception as e:
        if settings.env == "prod":
            send_notification(f"[{settings.env}] Internal Server Error: {str(e)}")
        logger.exception("Internal Server Error")
        return Response("Internal Server Error", 500)


app.middleware("http")(global_exceptions_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=settings.SESSION_SECRET_KEY)


@app.on_event("startup")
async def startup_event():
    if settings.env != "local":
        logger.debug("Server Starting Up...")
        async with get_session() as s:
            await s.execute(select(1))
        logger.debug("DB Connected...")


@app.get("/")
async def health():
    return {"version": settings.VERSION}


@app.on_event("shutdown")
async def shutdown_event():
    if settings.env != "local":
        logger.debug("Server Shutting Down...")


app.include_router(user_router)
app.include_router(products_search_router)
app.include_router(products_router)
app.include_router(export_router)
app.include_router(bulk_update_router)
app.include_router(sales_router)
app.include_router(grading_router)
