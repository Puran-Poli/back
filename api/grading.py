import itertools
import time
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import String, cast, extract, func, select

from api.helper import (
    check_if_role_exists,
    get_user_from_token,
)
from config import logger
from model import db
from model.db import get_session

router = APIRouter(prefix="/grading")


@router.post("/")
async def get_grading(filters: dict = None, user=Depends(get_user_from_token)):
    if filters is None:
        filters = {}
    if not check_if_role_exists(user["department"]["roles"], "sales analytics", "read"):
        raise HTTPException(403, "You don't have permission to see gradings")
    logger.info(f"{user['name']} is trying to see grading")

    async with get_session() as s:
        get_distict_products = select(db.Styles.product).distinct()
        product_categories = (await s.execute(get_distict_products)).all()

    product_filter = filters.get("product", [p[0] for p in product_categories])
    print(product_filter)
