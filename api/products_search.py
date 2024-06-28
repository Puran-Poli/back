#product_search.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from api.helper import check_if_role_exists, get_user_from_token, sql_to_dict
from config import logger
from model import db, ql
from model.db import get_session

router = APIRouter(prefix="/search")


@router.post("/")
async def search_products(
    range: list = [0, 10], filters: dict = {}, sort: list = [], user=Depends(get_user_from_token)
):
    """
    Inputs: filters, sorting, pagination
    Output: Basic view of all returned products

    PLP page with filters, sort and pagination
    """
    if not check_if_role_exists(user["department"]["roles"], "product design details", "read"):
        raise HTTPException(
            403, "You don't have permission to read the design details of a product"
        )

    logger.info(f"{user['name']} is trying to search for products")
    resp = []
    count_q = select(db.Styles.product_id)
    query = select(db.Styles)
    if filters:
        for filter in filters.keys():
            if filter == "deleted":
                continue
            if filter == 'ean':
                continue
            if filter == "mrp":
                query = query.filter(db.Styles.mrp.between(*filters[filter]))
                count_q = count_q.filter(db.Styles.mrp.between(*filters[filter]))
                continue
            if filter == "cost":
                query = query.filter(db.Styles.cost.between(*filters[filter]))
                count_q = count_q.filter(db.Styles.cost.between(*filters[filter]))
                continue
            if type(filters[filter]) == str:
                filters[filter] = [filters[filter]]
            query = query.filter(getattr(db.Styles, filter).in_(filters[filter]))
            count_q = count_q.filter(getattr(db.Styles, filter).in_(filters[filter]))
    if not filters.get("deleted"):
        filters["deleted"] = False
    if filters.get('ean'):
        query = query.join(db.Sizes).where(db.Sizes.ean == filters["ean"])
        count_q = query
    else:
        query = query.where(db.Styles.deleted == filters["deleted"])
        count_q = count_q.where(db.Styles.deleted == filters["deleted"])

    if sort:
        query = query.order_by(getattr(getattr(db.Styles, sort[0]), sort[1])())

    query = query.offset(range[0]).limit((range[1] - range[0]))
    styles =[]
    count =[]
    async with get_session() as s:
        styles = (await s.execute(query)).unique().all()
        count = (await s.execute(count_q)).unique().all()

    for style in styles:
        style = sql_to_dict(style[0])
        sizes = style.pop("sizes")
        if style["images"]:
            links = eval(style["images"])
            style["image"] = links[0] if links else None
        product = ql.ProductSummary(**style)
        if not check_if_role_exists(user["department"]["roles"], "product mrp", "read"):
            product.mrp = None
        if not check_if_role_exists(user["department"]["roles"], "product cost", "read"):
            product.cost = None
        product.sizes = [sql_to_dict(size)["standard_size"] for size in sizes]
        resp.append(product)
    return {
        "msg": "products searched successfully",
        "info": {
            "total_items": len(count),
            "items": resp,
        },
    }

