#products.py
import gzip
import io
import json
import re
from datetime import datetime
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import delete, func, insert, select, update

from api.helper import (
    check_if_role_exists,
    flatten_product,
    get_user_from_token,
    sql_to_pd,
    unflatten_product,
)
from config import logger
from model import db, ql
from model.db import get_session

router = APIRouter(prefix="/products")


@router.post("/")
async def create_product(product: ql.CreateProduct, user=Depends(get_user_from_token)):
    """
    Input: Most of the product and size details
    Output: product id

    Products are created in a 'Design' state
    """
    #logger.info(f"Received payload: {product.dict()}")
    
    if not check_if_role_exists(
        user["department"]["roles"], "product design details", "create"
    ):
        raise HTTPException(
            403, "You don't have permission to create the design details of a product"
        )

    if product.mrp or product.cost:
        logger.warning(
            f"{user['name']} doesnt have permission to create product mrp or cost, setting it to None"
        )
        product.mrp = None
        product.cost = None

    pattern = r"^\d{4}-.{0,13}$"
    if not re.match(pattern, product.style_id):
        raise HTTPException(
            400,
            "Style id should have 4 digits followed by a hyphen and 0-13 characters",
        )

    async with get_session() as s:
        query = select(db.Styles).where(
            db.Styles.style_id == product.style_id, db.Styles.deleted is not True
        )
        res = (await s.execute(query)).unique().all()
    if res:
        raise HTTPException(400, f"Style id {product.style_id} already exists")

    logger.info(f"{user['name']} is trying to create a product")
    product_id = str(uuid4())
    styles_row, sizes_rows = flatten_product(product, product_id)
    # cost_dict = {
    #     "cost": product.cost,
    #     "date": datetime.now(),
    #     "created_by": user["id"],
    # }
    styles_row.update(
        {
            "product_id": product_id,
            "created_by": user["id"],
            "updated_by": user["id"],
            "cost": product.cost,
            "product_description": product.product_description  # Add this line
        }
    )
    async with get_session() as s:
        query = insert(db.Styles).values(styles_row)
        await s.execute(query)
        query = insert(db.Sizes).values(sizes_rows)
        await s.execute(query)

    return {
        "msg": "Product created successfully",
        "info": {"product_id": styles_row["product_id"]},
    }

'''
@router.put("/{product_id}")
async def edit_product(
    product_id: str, product: ql.EditProduct, user=Depends(get_user_from_token)
):
    """
    Input: product id, details to be changed

    Used to Edit, Approve, Delete a product
    """
    if not check_if_role_exists(
        user["department"]["roles"], "product design details", "update"
    ):
        raise HTTPException(
            403, "You don't have permission to edit the design details of a product"
        )
    if product.mrp:
        logger.warning(
            f"{user['name']} doesnt have permission to edit product mrp, setting it to None"
        )
        product.mrp = None
    if not check_if_role_exists(user["department"]["roles"], "product cost", "update"):
        if product.cost:
            logger.warning(
                f"{user['name']} doesnt have permission to edit product cost, setting them to None"
            )
            product.cost = None
    logger.info(
        f"{user['name']} is trying to edit a product with a change in {product}"
    )

    # deleting a product
    if product.deleted == True:
        if product.status == "Design":
            async with get_session() as s:
                query = delete(db.Styles).where(db.Styles.product_id == product_id)
                await s.execute(query)
            return {
                "msg": "Product permanently deleted successfully",
                "info": {"product_id": product_id},
            }
        else:
            async with get_session() as s:
                query = (
                    update(db.Styles)
                    .where(db.Styles.product_id == product_id)
                    .values({"deleted": True})
                )
                await s.execute(query)
            return {
                "msg": "Product soft deleted successfully",
                "info": {"product_id": product_id},
            }

    styles_row, sizes_rows = flatten_product(product, product_id)
    styles_row.update({"updated_at": datetime.now(), "updated_by": user["id"]})
    async with get_session() as s:
        query = (
            update(db.Styles)
            .where(db.Styles.product_id == product_id)
            .values(styles_row)
        )
        await s.execute(query)
        for size in sizes_rows:
            query = (
                update(db.Sizes)
                .where(db.Sizes.product_id == product_id)
                .where(db.Sizes.standard_size == size["standard_size"])
                .values(size)
            )
            await s.execute(query)
    return {
        "msg": "Product edited successfully",
        "info": {"product_id": product_id},
    }

'''

@router.put("/{product_id}")
async def edit_product(
    product_id: str,
    product: ql.EditProduct,
    user=Depends(get_user_from_token)
):
    """
    Input: product id, details to be changed
    Used to Edit, Approve, Delete a product
    """
    logger.info(f"Received payload: {product.dict()}")

    if not check_if_role_exists(user["department"]["roles"], "product design details", "update"):
        raise HTTPException(403, "You don't have permission to edit the design details of a product")

    if product.mrp:
        logger.warning(f"{user['name']} doesn't have permission to edit product mrp, setting it to None")
        product.mrp = None

    if not check_if_role_exists(user["department"]["roles"], "product cost", "update"):
        if product.cost:
            logger.warning(f"{user['name']} doesn't have permission to edit product cost, setting them to None")
            product.cost = None

    logger.info(f"{user['name']} is trying to edit a product with a change in {product}")

    # deleting a product
    if product.deleted == True:
        if product.status == "Design":
            async with get_session() as s:
                query = delete(db.Styles).where(db.Styles.product_id == product_id)
                await s.execute(query)
            return {
                "msg": "Product permanently deleted successfully",
                "info": {"product_id": product_id},
            }
        else:
            async with get_session() as s:
                query = (
                    update(db.Styles)
                    .where(db.Styles.product_id == product_id)
                    .values({"deleted": True})
                )
                await s.execute(query)
            return {
                "msg": "Product soft deleted successfully",
                "info": {"product_id": product_id},
            }

    styles_row, sizes_rows = flatten_product(product, product_id)

    if product.hierarchy is None:
        # Ignore null values in styles_row if hierarchy is None
        styles_row = {k: v for k, v in styles_row.items() if v is not None}
    
    styles_row.update({
        "updated_at": datetime.now(),
        "updated_by": user["id"],
        
    })

    logger.info(f"Updated styles_row: {styles_row}")

    async with get_session() as s:
        query = (
            update(db.Styles)
            .where(db.Styles.product_id == product_id)
            .values(styles_row)
        )
        await s.execute(query)

        for size in sizes_rows:
            query = (
                update(db.Sizes)
                .where(db.Sizes.product_id == product_id)
                .where(db.Sizes.standard_size == size["standard_size"])
                .values(size)
            )
            await s.execute(query)

    return {
        "msg": "Product edited successfully",
        "info": {"product_id": product_id},
    }

@router.get("/{product_id}")
async def get_product_details(product_id: str, user=Depends(get_user_from_token)):
    """
    Input: product id
    Output: Complete product and size details
    """
    if not check_if_role_exists(
        user["department"]["roles"], "product design details", "read"
    ):
        raise HTTPException(
            403, "You don't have permission to read the design details of a product"
        )

    logger.info(
        f"{user['name']} is trying to fetch details for product id: {product_id}"
    )
    async with get_session() as s:
        query = select(db.Styles).where(db.Styles.product_id == product_id)
        style_row = (await s.execute(query)).unique().first()
    product = unflatten_product(style_row[0], style_row[0].sizes)
    product.product_description = style_row[0].product_description  # Add this line

    if not check_if_role_exists(user["department"]["roles"], "product mrp", "read"):
        product.mrp = None
    if not check_if_role_exists(user["department"]["roles"], "product cost", "read"):
        product.cost = None

    user_ids = {int(product.created_by), int(product.updated_by)}
    async with get_session() as s:
        query = select(db.User).where(db.User.id.in_(user_ids))
        res = (await s.execute(query)).unique().all()
        res = sql_to_pd(res)

    product.created_by = res[res["id"] == int(product.created_by)]["email"].values[0]
    product.updated_by = res[res["id"] == int(product.updated_by)]["email"].values[0]

    return {"msg": "Product fetched successfully", "info": product}


@router.get("/")
async def get_field_options(hierarchy: bool = False, user=Depends(get_user_from_token)):
    """
    Output:
        Allowed values for columns that have fixed options
        Colour family with the colours in it
        Hierarchy of the products with their available sizes and default measurements
    """
    logger.info(
        f"{user['name']} is trying to fetch the field options (hierarchy: {hierarchy})"
    )
    if hierarchy:
        result = {"field_options": {}}
        with open("static/fieldoptions.json", "r") as f:
            result["field_options"] = json.load(f)
        with open("static/hierarchy.csv", "r") as f:
            df = pd.read_csv(f)
            result["hierarchy_and_default_measurements"] = json.loads(
                df.to_json(orient="records")
            )
        byte_stream = io.BytesIO()
        with gzip.GzipFile(fileobj=byte_stream, mode="w") as fout:
            fout.write(json.dumps(result).encode("utf-8"))
        compressed_data = byte_stream.getvalue()

        return Response(compressed_data, media_type="bytes")
    else:
        async with get_session() as s:
            query = select([func.min(db.Styles.cost), func.max(db.Styles.cost)])
            cost_range = (await s.execute(query)).all()[0]
            query = select([func.min(db.Styles.mrp), func.max(db.Styles.mrp)])
            mrp_range = (await s.execute(query)).all()[0]
            query = select(db.Channel.name)
            channels = (await s.execute(query)).all()
            query = select(db.Styles.product).distinct()
            products = (await s.execute(query)).all()

        with open("static/colors.json", "r") as f:
            colors = json.load(f)
        with open("static/filters.json", "r") as f:
            filters = json.load(f)
        return {
            "msg": "Fetched data to initiate the app",
            "info": {
                "mrp": mrp_range,
                "cost": cost_range,
                "colors": colors,
                "filters": filters,
                "channels": [c[0] for c in channels],
                "products": sorted([p[0] for p in products]),
            },
        }
