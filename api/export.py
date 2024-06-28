#export.py
#export.py
from datetime import timedelta
from io import StringIO
import json
from typing import List
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select

import model.db as db
import model.ql as ql
from api.helper import check_if_role_exists, flatten_sql, get_user_from_token, upload_to_s3
from config import logger, s3_client
from model.db import get_session

router = APIRouter(prefix="/export")


@router.post("/")
async def export_products(
    export_params: ql.ExportParams,
    read_fabric: ql.ReadFabric,
    edit_product: ql.EditProduct,
    edit_hierarchy: ql.EditHierarchy,
    user=Depends(get_user_from_token)
):
    """
    Input: one/many of the below
    Myntra MP
    Ajio SOR
    Ajio B2B
    Flipkart MP
    Flipkart SOR
    Amazon Cocoblu
    Shopify
    DataKart
    or
    one product id (export version history of the product)
    Output: zip
    """
    if not check_if_role_exists(user["department"]["roles"], "export", "create"):
        raise HTTPException(403, "You don't have permission to export products")

    cols = [
        "product_id",
        "style_id",
        "standard_size",
        "barcode_size",
        "ean",
        "myntra_id",
        "ajio_id",
        "ajio_ean",
        "gender",
        "vertical",
        "fabric_category",
        "brand",
        "usage",
        "brick",
        "product",
        "sub_product",
        "story",
        "target_audience",
        "fit",
        "mrp",
        "cost",
        "tags",
        "colour_family",
        "primary_colour",
        "secondary_colour",
        "tertiary_colour",
        "season",
        "exclusive",
        "garment_pattern",
        "print_pattern_type",
        "number_of_components",
        "number_of_pockets",
        "pocket_type",
        "neck",
        "collar",
        "placket",
        "length",
        "sleeve_length",
        "sleeve_type",
        "hemline",
        "waist_rise",
        "closure",
        "dress_type",
        "top_style_type",
        "jacket_type",
        "distress",
        "fade",
        "dungaree_bottom_type",
        "footwear_ankle_type",
        "footwear_insole",
        "fabric_code",
        "fabric_rate",
        "fabric_story",
        "fabric_composition",
        "fabric_hsn",
        "fabric_weave_pattern",
        "fabric_vendor",
        "denim_cast",
        "denim_wash",
        "wash_care",
        "footwear_upper_material",
        "footwear_sole_material",
        "garment_waist",
        "inseam_length",
        "to_fit_waist",
        "across_shoulder",
        "chest",
        "front_length",
        "to_fit_foot_length",
        "shoe_weight",
        "first_grn_date",
        "first_live_date",
        "first_sold_date",
        "status",
        "images",
        "deleted",
        "created_at",
        "created_by",
        "updated_at",
        "updated_by",
        "product_description",  # Add this line
    ]
    if user["department"]["name"] == "merchandise":
        cols.remove("cost")

    end_date = export_params.end_date + timedelta(days=1)
    logger.info(
        f"{user['name']} is trying to export products between: {export_params.start_date}-{end_date}"
    )


    filters = [
        getattr(db.Styles, export_params.date_type.value) >= export_params.start_date,
        getattr(db.Styles, export_params.date_type.value) <= end_date,
        db.Styles.deleted == False,
    ]
    
     # Conditional status filter
    if user["department"]["name"] == "design":
        if edit_product.status:
            filters.append(db.Styles.status == edit_product.status)

    else:
        if edit_product.status is not None:
            filters.append(db.Styles.status == edit_product.status)
        elif edit_product.status is None:
            raise HTTPException(400, "Status cannot be None for your department")
    
    # Add filters for new dropdowns if values are provided
    if read_fabric.fabric_code:
        filters.append(db.Styles.fabric_code == read_fabric.fabric_code)
    if edit_hierarchy.gender:
        filters.append(db.Styles.gender == edit_hierarchy.gender)
    if edit_hierarchy.vertical:
        filters.append(db.Styles.vertical == edit_hierarchy.vertical)
    if edit_hierarchy.fabric_category:
        filters.append(db.Styles.fabric_category == edit_hierarchy.fabric_category)
    if edit_hierarchy.brand:
        filters.append(db.Styles.brand == edit_hierarchy.brand)
    if edit_hierarchy.brick:
        filters.append(db.Styles.brick == edit_hierarchy.brick)
    if edit_hierarchy.product:
        filters.append(db.Styles.product == edit_hierarchy.product)
    if edit_hierarchy.sub_product:
        filters.append(db.Styles.sub_product == edit_hierarchy.sub_product)

    
    async with get_session() as s:
        query = select(db.Styles).filter(and_(*filters))
        resp = (await s.execute(query)).unique().all()
    
    resp = flatten_sql(resp)
    df = pd.DataFrame.from_records(resp)

    logger.info(f"DataFrame before dropping duplicates: {df.shape}")
    df.drop_duplicates(subset=cols, inplace=True)
    logger.info(f"DataFrame after dropping duplicates: {df.shape}")

    
    if len(df):
        custom_sort = ['XXXS', 'XXS', 'XS', 'S', 'M', 'L', 'XL', '2XL', 'XXL', '3XL', '4XL', '5XL', '6XL'] + [str(x) for x in list(range(10, 50))] + ['UK 6/ 24.1 CM', 'UK 7/ 24.8 CM', 'UK 8/ 25.7 CM', 'UK 9/ 26.7 CM', 'UK 10/ 27.3 CM', 'UK 11/ 27.9 CM']
        size_dtype = pd.CategoricalDtype(custom_sort, ordered=True)
        df["standard_size"] = df["standard_size"].astype(size_dtype)
        df.sort_values(["product_id", "standard_size"], inplace=True)
        df = df[cols]
        blurb = upload_to_s3(df, 'exports')
        return {
            "msg": "Products exported successfully",
            "info": {
                "file": f"https://tigc-chanakya.s3.ap-south-1.amazonaws.com/{blurb}",
            },
        }
    else:
        return {"msg": "No Products found"}

