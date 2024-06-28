#helper.py
from datetime import datetime, timedelta
from typing import Union
from io import StringIO
from uuid import uuid4
import asyncio
import jwt
import pandas as pd
from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm.collections import InstrumentedList
import requests
from io import BytesIO
import boto3
import cv2
import numpy as np

from config import CREDENTIALS_EXCEPTION, logger, oauth2_scheme, send_notification, settings, s3_client
from model import db, ql
from model.db import get_session

'''
def flatten_product(product: Union[ql.CreateProduct, ql.EditProduct], product_id: str):
    product_dict = product.dict()
    styles_row = {}
    sizes_rows = []
    if not product_dict["sizes"]:
        product_dict["sizes"] = []
    for key in product_dict.keys():
        if key == "sizes":
            unique_sizes = set()
            for size in product_dict["sizes"]:
                if size["standard_size"] not in unique_sizes:
                    unique_sizes.add(size["standard_size"])
                    sizes_row = {"product_id": product_id}
                    clean_size = {}
                    for key in size.keys():
                        if size[key]:
                            clean_size[key] = size[key]
                    sizes_row.update(clean_size)
                    sizes_rows.append(sizes_row)
        elif key == "tags":
            if product_dict[key]:
                styles_row.update({key: ",".join(product_dict[key])})
        elif type(product_dict[key]) == dict:
            styles_row.update(product_dict[key])
        else:
            styles_row.update({key: product_dict[key]})
    clean_styles_row = {}
    for key in styles_row.keys():
        if styles_row[key] == 0 :
            clean_styles_row[key] = 0
        if styles_row[key]:
            clean_styles_row[key] = styles_row[key]
        else:
            if type(styles_row[key]) is bool:
                clean_styles_row[key] = styles_row[key]
    return clean_styles_row, sizes_rows
'''
def flatten_product(product: Union[ql.CreateProduct, ql.EditProduct], product_id: str):
    product_dict = product.dict()
    styles_row = {}
    sizes_rows = []

    if not product_dict["sizes"]:
        product_dict["sizes"] = []

    for key in product_dict.keys():
        if key == "sizes":
            unique_sizes = set()
            for size in product_dict["sizes"]:
                if size["standard_size"] not in unique_sizes:
                    unique_sizes.add(size["standard_size"])
                    sizes_row = {"product_id": product_id}
                    clean_size = {}
                    for size_key in size.keys():
                        clean_size[size_key] = size[size_key]  # Include all keys, even if None
                    sizes_row.update(clean_size)
                    sizes_rows.append(sizes_row)
        elif key == "tags":
            if product_dict[key]:
                styles_row.update({key: ",".join(product_dict[key])})
            else:
                styles_row.update({key: None})  # Ensure empty tags are included as None
        elif type(product_dict[key]) == dict:
            for sub_key in product_dict[key].keys():
                styles_row.update({sub_key: product_dict[key][sub_key]})  # Include sub-dictionary keys
            if not product_dict[key]:
                styles_row.update({key: None})  # Ensure dict keys are included as None if empty
        else:
            styles_row.update({key: product_dict[key]})  # Include all keys, even if None

    clean_styles_row = {}
    for key in styles_row.keys():
        clean_styles_row[key] = styles_row[key]  # No filtering, include everything

    return clean_styles_row, sizes_rows


def unflatten_product(styles_row: db.Styles, sizes_rows: list[db.Sizes]):
    sizes = []
    for row in sql_to_dict(sizes_rows):
        sizes.append(ql.Size(**row))
    style = sql_to_dict(styles_row)
    if not style["number_of_pockets"]:
        style["number_of_pockets"] = 0
    style["images"] = eval(style["images"]) if style["images"] else None
    tags = style.pop("tags")
    style.pop("sizes")
    return ql.ReadProduct(
        **style,
        tags=tags.split(",") if tags else [],
        hierarchy=ql.Hierarchy(**style),
        colour=ql.Colour(**style),
        design=ql.Design(**style),
        fabric=ql.ReadFabric(**style),
        dates=ql.Dates(**style),
        sizes=sizes,
    )


def sql_to_pd(resp: list):
    df = []
    for val in resp:
        df.append(sql_to_dict(val[0]))
    return pd.DataFrame.from_dict(df)


def flatten_sql(rows):
    resp = []
    for row in rows:
        row = sql_to_dict(row[0])
        sizes = row.pop("sizes")
        for size in sizes:
            size = sql_to_dict(size)
            temp = row.copy()
            temp.update(size)
            resp.append(temp)
    return resp


def sql_to_dict(model):
    """
    converts a model / list(model) to a dict
    """
    if (type(model) == list) or (type(model) == InstrumentedList):
        result = []
        for row in model:
            row = row.__dict__
            row.pop("_sa_instance_state")
            result.append(row)
        return result
    else:
        row = model.__dict__
        row.pop("_sa_instance_state")
        return row


def create_token(data, login_time, type="access"):
    logger.info(f"Creating {type} token for {data['name']}")
    if type == "access":
        data["token_type"] = "access"
        data.update(
            {
                "exp": login_time
                + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            }
        )
    elif type == "refresh":
        data["token_type"] = "refresh"
        data.update(
            {
                "exp": login_time
                + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
            }
        )
    else:
        return None
    return jwt.encode(data, settings.API_SECRET_KEY, algorithm="HS256")


def decode_token(token):
    return jwt.decode(token, settings.API_SECRET_KEY, algorithms=["HS256"])


async def get_user_from_email(email: str) -> dict:
    query = select(db.User).where(db.User.email == email, db.User.deleted == False)
    async with get_session() as s:
        user = (await s.execute(query)).first()
    if not user:
        logger.error(f"User with email {email} not found or deleted")
        raise HTTPException(
            status_code=401, detail="User email not found or is deactivated"
        )
    user = sql_to_dict(user[0])
    user["department"] = sql_to_dict(user["department"])
    user["department"]["roles"] = [
        sql_to_dict(role) for role in user["department"]["roles"]
    ]
    return user


async def get_user_from_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        logger.warning("Decoding token failed")
        raise CREDENTIALS_EXCEPTION

    if payload["token_type"] != "access":
        raise HTTPException(status_code=401, detail="Not an access token")

    # user = await get_user_from_email(payload["email"])
    # payload.update(user)

    # query = select(db.Session).where(
    #     db.Session.user_id == user["id"],
    #     db.Session.login_time == datetime.fromisoformat(payload["login_time"]),
    #     db.Session.deleted == False,
    # )
    # async with get_session() as s:
    #     session = (await s.execute(query)).first()
    # if not session:
    #     logger.warning("Session not found or has expired")
    #     raise HTTPException(status_code=401, detail="Session not found or has expired")

    return payload


async def refresh_helper(token: str = Depends(oauth2_scheme)):
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        logger.warning("Decoding token failed")
        raise CREDENTIALS_EXCEPTION

    if payload["token_type"] != "refresh":
        raise HTTPException(status_code=401, detail="Not a refresh token")

    user = await get_user_from_email(payload["email"])
    payload.update(user)
    return payload


def check_if_role_exists(user_roles, role, operation):
    for r in user_roles:
        if (r["feature"] == role) and (r[operation] == True):
            return True
    return False


def upload_to_s3(df, foldername):
    csv_buf = StringIO()
    df.to_csv(csv_buf, header=True, index=False)
    csv_buf.seek(0)
    exportid = str(uuid4())
    datetime_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    blurb = f"{foldername}/{settings.env}-{datetime_now}-{exportid}.csv"
    try:
        s3_client.put_object(
            Bucket="tigc-chanakya", Body=csv_buf.getvalue(), Key=blurb
        )
        return blurb
    except Exception as e:
        send_notification(f"[{settings.env}] Error uploading to s3: {foldername}/{str(e)}")
        raise HTTPException(status_code=422, detail="Error uploading to S3")

# resp = s3_client.list_objects_v2(
#     Bucket="tigc", Prefix="images/070fb6a3-9826-45cd-997e-ccc202ebbda9/"
# )
# images = []
# for obj in resp["Contents"]:
#     files = obj["Key"]
#     images.append("https://tigc.s3.amazonaws.com/" + files)


# def ctree():
#     """One of the python gems. Making possible to have dynamic tree structure."""
#     return defaultdict(ctree)


# def build_leaf(name, leaf):
#     """Recursive function to build desired custom tree structure"""
#     res = {"name": name}
#     defaults = leaf.pop("defaults", None)
#     # add children node if the leaf actually has any children
#     if len(leaf.keys()) > 0:
#         res["children"] = [build_leaf(k, v) for k, v in leaf.items()]
#     if defaults:
#         res["defaults"] = defaults
#     return res


# def get_hierarchy():
#     """The main thread composed from two parts.
#     First it's parsing the csv file and builds a tree hierarchy from it.
#     Second it's recursively iterating over the tree and building custom
#     json-like structure (via dict).
#     """
#     tree = ctree()
#     with open("csv/hierarchy_n_default_sizes.csv") as csvfile:
#         reader = csv.reader(csvfile)
#         for rid, row in enumerate(reader):
#             # skipping first header row
#             if rid == 0:
#                 header = row
#                 continue
#             # usage of python magic to construct dynamic tree structure and
#             # basically grouping csv values under their parents
#             leaf = tree[row[0]]
#             for cid in range(1, 10):
#                 leaf = leaf[row[cid]]
#             leaf["defaults"] = row[10:]
#     # building a custom tree structure
#     res = []
#     for name, leaf in tree.items():
#         res.append(build_leaf(name, leaf))
#     return {"header": header, "hierarchy": res}
'''
def add_tags_to_image(product_id, main_image_url, foldername, bucket_name, tag_url_list):
    s3_client = boto3.client('s3')
    base_s3_url = f"https://{bucket_name}.s3.amazonaws.com/{foldername}/"

    # Download the main image
    main_image_url = get_direct_download_url(main_image_url)
    main_image = download_image(main_image_url)
    if main_image is None:
        print(f"Error: Could not load main image from {main_image_url}")
        return None

    # Overlay tags onto the image
    fixed_margin = 20  # Margin between images
    tag_height, tag_width = 100, 100
    x = main_image.shape[1] - tag_width - fixed_margin

    for idx, tag_image_url in enumerate(tag_url. list):
        tag_image = download_image(tag_image_url)
        if tag_image is None:
            print(f"Error: Could not load tag image from {tag_image_url}")
            continue

        tag_image = cv2.resize(tag_image, (tag_width, tag_height))
        y = fixed_margin + idx * (tag_height + fixed_margin)

        # Overlay tag image on the main image
        if tag_image.shape[2] == 4:  # RGBA
            overlay_rgba(main_image, tag_image, x, y)
        else:  # RGB
            main_image[y:y+tag_height, x:x+tag_width] = tag_image

    # Save to S3
    export_id = str(uuid4())
    datetime_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{product_id}#{export_id}.webp"
    output_path = f"{base_s5_url}{filename}"
    _, buffer = cv2.imencode('.webp', main_image)
    s3_client.put_object(
        Bucket=bucket_name,
        Body=BytesIO(buffer).getvalue(),
        Key=f"{foldername}/{filename}",
        ContentType='image/webp'
    )
    print(f"Image saved to S3 at {output_path}")
    return output_path
    
def get_direct_download_url(url):
    """Converts various cloud storage URLs to direct download links if necessary."""
    if "drive.google.com" in url:
        base_drive_url = "https://drive.google.com/uc?export=download&id="
        file_id = url.split('/d/')[1].split('/view')[0]
        direct_url = f"{base_drive_url}{file_id}"
        logger.info("Direct download URL prepared: " + direct_url)
        return direct_url
    logger.info("No conversion needed, returning original URL.")
    return url

def download_image(url):
    try:
        new_url = get_direct_download_url(url)
        response = requests.get(new_url)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        return image
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def overlay_rgba(main_image, tag_image, x, y):
    """Helper function to overlay an RGBA image."""
    alpha_s = tag_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        main_image[y:y+tag_image.shape[0], x:x+tag_image.shape[1], c] = (
            alpha_s * tag_image[:, :, c] + alpha_l * main_image[y:y+tag_image.shape[0], x:x+tag_image.shape[1], c]
        )

        
        '''
        #########################################
def get_direct_download_url(url):
    """Converts various cloud storage URLs to direct download links if necessary."""
    if "drive.google.com" in url:
        base_drive_url = "https://drive.google.com/uc?export=download&id="
        file_id = url.split('/d/')[1].split('/view')[0]
        direct_url = f"{base_drive_url}{file_id}"
        logger.info("Direct download URL prepared: " + direct_url)
        return direct_url
    logger.info("No conversion needed, returning original URL.")
    return url

def download_image(url):
    """Download an image from a URL."""
    try:
        new_url = get_direct_download_url(url)
        response = requests.get(new_url, timeout=10)
        response.raise_for_status()
        image_bytes = BytesIO(response.content)
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        logger.info(f"Image downloaded successfully from {url}")
        return image
    except requests.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

def overlay_rgba(main_image, tag_image, x, y):
    """Overlay an RGBA image onto another image."""
    alpha_s = tag_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        main_image[y:y+tag_image.shape[0], x:x+tag_image.shape[1], c] = (
            alpha_s * tag_image[:, :, c] + alpha_l * main_image[y:y+tag_image.shape[0], x:x+tag_image.shape[1], c]
        )
    logger.info("RGBA image overlay complete.")

def add_tags_to_image(product_id, main_image_url, foldername, bucket_name, tag_url_list):
    s3_client = boto3.client('s3')
    base_s3_url = f"https://{bucket_name}.s3.ap-south-1.amazonaws.com/"

    # Download the main image
    main_image = download_image(main_image_url)
    if main_image is None:
        logger.error(f"Could not load main image from {main_image_url}")
        return None

    if not tag_url_list:
        logger.info("No tags provided, uploading main image directly.")
    else:
        # Overlay tags onto the image
        fixed_margin = 20  # Margin between images
        tag_height, tag_width = 100, 100
        x = main_image.shape[1] - tag_width - fixed_margin

        for idx, tag_image_url in enumerate(tag_url_list):
            tag_image = download_image(tag_image_url)
            if tag_image is None:
                logger.error(f"Could not load tag image from {tag_image_url}")
                continue

            tag_image = cv2.resize(tag_image, (tag_width, tag_height))
            y = fixed_margin + idx * (tag_height + fixed_margin)

            # Overlay tag image on the main image
            if tag_image.shape[2] == 4:  # RGBA
                overlay_rgba(main_image, tag_image, x, y)
            else:  # RGB
                main_image[y:y+tag_height, x:x+tag_width] = tag_image
            logger.info(f"Tag image overlaid for tag {idx+1}")

    # Save to S3
    export_id = str(uuid4())
    datetime_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{foldername}/{settings.env}--{datetime_now}--{export_id}.webp"
    output_path = f"{base_s3_url}{filename}"
    _, buffer = cv2.imencode('.webp', main_image)
    s3_client.put_object(
        Bucket=bucket_name,
        Body=BytesIO(buffer).getvalue(),
        Key=filename,
        ContentType='image/webp'
    )
    logger.info(f"Image saved to S3 at {output_path}")
    return output_path
'''
earlier nomenclature
def add_tags_to_image(product_id, main_image_url, foldername, bucket_name, tag_url_list):
    s3_client = boto3.client('s3')
    base_s3_url = f"https://{bucket_name}.s3.ap-south-1.amazonaws.com/"

    # Download the main image
    main_image = download_image(main_image_url)
    if main_image is None:
        logger.error(f"Could not load main image from {main_image_url}")
        return None

    # Overlay tags onto the image
    fixed_margin = 20  # Margin between images
    tag_height, tag_width = 100, 100
    x = main_image.shape[1] - tag_width - fixed_margin

    for idx, tag_image_url in enumerate(tag_url_list):
        tag_image = download_image(tag_image_url)
        if tag_image is None:
            logger.error(f"Could not load tag image from {tag_image_url}")
            continue

        tag_image = cv2.resize(tag_image, (tag_width, tag_height))
        y = fixed_margin + idx * (tag_height + fixed_margin)

        # Overlay tag image on the main image
        if tag_image.shape[2] == 4:  # RGBA
            overlay_rgba(main_image, tag_image, x, y)
        else:  # RGB
            main_image[y:y+tag_height, x:x+tag_width] = tag_image
        logger.info(f"Tag image overlaid for tag {idx+1}")

    # Save to S3
    export_id = str(uuid4())
    datetime_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"{foldername}/{product_id}--0--{datetime_now}-{export_id}.webp"
    output_path = f"{base_s3_url}{filename}"
    _, buffer = cv2.imencode('.webp', main_image)
    s3_client.put_object(
        Bucket=bucket_name,
        Body=BytesIO(buffer).getvalue(),
        Key=f"{foldername}/{filename}",
        ContentType='image/webp'
    )
    logger.info(f"Image saved to S3 at {output_path}")
    return output_path
    '''
'''
def get_direct_download_url(url):
    """Converts various cloud storage URLs to direct download links if necessary."""
    logger.debug(f"Converting URL if necessary: {url}")
    if "drive.google.com" in url:
        base_drive_url = "https://drive.google.com/uc?export=download&id="
        file_id = url.split('/d/')[1].split('/view')[0]
        direct_url = f"{base_drive_url}{file_id}"
        logger.info("Direct download URL prepared: " + direct_url)
        return direct_url
    logger.info("No conversion needed, returning original URL.")
    return url

async def download_image(session, url):
    """Download an image from a URL."""
    try:
        new_url = get_direct_download_url(url)
        logger.debug(f"Downloading image from URL: {new_url}")
        async with session.get(new_url, timeout=10) as response:
            response.raise_for_status()
            logger.debug(f"Response status: {response.status}")
            image_bytes = BytesIO(await response.read())
            logger.debug(f"Image bytes size: {len(image_bytes.getvalue())}")
            image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            logger.info(f"Image downloaded successfully from {url}")
            return image
    except aiohttp.ClientError as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

def overlay_rgba(main_image, tag_image, x, y):
    """Overlay an RGBA image onto another image."""
    logger.debug(f"Overlaying RGBA image at position: ({x}, {y})")
    alpha_s = tag_image[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        main_image[y:y+tag_image.shape[0], x:x+tag_image.shape[1], c] = (
            alpha_s * tag_image[:, :, c] + alpha_l * main_image[y:y+tag_image.shape[0], x:x+tag_image.shape[1], c]
        )
    logger.info("RGBA image overlay complete.")

async def add_tags_to_image(product_id, main_image_url, foldername, bucket_name, tag_url_list):
    s3_client = boto3.client('s3')
    base_s3_url = f"https://{bucket_name}.s3.amazonaws.com/{foldername}/"
    logger.debug(f"Starting add_tags_to_image for product_id: {product_id}")

    async with aiohttp.ClientSession() as session:
        # Download the main image
        main_image = await download_image(session, main_image_url)
        if main_image is None:
            logger.error(f"Could not load main image from {main_image_url}")
            return None
        logger.debug(f"Main image loaded successfully for product_id: {product_id}")

        # Overlay tags onto the image
        fixed_margin = 20  # Margin between images
        tag_height, tag_width = 100, 100
        x = main_image.shape[1] - tag_width - fixed_margin

        tasks = []
        for idx, tag_image_url in enumerate(tag_url_list):
            logger.debug(f"Queueing tag image for overlay: {tag_image_url}")
            tasks.append(process_tag_image(session, main_image, tag_image_url, x, fixed_margin, idx, tag_height, tag_width))

        await asyncio.gather(*tasks)

        # Save to S3
        export_id = str(uuid4())
        datetime_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{product_id}--{export_id}.webp"
        output_path = f"{base_s3_url}{filename}"
        _, buffer = cv2.imencode('.webp', main_image)
        s3_client.put_object(
            Bucket=bucket_name,
            Body=BytesIO(buffer).getvalue(),
            Key=f"{foldername}/{filename}",
            ContentType='image/webp'
        )
        logger.info(f"Image saved to S3 at {output_path}")
        return output_path

async def process_tag_image(session, main_image, tag_image_url, x, fixed_margin, idx, tag_height, tag_width):
    logger.debug(f"Processing tag image {idx+1} from URL: {tag_image_url}")
    tag_image = await download_image(session, tag_image_url)
    if tag_image is None:
        logger.error(f"Could not load tag image from {tag_image_url}")
        return

    tag_image = cv2.resize(tag_image, (tag_width, tag_height))
    y = fixed_margin + idx * (tag_height + fixed_margin)

    if tag_image.shape[2] == 4:  # RGBA
        overlay_rgba(main_image, tag_image, x, y)
    else:  # RGB
        main_image[y:y+tag_height, x:x+tag_width] = tag_image
    logger.info(f"Tag image overlaid for tag {idx+1}")
'''