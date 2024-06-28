import requests

# from dotenv import load_dotenv

# load_dotenv()  # noqa

# from config import settings

base_url = "http://127.0.0.1:8000"


# def test_yoy():
#     resp = requests.post(
#         base_url + "/sales/yoy",
#         json={
#             "brand": "The Indian Garage Co",
#             "gender": "Men",
#             "vertical": "Apparel",
#             "fabric_category": "Woven",
#             "usage": "Casual",
#             "brick": "Topwear",
#             "product": "Blazer",
#             "sub_product": "Blazer",
#             "look": "Casual",
#             "target_audience": "Core",
#             "fit": "Slim",
#             "colour_family": "black",
#             "mrp": [0, 50000],
#             "cost": [1, 10000],
#             "t_from": "2021-09-20",
#             "t_to": "2022-09-20",
#             "metric": "sales",
#             "resolution": "month",
#         },
#     )
#     # assert
#     # print(resp.json())
#     # assert resp.json() == {"ping": "pong"}


# def test_yoy():
#     resp = requests.post(
#         base_url + "/sales/yoy",
#         json={
#             "brand": "The Indian Garage Co",
#             "gender": "Men",
#             "vertical": "Apparel",
#             "fabric_category": "Woven",
#             "usage": "Casual",
#             "brick": "Topwear",
#             "product": "Blazer",
#             "sub_product": "Blazer",
#             "look": "Casual",
#             "target_audience": "Core",
#             "fit": "Slim",
#             "colour_family": "black",
#             "mrp": [0, 50000],
#             "cost": [1, 10000],
#             "t_from": "2021-09-20",
#             "t_to": "2022-09-20",
#             "metric": "profit",
#             "resolution": "week",
#         },
#     )


def test_login():
    resp = requests.get(
        base_url + "/",
        json={},
    )
