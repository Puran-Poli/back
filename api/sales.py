import json

from fastapi import APIRouter, Depends, HTTPException

from api.helper import (
    check_if_role_exists,
    get_user_from_token,
)
from api.sales_helper import (
    calculate_metrics,
    get_ids_from_channels,
    make_json,
    run_pdp_queries,
    run_plp_queries,
)
from config import logger

router = APIRouter(prefix="/sales")


@router.post("/{product_id}")
async def get_sales_data(
    product_id: str, filters: dict = None, user=Depends(get_user_from_token)
):
    if filters is None:
        filters = {}
    if not check_if_role_exists(user["department"]["roles"], "sales analytics", "read"):
        raise HTTPException(
            403, "You don't have permission to see sales analysis chart"
        )
    logger.info(f"{user['name']} is trying to see sales analytics of {product_id}")
    filters["channels"] = await get_ids_from_channels(filters["channels"])

    (
        wip,
        grn,
        soh,
        returns,
        sales,
        cost,
    ) = await run_pdp_queries(product_id, filters)

    (
        rate_of_sales,
        revenue,
        rate_of_returns,
        return_ratio,
        net_sales,
        sale_profit,
        instock,
        stock_turn_ratio,
        sell_through_ratio,
    ) = calculate_metrics(
        grn,
        soh,
        returns,
        sales,
        cost,
    )

    return {
        "msg": "Sales data fetched successfully",
        "info": {
            # sales
            "sales": make_json(sales),
            "rate_of_sales": make_json(rate_of_sales),
            "sale_revenue": make_json(revenue),
            "sale_profit": make_json(sale_profit),
            # returns
            "returns": make_json(returns),
            "rate_of_return": make_json(rate_of_returns),
            # sales / return
            "return_ratio": make_json(return_ratio),
            # sales - returns
            "net_sales": make_json(net_sales),
            # soh
            "soh": make_json(soh),
            "instock": make_json(instock),
            "stock_turn": make_json(stock_turn_ratio),
            # grn
            "grn": make_json(grn),
            "sell_though": make_json(sell_through_ratio),
            # wip
            "wip": make_json(wip),
        },
    }


@router.post("/")
async def get_sales_overview(filters: dict = None, user=Depends(get_user_from_token)):
    if filters is None:
        filters = {}
    if not check_if_role_exists(user["department"]["roles"], "sales analytics", "read"):
        raise HTTPException(
            403, "You don't have permission to see sales analysis chart"
        )
    logger.info(f"{user['name']} is trying to see sales overview")
    filters["channels"] = await get_ids_from_channels(filters["channels"])

    (
        soh,
        sales,
        details,
    ) = await run_plp_queries(filters)

    sales = make_json(sales, "product")
    for month in sales:
        month["target"] = 600000

    return {
        "msg": "Sales data fetched successfully",
        "info": {
            # Header 1 - Sales Overview - chart showing stacked bar graph
            #     x axis is month, year (eg 2024-05)
            #     each part of the stack (y axis) is sales of a particular product category (eg shirt)
            #     there is a line representing monthly sale target
            "sales": sales,
            # Header 2 - Stock Overview - chart showing stacked bar graph
            #     x axis is month, year (eg 2024-05)
            #     each part of the stack (y axis) is sales of a particular product category (eg shirt)
            #     no target line
            "soh": make_json(soh, "product"),
            # Header 3 - Detailed view
            "details": json.loads(details.to_json(orient="records")),
        },
    }
