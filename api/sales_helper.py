import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import select

from model import db
from model.db import get_session


async def get_ids_from_channels(channel_names):
    async with get_session() as s:
        channel_name_id_map = select(db.Channel.name, db.Channel.id)
        channel_name_id_map = (await s.execute(channel_name_id_map)).all()
    return [
        next(id for (n, id) in channel_name_id_map if n == name)
        for name in channel_names
    ]


def create_query(table, filters, ean_filter=None, has_channel=False, is_sales=False):
    if is_sales:
        query = select(table.date, table.ean, table.qty, table.sp_per_pc, table.value)
    else:
        query = select(table.date, table.ean, table.qty)
    if has_channel:
        query = query.filter(table.channel_id.in_(filters["channels"]))
    if ean_filter:
        query = query.where(table.ean.in_(ean_filter))
    query = query.filter(
        table.date.between(
            datetime.strptime(filters["month_start"] + "-01", "%Y-%m-%d"),
            datetime.strptime(filters["month_end"] + "-01", "%Y-%m-%d"),
        )
    )
    return query


def make_date(df):
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
    return df


def make_json(df, index_key="ean"):
    df.reset_index(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    df = df.round(2)

    if index_key in df.columns:
        df = df[["date", index_key, "qty"]]
        df = df.pivot(index="date", columns=index_key, values="qty")
        df_dict = df.to_dict("index")
        resp = list(df_dict.values())
        for i, date in enumerate(df_dict.keys()):
            resp[i]["name"] = date
    else:
        # for instock %
        resp = df.to_dict("records")
        resp = [{"name": d["date"], "instock": d["instock"]} for d in resp]

    return resp


def calculate_metrics(grn, soh, returns, sales, cost=None, index_key="ean"):
    returns_idx = returns.set_index(["date", index_key])
    sales_idx = sales.set_index(["date", index_key])
    soh_idx = soh.set_index(["date", index_key])
    grn_idx = grn.set_index(["date", index_key])

    rate_of_sales = sales.copy()
    rate_of_sales["qty"] = sales["qty"] / 30

    sale_revenue = sales.copy()
    sale_revenue.drop(columns=["qty"], inplace=True)
    sale_revenue.rename(columns={"value": "qty"}, inplace=True)
    sale_revenue_idx = sale_revenue.set_index(["date", index_key])

    rate_of_returns = returns.copy()
    rate_of_returns["qty"] = returns["qty"] / 30

    return_ratio = (returns_idx * 100) / sales_idx

    net_sales = sales_idx - returns_idx

    if index_key == "ean":
        sale_cost_idx = sales_idx.copy()
        sale_cost_idx.fillna(0, inplace=True)
        sale_cost_idx["qty"] = sales_idx["qty"] * cost
        sale_profit = sale_revenue_idx - sale_cost_idx

        instock = soh.copy()
        instock["instock"] = instock["qty"] > 0
        instock = instock.groupby("date")["instock"].mean() * 100
        instock = instock.to_frame(name="instock")
    else:
        sale_profit = None
        instock = None

    stock_turn_ratio = (sales_idx * 100) / soh_idx

    sell_through_ratio = (sales_idx * 100) / grn_idx

    return (
        rate_of_sales,
        sale_revenue,
        rate_of_returns,
        return_ratio,
        net_sales,
        sale_profit,
        instock,
        stock_turn_ratio,
        sell_through_ratio,
    )


async def run_pdp_queries(product_id, filters):
    ean_size_map = select(db.Sizes.ean, db.Sizes.standard_size).where(
        db.Sizes.product_id == product_id
    )
    async with get_session() as s:
        ean_size_map = (await s.execute(ean_size_map)).all()
    ean_filter = [i[0] for i in ean_size_map]

    wip_query = create_query(db.WIP, filters, ean_filter)
    grn_query = create_query(db.GRN, filters, ean_filter)
    soh_query = create_query(db.SOH, filters, ean_filter, has_channel=True)
    returns_query = create_query(db.Returns, filters, ean_filter, has_channel=True)
    sales_query = create_query(
        db.Sales, filters, ean_filter, has_channel=True, is_sales=True
    )
    cost_query = select(db.Styles.cost).where(db.Styles.product_id == product_id)

    # each query is 0.7-0.8 sec
    async with get_session() as s:
        wip = pd.DataFrame(
            (await s.execute(wip_query)).all(), columns=["date", "ean", "qty"]
        )
        grn = pd.DataFrame(
            (await s.execute(grn_query)).all(), columns=["date", "ean", "qty"]
        )
        soh = pd.DataFrame(
            (await s.execute(soh_query)).all(), columns=["date", "ean", "qty"]
        )
        returns = pd.DataFrame(
            (await s.execute(returns_query)).all(),
            columns=["date", "ean", "qty"],
        )
        sales = pd.DataFrame(
            (await s.execute(sales_query)).all(),
            columns=["date", "ean", "qty", "sp_per_pc", "value"],
        )
        cost = (await s.execute(cost_query)).first()[0]

    all_months = pd.date_range(
        start=f'{filters["month_start"]}', end=f'{filters["month_end"]}', freq="MS"
    ).strftime("%Y-%m")
    index = pd.MultiIndex.from_tuples(
        list(itertools.product(all_months, [x[1] for x in ean_size_map])),
        names=["date", "ean"],
    )
    empty_df = pd.DataFrame(index=index)

    def group(df):
        if not len(df):
            return df
        df = df.groupby(["date", "ean"]).sum()
        df.reset_index(inplace=True)
        return df

    def fill_gaps(df):
        df["ean"] = df["ean"].map(dict(ean_size_map))
        df.set_index(["date", "ean"], inplace=True)
        temp = empty_df.copy()
        temp = temp.merge(df, on=["date", "ean"], how="left")
        temp.reset_index(inplace=True)
        temp.fillna(0, inplace=True)
        return temp

    return (
        fill_gaps(group(make_date(wip))),
        fill_gaps(group(make_date(grn))),
        fill_gaps(group(make_date(soh))),
        fill_gaps(group(make_date(returns))),
        fill_gaps(group(make_date(sales))),
        cost,
    )


# def get_top_10_from_raw(df):
#     if not len(df):
#         return df

#     df_grouped = df.groupby(["date", "pid"])["qty"].sum().reset_index()
#     df_grouped["rank"] = df_grouped.groupby("date")["qty"].rank(
#         method="dense", ascending=False
#     )
#     df_top10 = df_grouped[df_grouped["rank"] <= 10]
#     df_top10 = df_top10.merge(df[["pid", "product"]].drop_duplicates(), on="pid")
#     df_top10 = df_top10.sort_values(by=["date", "rank"])
#     return df_top10


async def run_plp_queries(filters):
    wip_query = create_query(db.WIP, filters)
    grn_query = create_query(db.GRN, filters)
    soh_query = create_query(db.SOH, filters, has_channel=True)
    returns_query = create_query(db.Returns, filters, has_channel=True)
    sales_query = create_query(db.Sales, filters, has_channel=True, is_sales=True)
    get_ean_pid_map = select(db.Sizes.ean, db.Sizes.product_id)
    get_pid_product_map = select(db.Styles.product_id, db.Styles.product)
    get_distict_products = select(db.Styles.product).distinct()

    # 0.5 to 3.8 sec for each query
    async with get_session() as s:
        wip = pd.DataFrame(
            (await s.execute(wip_query)).all(), columns=["date", "ean", "qty"]
        )
        grn = pd.DataFrame(
            (await s.execute(grn_query)).all(), columns=["date", "ean", "qty"]
        )
        soh = pd.DataFrame(
            (await s.execute(soh_query)).all(), columns=["date", "ean", "qty"]
        )
        returns_df = pd.DataFrame(
            (await s.execute(returns_query)).all(),
            columns=["date", "ean", "qty"],
        )
        sales = pd.DataFrame(
            (await s.execute(sales_query)).all(),
            columns=["date", "ean", "qty", "sp_per_pc", "value"],
        )
        ean_pid_map = (await s.execute(get_ean_pid_map)).all()
        pid_product_map = (await s.execute(get_pid_product_map)).all()
        products = (await s.execute(get_distict_products)).all()

    def map_pids(df):
        df["pid"] = df["ean"].map(dict(ean_pid_map))
        df["product"] = df["pid"].map(dict(pid_product_map))
        return df

    wip = map_pids(wip)
    grn = map_pids(grn)
    soh = map_pids(soh)
    returns_df = map_pids(returns_df)
    sales = map_pids(sales)

    unique_pids = set()
    for df in [sales, returns_df]:
        unique_pids.update(df["pid"].unique())
    unique_pids = [x for x in unique_pids if isinstance(x, str)]

    async with get_session() as s:
        pid_cost_map = (
            await s.execute(
                select(db.Styles.product_id, db.Styles.cost).where(
                    db.Styles.product_id.in_(unique_pids)
                )
            )
        ).all()

    # sales["cost"] = sales["pid"].map(dict(pid_cost_map))

    all_months = pd.date_range(
        start=f'{filters["month_start"]}', end=f'{filters["month_end"]}', freq="MS"
    ).strftime("%Y-%m")
    index = pd.MultiIndex.from_tuples(
        list(itertools.product(all_months, [x[0] for x in products])),
        names=["date", "product"],
    )
    empty_df = pd.DataFrame(index=index)

    def group_date_product(df):
        if not len(df):
            return df
        df = df.groupby(["date", "product"]).sum()
        df.reset_index(inplace=True)
        return df

    def group_date_pid(df):
        if not len(df):
            return df
        df = df.groupby(["date", "pid"]).sum()
        df.reset_index(inplace=True)
        df.drop(columns=["ean"], inplace=True)
        df["product"] = df.pid.map(dict(pid_product_map))
        return df

    def fill_gaps(df):
        df.set_index(["date", "product"], inplace=True)
        temp = empty_df.copy()
        temp = temp.merge(df, on=["date", "product"], how="left")
        temp.reset_index(inplace=True)
        temp.fillna(0, inplace=True)
        return temp

    # Header 3 - Detailed view - table with filters and sorting
    #     response table will include:
    #         -------- product info
    #         img link - if you can show the image, better, otherwise optional x
    #         pdp link - to connect to the chanakya pdp page.
    #         style id x
    #         grading - A+, B, C etc
    #         -------- sales metrics
    #         sales qty.
    #         returns qty.
    #         net qty = sales - returns.
    #         return ratio = returns/sales.
    #         -------- stock metrics
    #         SOH.
    #         GRN.
    #         WIP.
    #         Instock.
    #         -------- advanced metrics
    #         sell through = sales / grn.
    #         stock turn = sales / soh.
    #         active cover.

    #     frontend based sorting available on all columns
    #     filters available in the api:
    #         product type: type=string (not a list), default=can be shirt?
    #         channel: type=list, default=all channels
    #         month: type=string (2024-04), default=last month

    # merge raw tables
    details = group_date_pid(make_date(sales))
    details.drop(columns=["sp_per_pc"], inplace=True)
    details.rename(columns={"value": "revenue_sales"}, inplace=True)
    details["cost_per_pc"] = details["pid"].map(dict(pid_cost_map))
    details["cost_sales"] = details["qty"] * details["cost_per_pc"]
    details["profit_sales"] = details["revenue_sales"] - details["cost_sales"]
    details.drop(columns=["cost_per_pc"], inplace=True)

    wip_grouped = group_date_pid(make_date(wip))
    details = details.merge(
        wip_grouped, on=["date", "pid"], how="outer", suffixes=("_sales", "_wip")
    )

    grn_grouped = group_date_pid(make_date(grn))
    details = details.merge(grn_grouped, on=["date", "pid"], how="outer")
    details.rename(columns={"qty": "qty_grn"}, inplace=True)
    details.rename(columns={"product": "product_grn"}, inplace=True)

    soh_grouped = group_date_pid(make_date(soh))
    details = details.merge(soh_grouped, on=["date", "pid"], how="outer")
    details.rename(columns={"qty": "qty_soh"}, inplace=True)
    details.rename(columns={"product": "product_soh"}, inplace=True)

    returns_df = group_date_pid(make_date(returns_df))
    details = details.merge(returns_df, on=["date", "pid"], how="outer")
    details.rename(columns={"qty": "qty_returns"}, inplace=True)
    details.rename(columns={"product": "product_returns"}, inplace=True)

    # generate metrics
    details["qty_sales"].fillna(0, inplace=True)
    details["qty_returns"].fillna(0, inplace=True)
    details["qty_net"] = details["qty_sales"] - details["qty_returns"]
    details["return_ratio"] = details["qty_returns"] / details["qty_sales"]
    details["qty_wip"].fillna(0, inplace=True)
    details["qty_soh"].fillna(0, inplace=True)
    details["qty_grn"].fillna(0, inplace=True)
    details["sell_through"] = details["qty_sales"] / details["qty_grn"]
    details["stock_turn"] = details["qty_sales"] / details["qty_soh"]

    # have only one product column
    details["product"] = details["product_sales"]
    details["product"].fillna(details["product_wip"], inplace=True)
    details["product"].fillna(details["product_grn"], inplace=True)
    details["product"].fillna(details["product_soh"], inplace=True)
    details["product"].fillna(details["product_returns"], inplace=True)
    details.drop(
        columns=[
            "product_sales",
            "product_wip",
            "product_grn",
            "product_soh",
            "product_returns",
        ],
        inplace=True,
    )

    if "ean" in details.columns:
        details.drop(columns=["ean"], inplace=True)

    # ean_size_map = select(db.Sizes.ean, db.Sizes.standard_size).where(
    #     db.Sizes.product_id == product_id
    # )
    # async with get_session() as s:
    #     ean_size_map = (await s.execute(ean_size_map)).all()

    # def fill_ean_gaps(df):
    #     df["ean"] = df["ean"].map(dict(ean_size_map))
    #     df.set_index(["date", "ean"], inplace=True)
    #     temp = empty_df.copy()
    #     temp = temp.merge(df, on=["date", "ean"], how="left")
    #     temp.reset_index(inplace=True)
    #     temp.fillna(0, inplace=True)
    #     return temp

    # instock = soh.copy()
    # instock["instock"] = instock["qty"] > 0
    # instock = instock.groupby("date")["instock"].mean() * 100
    # instock = instock.to_frame(name="instock")

    # instock %
    instock = soh.copy()
    # unique_pids = instock.eans.unique()

    instock = make_date(instock)

    instock["instock"] = instock["qty"] > 0

    instock = instock.groupby(["date", "pid"])["instock"].mean() * 100
    instock = instock.to_frame(name="qty")
    instock.reset_index(inplace=True)

    details = details.merge(instock, on=["date", "pid"], how="outer")
    details.rename(columns={"qty": "instock_pct"}, inplace=True)

    # active cover
    details.sort_values(by=["pid", "date"], inplace=True)
    rolling_avg = (
        details.groupby("pid")["qty_sales"].rolling(window=3, min_periods=1).mean()
    )
    rolling_avg = rolling_avg.reset_index(level=0, drop=True)
    details["3m_rolling_avg_sales"] = rolling_avg
    details["active_cover"] = details["qty_soh"] / details["3m_rolling_avg_sales"]

    # grading
    details.sort_values(
        by=["product", "date", "qty_sales"], ascending=[True, True, False], inplace=True
    )

    def assign_grades(group):
        cutoffs = np.percentile(
            group["qty_sales"], [90, 70, 40]
        )  # 90th, 70th, 40th percentile
        conditions = [
            group["qty_sales"] > cutoffs[0],
            (group["qty_sales"] <= cutoffs[0]) & (group["qty_sales"] > cutoffs[1]),
            (group["qty_sales"] <= cutoffs[1]) & (group["qty_sales"] > cutoffs[2]),
            group["qty_sales"] <= cutoffs[2],
        ]
        grades = ["A+", "A", "B", "C"]
        group["grade"] = np.select(conditions, grades, default="C")
        return group

    details = details.groupby(["product", "date"]).apply(assign_grades)

    details.replace([np.inf, -np.inf], np.nan, inplace=True)
    if "ean_x" in details.columns:
        details.drop(columns=["ean_x"], inplace=True)
    if "ean_y" in details.columns:
        details.drop(columns=["ean_y"], inplace=True)
    details = details.round(2)

    return (
        fill_gaps(group_date_product(make_date(soh.copy()))),
        fill_gaps(group_date_product(make_date(sales.copy()))),
        details,
    )
