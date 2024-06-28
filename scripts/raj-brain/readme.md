# Upload business plan sheet (File name: Business Plan 23-24, Sheet name: Channel level)
# Upload last month bestseller list (File name: Bestseller OTB, Sheet: As on May 23 all the style codes are best sellers)

---

# For Last 6 months

## Upload SOH: (File name: Active cover report june 23, Sheet name: Raw Data, last 3 yr)
    ```(month, channel, ean), qty```
    channels = "b2c" or individual b2b channels
    Channel Header:Fashion Factory Apr 23
    EAN header: ean
    Qty header: value in channel header

## Upload Intransit + pending GRN stock: (File name: Active cover report june 23, Sheet name: Raw Data, current month)
    ```ean, qty```
    Header: Intransit

## Calculate instock%

## Upload WIP: (File name: Active cover report june 23, Sheet name: Raw Data, current month)
    ean, qty
    Header: Total WIP

## Upload GRN (File name: Active cover report june 23, Sheet name: Raw Data, current month)
    ean, qty
    Header: total grn

## Assuming Returns = last month gross sales * 0.35

## Upload day level sales (File name: MOM sale Report)
    channel, date, ean, qty, total_sp

## Calculate max potential for each style

---

# Calculate global inventory sheet based on above data (GIN sheet=Raw Data Sheet from Active cover report june 23)

# Calculate active cover report (File name: Active cover report june 23, Sheet name: June 23 Opening)

# Bestseller Maturity (Filename: New bestseller working Sheetname: Sheet1)

First GRN date - take it from the file for now. Update it in Chanakya app when you get it.
all options  -  remove blocked codes and current bestsellers from this list
Sheet 2:
calculate no of days since first GRN. > 180 days, drop them
map last 6 months GRN quantity and sales
Rate of sale (month sale/30) = avg of the specific month for the last 6 months (Don’t Consider Fashion Factory & Ajio B2B Sales)
Sell through = sold / grn
if sell through < 30% = if GRN quantity > MOQ (merchandising team created more stock than required) (manual bet based on multiple factors)
if sellthrough > 35% within 30 days of launch/ 50%, 60 days/ 70%, 120 days mature to bestseller
if lanuched in last 60days and rating on myntra is 4+, mature to bestseller (manual call)

go through last 3 months ROS
- if it is BAU month, consider > 3 units per day
- if Event month, consider > 6 units per day (June, Dec and Oct) (a month is Event month if > 50% sales is coming from sale days)
mature these styles as bestsellers


# and degrade Maturity (File name: Bestseller OTB, Sheet: As on May 23 if "Blocked Type" is not "Active")
last month's bestseller options
map all channel's this month SOH and WIP
calculate 3 months avg opening inventory
3m avg opening inv > 500 and 3m gross sale < 100 (its not selling even through stock exists - check SOH for each channel. if stock is not able to go live, make it live. if live and not selling, do PLA)
    map maturity month (when a style became a bestseller) < 6 months
    check active cover > 6, 1 month PLA and if doesnt work degrade

# OTB calculation (File name: Bestseller OTB, Sheet: As on May 23, final value: "OTB Quantity")
take bestsellers and their available sizes (sizes for production), sleeve type
map fabric codes, program type, avg fabric consumption, width to these style ids (manually check carefully)
map ajio and myntra id for reference of merchandising team

map MOQ based on business plan (category level not style level)
minimum% i can sell guarentee (manually)
maturity projection = guarentee how much i can sell

fabric count = for each fabric code, how many styles are created

this month SOH for all channels

depth plan for each channel
max(last 30 ROS for each channel, last 60 ROS, last 90 ROS, last 180 ROS) * 90

avg opening inventory for last 3 months
filter the ones > 500 total opening SOH for last 3 months continuously
A = avg ROS last 3 months = (total sales of last 3 months) / 90
B = avg ROS last 1 month = (total sales of last 1 months) / 30
max(A,B)
ROS/month = Avg ROS 3 month * 30

avg stock turn ratio for last 3 months = sold units last 3 months / 3m avg((month opening SOH + closing SOH)/2)

avg sale per day = last 1 yr myntra-MP + ajio-SOR (max sale, 1<max sale, 2<max sale)/3

stock cover = active cover

grading based on ROS/month column
    avg ROS > 500 A+
    avg ROS 300-500 A
    avg ROS 100-300 B
    avg ROS < 100 C

max net sold / month for last 6 months

avg 3 months (gross sales)

returns%

last month gross sales

projected sales (on avg for the next 3m this is the sale per month)
    1. avg opening SOH for last 3 mobths > 500
        max(avg ROS 3m, avg ROS 1m) *75% (considering returns) * 30
    2. stock turn ratio >50% and avg sale per day > 35 and returns% < 40% and 3mavg(opening SOH) > 100
        (avg sale per day * 30) * 130%
    3. rest
        old: 6m max(net sold per month) * 30
        new: all (sum of all 3 channels depth plan) / 3 = 1 month projection
             if projection < maturity proj, cap it at maturity proj

base stock to maintain = projected sales * 3

base stock to maintain = max(depth plan total , base stock to maintain)

end to end base stock req
    if projected sales = 0 , ans = 0
    if projected sales > maturity projection, ans = maturity projection*7
    if projected sales < maturity projection, ans = max(maturity projection*4, sales projection*7)

gap = end to end base stock req - (total SOH + WIP)

basestock + wip gap column (need this much stock imeediately) = base stock to maintain*4 - (total SOH + WIP)

fabric ordered not inhouse = fabric has been ordered but not recieved by tigc (manually add)
fabric ordered not issued = fabric has come but not given for production (manually add)
total fabric in qty = (fabric ordered not inhouse + fabric ordered not issued) / avg consumption of fabric for a style

----
fabric level calculation:

common fabric contribnution = divide (base stock + wip gap) into the style's individual contribution
common fabric in qty in style level =  common fabric contribnution * total fabric in qty

balance to source = end to end base stock req - (total SOH + WIP + common fabric in qty in style level)

OTB =
    if balance to source > 700 then min(base stock to maintain*2.5, balance to source)
    if balance to source < 700 then balance to source

---

# to find dates in master
created_at - find from style code prefix
updated_at - find from style code prefix
no of traded days = today - first sold date

(skipping) first grn - can be found manually
(skipping) first sold date = first go live date - fetch from sales data

## PLA

WH SOH filter What is live on myntra (P(Live) on myntra)
map total last 3m Sales on myntra
forward days on hand = SOH / (3m total sold units / 3)
if forward days on hand>3 PLA

total PLA budget should be max 2% of total targeted GMV in business plan on that month in that channel

new styles we bid 1.5 CPC to get higher visibility

for bestsellers
for high DOH (b/w 500,1000) and low ROS - 1.3 to 1.8 CPC
for high DOH (>1000) and low ROS - 2 to 2.5 CPC

for regular or any cut sizes
CPC 1 - 1.2

Get data from myntra about ROAS

ROI > 18%
rate of sale is picking up so remove PLA next month

ROI < 18%
continue PLA