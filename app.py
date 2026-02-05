from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import yfinance as yf
import csv
import math
import statistics


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return f.read()

# ---------------- STOCK UNIVERSE ----------------
STOCK_LIST = []

def load_stock_universe():
    global STOCK_LIST
    STOCK_LIST = []
    try:
        with open("nse.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                STOCK_LIST.append({
                    "symbol": row["SYMBOL"],
                    "name": row["NAME OF COMPANY"]
                })
    except Exception as e:
        print("Error loading NSE universe:", e)

load_stock_universe()

# ---------------- INPUT MODEL ----------------
class ValuationInput(BaseModel):
    symbol: str

# ---------------- HELPERS ----------------
def safe(x):
    try:
        if x is None or math.isnan(x):
            return 0
        return float(x)
    except:
        return 0

def get_stock(symbol):
    for suffix in [".NS", ".BO"]:
        try:
            stock = yf.Ticker(symbol + suffix)
            info = stock.info
            if info and info.get("currentPrice"):
                return stock, info
        except:
            pass
    return None, None

def get_cash(stock):
    """
    Fetch 'Cash And Cash Equivalents' from the latest balance sheet.
    Returns value in Cr (₹ Crores) as float.
    """
    try:
        bs = stock.balance_sheet
        if bs.empty:
            return 0.0

        # latest column is usually the first one
        latest_col = bs.columns[0]

        if "Cash And Cash Equivalents" in bs.index:
            cash_value = bs.loc["Cash And Cash Equivalents", latest_col]
            if cash_value is None or math.isnan(cash_value):
                return 0.0
            return float(cash_value) / 1e7
    except Exception as e:
        print("Error fetching cash:", e)

    return 0.0


def clamp_pct(x, low=-0.10, high=0.10):
    return max(low, min(high, x))



# ---------------- Capital Structure ----------------
def calculate_capital_structure(info, stock=None):
    debt, equity = 0, 0

    if stock is not None:
        try:
            bs = stock.balance_sheet
            if not bs.empty:
                latest_col = bs.columns[0]

                # Debt (YFinance is already in thousands INR)
                if "Long Term Debt" in bs.index:
                    debt = safe(bs.loc["Long Term Debt", latest_col])
                elif "Long Term Debt And Capital Lease Obligation" in bs.index:
                    debt = safe(bs.loc["Long Term Debt And Capital Lease Obligation", latest_col])
                else:
                    debt = 0

                # Equity
                for name in ["Stockholders' Equity", "Common Stock Equity", "Total Stockholder Equity"]:
                    if name in bs.index:
                        equity = safe(bs.loc[name, latest_col])
                        break

        except Exception as e:
            print("Error fetching balance sheet:", e)

    # fallback to book value (in INR) → convert to thousands
    if equity == 0:
        equity = safe(info.get("bookValue", 0)) / 1000

    total = debt + equity if (debt + equity) > 0 else 1
    w_debt = debt / total
    w_equity = equity / total

    return {
        "debt": round(debt / 1e7, 2),       # Convert to Cr
        "equity": round(equity / 1e7, 2),   # Convert to Cr
        "w_debt": round(w_debt, 4),
        "w_equity": round(w_equity, 4)
    }






def calculate_wacc(w_debt, w_equity, cost_debt=0.08, cost_equity=0.12):
    wacc = w_debt * cost_debt + w_equity * cost_equity
    return round(wacc, 4)



def calculate_growth(series):
    s = [x for x in series if x > 0]
    n = len(s)
    if n >= 6:
        return (s[-1] / s[-6]) ** (1/5) - 1
    elif n >= 4:
        return (s[-1] / s[-4]) ** (1/3) - 1
    elif n >= 2:
        return (s[-1] / s[-2]) - 1
    return 0

def safe_get(df, key, col):
    try:
        v = df.loc[key, col]
        if v is None or math.isnan(v):
            return 0
        return float(v)
    except:
        return 0

def get_latest_cf(stock):
    """
    Fetch latest cash flow drivers: Depreciation, CapEx, Change in NWC.
    Pulls from Cash Flow first, then Income Statement if missing.
    Returns values in ₹ Cr.
    """
    cf = stock.cashflow
    fin = stock.financials
    bs = stock.balance_sheet

    latest_col_cf = cf.columns[0] if not cf.empty else None
    latest_col_fin = fin.columns[0] if not fin.empty else None
    bs_cols = bs.columns.tolist() if not bs.empty else []

    # ---- Depreciation ----
    dep = 0
    if cf is not None and latest_col_cf:
        for l in ["Depreciation And Amortization","Depreciation","Amortization"]:
            if l in cf.index:
                dep = safe(cf.loc[l, latest_col_cf])
                break

    # fallback to Income Statement
    if dep == 0 and fin is not None and latest_col_fin:
        for l in ["Depreciation","Depreciation And Amortization","Amortization"]:
            if l in fin.index:
                dep = safe(fin.loc[l, latest_col_fin])
                break

    # ---- CapEx ----
    capex = 0
    if cf is not None and latest_col_cf:
        for l in ["Capital Expenditure", "Capital Expenditures"]:
            if l in cf.index:
                capex = abs(safe(cf.loc[l, latest_col_cf]))
                break

    def get_change_in_nwc(bs):
        bs_cols = bs.columns.tolist()
        if len(bs_cols) < 2:
            return 0

        current_assets_names = ["Total Current Assets", "Current Assets"]
        current_liabilities_names = ["Total Current Liabilities", "Current Liabilities"]

        ca_now = ca_prev = cl_now = cl_prev = 0

        for name in current_assets_names:
            if name in bs.index:
                ca_now = safe(bs.loc[name, bs_cols[0]])
                ca_prev = safe(bs.loc[name, bs_cols[1]])
                break

        for name in current_liabilities_names:
            if name in bs.index:
                cl_now = safe(bs.loc[name, bs_cols[0]])
                cl_prev = safe(bs.loc[name, bs_cols[1]])
                break

        return (ca_now - cl_now) - (ca_prev - cl_prev)


    change_nwc = 0
    if bs is not None and len(bs_cols) >= 2:
        change_nwc = get_change_in_nwc(bs)



    return {
        "year": "Latest",
        "depreciation": round(dep / 1e7, 2),
        "capex": round(capex / 1e7, 2),
        "change_nwc": round(change_nwc / 1e7, 2)
    }



def get_historical_financials(stock, is_financial=False):
    """
    Fetch historical financials and latest cashflow drivers.
    Returns:
        annual_data: List of dicts for past years
        latest_cf: Dict for latest cashflow drivers
    """
    try:
        fin = stock.financials
        cf = stock.cashflow
        bs = stock.balance_sheet
        if fin.empty or cf.empty or bs.empty:
            return [], {}
    except Exception as e:
        print("Financial fetch error:", e)
        return [], {}

    annual_data = []

    # Columns for balance sheet (latest → oldest)
    bs_cols = bs.columns.tolist()

    # Annual financials (oldest → latest)
    years = list(fin.columns[::-1])

    for y in years:
        revenue = safe_get(fin, "Total Revenue", y)
        if revenue <= 0:
            continue

        operating_income = safe_get(fin, "Operating Income", y)
        if operating_income == 0:
            operating_income = revenue * 0.15  # fallback 15%

        pat = safe_get(fin, "Net Income", y)

        depreciation = safe_get(cf, "Depreciation And Amortization", y)
        if depreciation == 0:
            depreciation = operating_income * 0.12  # fallback 12%

        capex = abs(safe_get(cf, "Capital Expenditure", y))

        # Working Capital change (year-specific)
        try:
            col_idx = bs_cols.index(y)
            if col_idx + 1 < len(bs_cols):
                wc_now = safe_get(bs, "Working Capital", bs_cols[col_idx])
                wc_prev = safe_get(bs, "Working Capital", bs_cols[col_idx + 1])
                change_nwc = wc_now - wc_prev
            else:
                change_nwc = 0
        except:
            change_nwc = 0

        opm = operating_income / revenue if revenue else 0

        annual_data.append({
            "year": str(y.year),
            "revenue": revenue / 1e7,
            "operating_income": operating_income / 1e7,
            "pat": pat / 1e7,
            "depreciation": depreciation / 1e7,
            "capex": capex / 1e7,
            "change_nwc": change_nwc / 1e7,
            "opm": opm
        })

    # ---- Latest cashflow drivers ----
    try:
        latest_col = cf.columns[0]  # latest column
        latest_dep = safe_get(cf, "Depreciation And Amortization", latest_col)
        latest_capex = abs(safe_get(cf, "Capital Expenditure", latest_col))

        # Latest NWC change
        if len(bs_cols) >= 2:
            latest_nwc = safe_get(bs, "Working Capital", bs_cols[0]) - safe_get(bs, "Working Capital", bs_cols[1])
        else:
            latest_nwc = 0

        latest_cf = {
            "year": "Latest",
            "depreciation": round(latest_dep / 1e7, 2),
            "capex": round(latest_capex / 1e7, 2),
            "change_nwc": round(latest_nwc / 1e7, 2)
        }
    except Exception as e:
        print("Latest CF error:", e)
        latest_cf = {}

    return annual_data, latest_cf



    # ---- Latest (TTM-ish) ----
    latest_cf = {}
    try:
        latest_col = fin.columns[0]
        latest_cf = {
            "depreciation": round(safe_get(cf, "Depreciation And Amortization", latest_col), 2),
            "capex": round(safe_get(cf, "Capital Expenditure", latest_col), 2),
            "change_nwc": round(safe_get(cf, "Change In Working Capital", latest_col), 2),
            "year": "Latest"
        }
    except:
        latest_cf = {}




    # ---- Annual Data ----
    years = list(fin.columns[::-1])  # Oldest → Latest
    for y in years:
        revenue = safe_get(fin, "Total Revenue", y) 
        if revenue == 0:
            continue  # Skip this year completely
        operating_income = safe_get(fin, "Operating Income", y)
        pat = safe_get(fin, "Net Income", y)
        depreciation = safe_get(cf, "Depreciation", y)

        if operating_income == 0 and revenue != 0:
            operating_income = revenue * 0.15
        if depreciation == 0:
            depreciation = operating_income * 0.12

        opm = operating_income / revenue if revenue else 0

        capex = abs(safe_get(cf, "Capital Expenditure", y))
        bs = stock.balance_sheet
        bs_cols = bs.columns.tolist()  # latest → oldest

        wc_now = safe_get(bs, "Working Capital", bs_cols[0])
        wc_prev = safe_get(bs, "Working Capital", bs_cols[1])

        change_nwc = wc_now - wc_prev




        annual_data.append({
            "year": str(y.year),
            "revenue": revenue,
            "operating_income": operating_income,
            "pat": pat,
            "depreciation": depreciation,
            "capex": capex,
            "change_nwc": change_nwc,
            "opm": opm
        })


    # ---- TTM Data ----
    try:
        ttm_col = fin.columns[0]  # yfinance already has TTM/latest column
        ttm_revenue = safe_get(fin, "Total Revenue", ttm_col)
        ttm_op_income = safe_get(fin, "Operating Income", ttm_col)
        ttm_pat = safe_get(fin, "Net Income", ttm_col)
        ttm_depreciation = safe_get(cf, "Depreciation", ttm_col)
        if ttm_op_income == 0 and ttm_revenue != 0:
            ttm_op_income = ttm_revenue * 0.15
        if ttm_depreciation == 0:
            ttm_depreciation = ttm_op_income * 0.12
        ttm_opm = ttm_op_income / ttm_revenue if ttm_revenue else 0

        ttm_data = {
            "year": "TTM",
            "revenue": ttm_revenue,
            "operating_income": ttm_op_income,
            "pat": ttm_pat,
            "depreciation": ttm_depreciation,
            "opm": ttm_opm
        }
    except:
        ttm_data = None

    return annual_data, ttm_data

# ---------------- SEARCH ----------------
@app.get("/search")
def search_company(q: str):
    q = q.lower()
    results = []
    for s in STOCK_LIST:
        if q in s["symbol"].lower() or q in s["name"].lower():
            results.append(s)
            if len(results) >= 8:
                break
    return results

# ---------------- VALUATION ----------------
@app.post("/value")
def value(input: ValuationInput):
    stock, info = get_stock(input.symbol)
    if not info:
        return {"error": True, "message": "Invalid symbol"}

    company_name = info.get("longName", input.symbol)

    financial_keywords = [
        "finance","financial","bank","insurance","broking","broker",
        "nbfc","holding","holdings","capital","investment","securities","financial services","lending"
    ]
    is_financial = any(k in company_name.lower() for k in financial_keywords)

    book_value = safe(info.get("bookValue"))
    shares = safe(info.get("sharesOutstanding", 1)) / 1e7  # convert to Crore shares
    current_price = safe(info.get("currentPrice"))
    cash = get_cash(stock)
    capital_structure = calculate_capital_structure(info, stock)
    debt = capital_structure["debt"]   # already in ₹ Cr


    # ---------------- Capital Structure & WACC ----------------
    capital_structure = calculate_capital_structure(info, stock)
    wacc = calculate_wacc(capital_structure["w_debt"], capital_structure["w_equity"])



    annual_data, _ = get_historical_financials(stock, is_financial)
    latest_cf = get_latest_cf(stock)

    five_years = annual_data[-5:] if len(annual_data) >= 5 else annual_data



    # ---------------- Cash Flow Driver Ratios (4Y Median) ----------------
    dep_pct = capex_pct = nwc_pct = 0
    median_revenue = median_dep = median_capex = median_nwc = 0

    if len(five_years) >= 3:
        last_4 = five_years[-4:]

        revenues = [y["revenue"] for y in last_4 if y["revenue"] > 0]
        deps = [y["depreciation"] for y in last_4]
        capex = [abs(y["capex"]) for y in last_4]
        
        # ---- Correct NWC median: derive from WC levels (3 deltas from 4 years) ----
        bs = stock.balance_sheet
        bs_cols = bs.columns.tolist()  # latest → oldest

        nwc_changes = []
        for i in range(3):  # 4 years → 3 changes
            wc_now = safe_get(bs, "Working Capital", bs_cols[i])
            wc_prev = safe_get(bs, "Working Capital", bs_cols[i + 1])
            if wc_now != 0 or wc_prev != 0:
                nwc_changes.append((wc_now - wc_prev) / 1e7)  # convert to ₹ Cr

        mean_nwc = statistics.mean(nwc_changes) if nwc_changes else 0
        median_nwc = mean_nwc


        if revenues:
            median_revenue = statistics.median(revenues)
            median_dep = statistics.median(deps)
            median_capex = statistics.median(capex)
            

            if is_financial:
                raw_dep_pct = 0.02
            else:
                raw_dep_pct = median_dep / median_revenue if median_dep > 0 else 0.055

            raw_capex_pct = median_capex / median_revenue
            raw_nwc_pct = mean_nwc / median_revenue

            dep_pct = clamp_pct(raw_dep_pct)
            capex_pct = clamp_pct(raw_capex_pct)
            nwc_pct = clamp_pct(raw_nwc_pct)


            
    else:
        # fallback to latest year
        latest = five_years[-1]
        rev = latest["revenue"]

        dep_pct = clamp_pct(latest["depreciation"] / rev) if rev > 0 else 0.055
        capex_pct = clamp_pct(abs(latest["capex"]) / rev) if rev > 0 else 0
        nwc_pct = clamp_pct(latest["change_nwc"] / rev) if rev > 0 else 0


    if is_financial:
        nwc_pct = 0
        median_nwc = 0



    revenue_series = [x["revenue"] for x in five_years]
    pat_series = [x["pat"] for x in five_years]

    # Growth calculation
    growth = calculate_growth(revenue_series)
    if growth <= 0:
        growth = calculate_growth(pat_series)
    if growth <= 0:
        growth = 0.12
    growth = min(max(growth, 0.02), 0.15)  # clamp 2%-15%

    discount = wacc
    terminal_growth = 0.04
    tax_rate = 0.25

    # ---------------- Determine margins correctly ----------------
    # Operating Margin (for FCFF)
    op_margins = [x["opm"] for x in five_years if x["opm"] > 0]
    operating_margin = statistics.median(op_margins) if op_margins else 0.15
    operating_margin = min(max(operating_margin, 0.02), 0.60)

    # Net Margin (for FCFE)
    net_margins = []
    for rev, pat in zip(revenue_series, pat_series):
        if rev > 0:
            net_margins.append(pat / rev)

    net_margin = statistics.median(net_margins) if net_margins else 0.10
    net_margin = min(max(net_margin, 0.01), 0.40)

    # Select margin for model
    base_margin = operating_margin
    key_margin = net_margin if is_financial else operating_margin






    # ======================= FCFF MODEL TABLE =======================
    revenue = revenue_series[-1] if revenue_series else 0

    fcff_table = {
        "years": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
        "rows": {
            "Revenue": [],
            "Operating Margin": [],
            "Operating Income": [],
            "Tax Rate": [],
            "Tax": [],
            "NOPAT": [],
            "Depreciation": [],
            "Capex": [],
            "Change in NWC": [],
            "FCFF": [],
            "Discount Factor": [],
            "Present Value of FCFF": []
        }
    }

    for i in range(1, 6):
        rev = revenue * ((1 + growth) ** i)
        op_income = rev * base_margin
        tax = op_income * tax_rate
        nopat = op_income - tax

        # Use SAME reinvestment logic as FCFE
        depreciation = rev * dep_pct
        capex = rev * capex_pct
        change_nwc = 0 if is_financial else rev * nwc_pct


        fcff = nopat + depreciation - capex - change_nwc


        discount_factor = 1 / ((1 + discount) ** i)
        pv_fcff = fcff * discount_factor

        # Fill table
        fcff_table["rows"]["Revenue"].append(round(rev, 2))
        fcff_table["rows"]["Operating Margin"].append(round(base_margin * 100, 2))
        fcff_table["rows"]["Operating Income"].append(round(op_income, 2))
        fcff_table["rows"]["Tax Rate"].append(round(tax_rate * 100, 2))
        fcff_table["rows"]["Tax"].append(round(tax, 2))
        fcff_table["rows"]["NOPAT"].append(round(nopat, 2))
        fcff_table["rows"]["FCFF"].append(round(fcff, 2))
        fcff_table["rows"]["Discount Factor"].append(round(discount_factor, 4))
        fcff_table["rows"]["Present Value of FCFF"].append(round(pv_fcff, 2))
        fcff_table["rows"]["Depreciation"].append(round(depreciation, 2))
        fcff_table["rows"]["Capex"].append(round(capex, 2))
        fcff_table["rows"]["Change in NWC"].append(round(change_nwc, 2))


        # Sum of PV of yearly FCFF
        pv_sum_fcff = sum(fcff_table["rows"]["Present Value of FCFF"])



    # Terminal & equity values
    # Use Year 5 FCFF directly
    year5_fcff = fcff_table["rows"]["FCFF"][-1]

    terminal_value_fcff = (
        year5_fcff * (1 + terminal_growth)
    ) / (discount - terminal_growth)

    pv_terminal_fcff = terminal_value_fcff / ((1 + discount) ** 5)



    enterprise_value = sum(fcff_table["rows"]["Present Value of FCFF"]) + pv_terminal_fcff
    equity_value = enterprise_value - debt + cash
    fcff_value = equity_value / shares

    # FCFE
    last_pat = pat_series[-1] if pat_series else 0
    fcfe_list = [last_pat * ((1 + growth) ** i) for i in range(1, 6)]
    pv_fcfe = sum(f / ((1 + discount) ** t) for t, f in enumerate(fcfe_list, 1))
    terminal_fcfe = fcfe_list[-1] * (1 + terminal_growth) / (discount - terminal_growth)
    pv_terminal_fcfe = terminal_fcfe / ((1 + discount) ** 5)
    fcfe_value = (pv_fcfe + pv_terminal_fcfe) / shares

    # ---------------- Upside Calculations ----------------
    fcff_upside = (fcff_value - current_price) / current_price * 100
    fcfe_upside = (fcfe_value - current_price) / current_price * 100

    # ---------------- Smarter Model Selection ----------------
    if is_financial:
        # ---------------- Banking / NBFC Logic ----------------

        fcfe_upside = (fcfe_value - current_price) / current_price

        # Case 1: FCFE too optimistic → anchor to Book Value
        if fcfe_upside > 0.50:
            intrinsic = book_value
            model = "Book Value"

        # Case 2: Both FCFE & Book below CMP → pick higher
        elif fcfe_value < current_price and book_value < current_price:
            if fcfe_value >= book_value:
                intrinsic = fcfe_value
                model = "FCFE"
            else:
                intrinsic = book_value
                model = "Book Value"

        # Case 3: Normal banking valuation → FCFE
        else:
            intrinsic = fcfe_value
            model = "FCFE"

    else:
        # ---------------- Non-Financial Selection Logic ----------------

        candidates = {
            "FCFF": fcff_value,
            "FCFE": fcfe_value,
            "Book Value": book_value
        }

        # Step 1: check if ALL are below market
        below_market = {k: v for k, v in candidates.items() if v <= current_price}

        if len(below_market) == 3:
            # pick the highest intrinsic below market
            model, intrinsic = max(below_market.items(), key=lambda x: x[1])

        else:
            # normal priority logic
            intrinsic = fcff_value
            model = "FCFF"

            if intrinsic > current_price or intrinsic < book_value:
                intrinsic = fcfe_value
                model = "FCFE"

                if intrinsic > current_price or intrinsic < book_value:
                    intrinsic = book_value
                    model = "Book Value"


    upside = (intrinsic - current_price) / current_price * 100
    valuation_model = model
    BASE_PRICE = intrinsic

    

    # ---------------- Base Parameters ----------------
    base_growth = growth
    base_wacc = discount



    # ---- Force sensitivity engine to match FCFF fair value ----
    def calibrate_base_fcff(target_price, g, wacc):
        if wacc <= terminal_growth:
            return 0

        factor = 0
        for i in range(1, 6):
            factor += ((1 + g) ** i) / ((1 + wacc) ** i)

        tv_factor = ((1 + g) ** 5) * (1 + terminal_growth) / (wacc - terminal_growth)
        tv_factor /= ((1 + wacc) ** 5)

        total_factor = factor + tv_factor

        # target_price is per share → multiply by shares to get total equity value
        base_fcff = (target_price * shares) / total_factor
        return base_fcff


    # ---------------- Calibrate BASE cash flow ----------------
    BASE_FCFF = calibrate_base_fcff(fcff_value, base_growth, base_wacc)
    BASE_FCFE = last_pat






    def calc_price(g, wacc, valuation_model, custom_fcf=None):
        if wacc <= terminal_growth:
            return None

        if valuation_model == "FCFE":
            base = BASE_FCFE if custom_fcf is None else custom_fcf
        else:
            base = BASE_FCFF if custom_fcf is None else custom_fcf

        flows = [base * ((1 + g) ** i) for i in range(1, 6)]
        pv = sum(f / ((1 + wacc) ** t) for t, f in enumerate(flows, 1))
        tv = flows[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv / ((1 + wacc) ** 5)

        return round((pv + pv_tv) / shares, 2)



    # Growth Sensitivity
    growth_points = sorted(set([0.01, 0.05, 0.08, 0.10, base_growth, 0.15, 0.18, 0.20]))
    growth_labels = [f"Base({round(base_growth*100,2)}%)" if abs(g-base_growth)<1e-6 else f"{round(g*100,1)}%" for g in growth_points]
    growth_values = [calc_price(g, base_wacc, model) for g in growth_points]

    # ---------------- Margin Sensitivity (Dynamic) ----------------
    if valuation_model == "FCFE":
        sensitivity_base_margin = net_margin
        base_cashflow = BASE_FCFE
        margin_label_name = "Net Margin Sensitivity"
    else:
        sensitivity_base_margin = base_margin
        base_cashflow = BASE_FCFF
        margin_label_name = "Operating Margin Sensitivity"



    margin_multipliers = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    margin_labels = [
        f"Base({round(sensitivity_base_margin*100,2)}%)"
        if abs(m - 1.0) < 1e-6
        else f"{round(sensitivity_base_margin*m*100,2)}%"
        for m in margin_multipliers
    ]

    margin_values = [
        calc_price(
            base_growth,
            base_wacc,
            model,
            base_cashflow * m
        )
        for m in margin_multipliers
    ]


    # WACC Sensitivity
    wacc_points = [0.06, 0.08, 0.10, base_wacc, 0.14, 0.16, 0.18, 0.20]
    wacc_labels = ["6%", "8%", "10%", f"Base({round(base_wacc*100,2)}%)", "14%", "16%", "18%", "20%"]
    wacc_values = [calc_price(base_growth, w, model) for w in wacc_points]



    # ======================= FCFE MODEL TABLE =======================
    last_revenue = revenue_series[-1] if revenue_series else 0


    fcfe_table = {
        "years": ["Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
        "rows": {
            "Revenue": [],
            "Net Margin (%)": [],
            "Net Income": [],
            "Depreciation": [],
            "Capex": [],
            "Change in NWC": [],
            "FCFE": [],
            "Discount Factor": [],
            "Present Value of FCFE": []
        }
    }

    for i in range(1, 6):
        rev = last_revenue * ((1 + growth) ** i)
        net_income = rev * net_margin

        depreciation_fcfe = rev * dep_pct
        capex_fcfe = rev * capex_pct
        change_nwc_fcfe = 0 if is_financial else rev * nwc_pct

        fcfe = (
            net_income
            + depreciation_fcfe
            - capex_fcfe
            - change_nwc_fcfe
        )

        discount_factor = 1 / ((1 + discount) ** i)
        pv_fcfe = fcfe * discount_factor

        fcfe_table["rows"]["Revenue"].append(round(rev, 2))
        fcfe_table["rows"]["Net Margin (%)"].append(round(net_margin * 100, 2))
        fcfe_table["rows"]["Net Income"].append(round(net_income, 2))
        fcfe_table["rows"]["Depreciation"].append(round(depreciation_fcfe, 2))
        fcfe_table["rows"]["Capex"].append(round(capex_fcfe, 2))
        fcfe_table["rows"]["Change in NWC"].append(round(change_nwc_fcfe, 2))
        fcfe_table["rows"]["FCFE"].append(round(fcfe, 2))
        fcfe_table["rows"]["Discount Factor"].append(round(discount_factor, 4))
        fcfe_table["rows"]["Present Value of FCFE"].append(round(pv_fcfe, 2))



    # ---- FCFE totals MUST be outside loop ----
    pv_sum_fcfe = sum(fcfe_table["rows"]["Present Value of FCFE"])

    terminal_value_fcfe = fcfe_table["rows"]["FCFE"][-1] * (1 + terminal_growth) / (discount - terminal_growth)
    pv_terminal_fcfe = terminal_value_fcfe / ((1 + discount) ** 5)

    equity_value_fcfe = pv_sum_fcfe + pv_terminal_fcfe

        
    # ---- FINAL DISPLAY EPS (Single Source of Truth) ----
    fcff_eps_display = round(fcff_value, 2)
    fcfe_eps_display = round(fcfe_value, 2)



    

    # ---------------- Sensitivity Anchor ----------------
    if model == "FCFF":
        sensitivity_base = fcff_value
    elif model == "FCFE":
        sensitivity_base = fcfe_value
    else:  # Book Value case
        # Still anchor sensitivity to cashflow model
        sensitivity_base = fcfe_value if is_financial else fcff_value

    # Determine label and value
    key_margin_label = "Net Margin Used" if is_financial else "Operating Margin Used"



    # Create the return dict
    result = {
    "company_name": company_name,
    "model_used": model,
    "intrinsic_value_per_share": round(intrinsic, 2),
    "current_price": round(current_price, 2),
    "upside_percent": round(upside, 2),
    "revenue": revenue,
    "growth_rate": growth,
    "discount_rate": discount,
    "terminal_growth": terminal_growth,
    "book_value": round(book_value, 2),
    "shares": shares,
    "cash": cash,
    "debt": debt,

    "cashflow_drivers": {
        "year": "4Y Median",
        "revenue_median": round(median_revenue, 2),

        # 🔒 RAW MEDIAN ₹ VALUES (NO CLAMP)
        "depreciation": round(median_dep, 2),
        "capex": round(median_capex, 2),
        "change_nwc": round(median_nwc, 2),

        # 🔒 CLAMPED % VALUES (FOR PROJECTIONS ONLY)
        "depreciation_pct": round(dep_pct * 100, 2),
        "capex_pct": round(capex_pct * 100, 2),
        "nwc_pct": round(nwc_pct * 100, 2),
    },


    "operating_margin": base_margin,
    "key_margin": key_margin,
    "is_financial": is_financial,

    "sensitivity_base_price": round(BASE_PRICE, 2),
    "sensitivity_margin_type": "Net Margin" if is_financial else "Operating Margin",

    "five_years": five_years,

    "debug": {
        "fcff_value": round(fcff_value, 2),
        "fcfe_value": round(fcfe_value, 2),
        "book_value": round(book_value, 2)
    },

    "sensitivity": {
        "growth": {"labels": growth_labels, "values": growth_values},
        "margin": {"labels": margin_labels, "values": margin_values},
        "wacc": {"labels": wacc_labels, "values": wacc_values}
    },

    "fcff_model": {
        "table": fcff_table,
        "terminal_value": round(terminal_value_fcff, 2),
        "pv_terminal_value": round(pv_terminal_fcff, 2),
        "sum_pv_fcFF": round(pv_sum_fcff, 2),
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "equity_value_per_share": fcff_eps_display
    },

    "fcfe_model": {
        "table": fcfe_table,
        "terminal_value": round(terminal_value_fcfe, 2),
        "pv_terminal_value": round(pv_terminal_fcfe, 2),
        "sum_pv_fcFE": round(pv_sum_fcfe, 2),
        "equity_value": round(equity_value_fcfe, 2),
        "equity_value_per_share": fcfe_eps_display
    }
}


    # Add dynamic key for margin
    result[key_margin_label] = round(key_margin * 100, 2)

    result.update({
    "capital_structure": capital_structure,
    "cost_of_capital": {
        "cost_of_debt": 0.08,
        "cost_of_equity": 0.12,
        "wacc": wacc
    }
})


    return result