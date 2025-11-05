#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Realistic Azure Data Engineering Capstone Datasets (CSV + JSON)
Project: "Unlocking Insights: A Comprehensive Retail Sales Analysis"

Creates 5 datasets with 10,000 rows each:
  - data/customers.csv
  - data/transactions.csv
  - data/products.csv
  - data/regions.json
  - data/support_tickets.json

Each dataset includes:
  - ≥12 columns
  - Intentional data quality issues: duplicates, missing values, outliers,
    data type mismatches, and formatting inconsistencies.
  - Shared keys to enable joins: CustomerID, ProductID, RegionID

Requirements:
  - Python 3.9+
  - pandas, numpy
"""

import os
import random
import string
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# -----------------------------
# Global config / constants
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_ROWS = 10_000  # exact row count per dataset (duplicates are created in-place, not by adding rows)
DATA_DIR = "data"

# percentages for issues
PCT_MISSING = 0.03            # 3% cells per selected columns will be set missing
PCT_DUPLICATE_ROWS = 0.05     # 5% rows will be duplicated (in-place copy to other indices)
PCT_TYPE_MISMATCH = 0.01      # 1% numeric cells replaced with strings
PCT_OUTLIERS = 0.01           # 1% values made extreme
PCT_INCONSISTENT_CASE = 0.10  # 10% strings cased inconsistently

# date ranges
TODAY = datetime(2025, 1, 1)
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 11, 1)

# helper pools
FIRST_NAMES = ["Aarav", "Vivaan", "Riya", "Isha", "Kabir", "Ananya", "Rahul", "Sneha", "Arjun", "Priya",
               "David", "Emma", "Olivia", "Liam", "Noah", "Sophia", "Mia", "Ava", "Lucas", "Amelia"]
LAST_NAMES = ["Sharma", "Reddy", "Patel", "Khan", "Gupta", "Singh", "Iyer", "Kapoor", "Das", "Nair",
              "Smith", "Brown", "Johnson", "Williams", "Jones", "Miller", "Davis", "Garcia", "Wilson", "Moore"]
CITIES = ["Hyderabad", "Bengaluru", "Pune", "Mumbai", "Delhi", "Chennai", "Kolkata",
          "San Francisco", "New York", "London", "Sydney", "Toronto", "Dublin", "Berlin", "Paris"]
STATES = ["Telangana", "Karnataka", "Maharashtra", "Delhi NCR", "Tamil Nadu", "West Bengal",
          "California", "New York", "Ontario", "NSW", "Bayern", "Île-de-France"]
COUNTRIES = ["India", "USA", "Canada", "Australia", "UK", "Germany", "France"]
LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]
PAYMENT_TYPES = ["Credit Card", "Debit Card", "UPI", "Net Banking", "COD", "Wallet"]
CHANNELS = ["Web", "Mobile", "Marketplace", "In-Store"]
CURRENCIES = ["INR", "USD", "EUR", "GBP", "AUD", "CAD"]
BRANDS = ["GeoBasics", "CartPro", "PrimeLine", "UrbanEdge", "HomeSense", "ActiveX", "Luxora"]
CATEGORIES = ["Electronics", "Home & Kitchen", "Fashion", "Sports", "Beauty", "Toys", "Grocery"]
ISSUE_TYPES = ["Late Delivery", "Damaged Product", "Refund Request", "Payment Failed", "Wrong Item", "Account Issue"]
PRIORITIES = ["Low", "Medium", "High", "Urgent"]
STATUSES = ["Open", "In Progress", "Escalated", "Resolved", "Closed"]
AGENTS = ["Alex", "Bhavna", "Carlos", "Diana", "Ethan", "Fatima", "George", "Harini", "Ivan", "Jia"]

# -----------------------------
# Utility functions
# -----------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def random_phone():
    # generate a 10-digit phone with varied formatting
    base = "".join(random.choices(string.digits, k=10))
    formats = [
        f"+91-{base[:5]}-{base[5:]}",
        f"({base[:3]}) {base[3:6]}-{base[6:]}",
        f"{base[:3]}-{base[3:6]}-{base[6:]}",
        base
    ]
    return random.choice(formats)

def random_date(start=START_DATE, end=END_DATE):
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days),
                          seconds=random.randint(0, 86399))
    # introduce inconsistent date formats
    fmts = [
        d.strftime("%Y-%m-%d"),
        d.strftime("%d/%m/%Y"),
        d.strftime("%b %d, %Y"),
        d.strftime("%Y-%m-%d %H:%M:%S"),
        d.isoformat(timespec="seconds")
    ]
    return random.choice(fmts)

def random_dob(min_age=18, max_age=80):
    age = random.randint(min_age, max_age)
    dob = TODAY - timedelta(days=age * 365 + random.randint(0, 364))
    fmts = [
        dob.strftime("%Y-%m-%d"),
        dob.strftime("%d-%m-%Y"),
        dob.strftime("%d/%m/%y"),
        dob.strftime("%b %d, %Y")
    ]
    return random.choice(fmts), age

def random_sku():
    return f"SKU-{random.choice(string.ascii_uppercase)}{random.randint(10000, 99999)}"

def random_dims():
    return f"{round(random.uniform(5, 80),1)}x{round(random.uniform(5,80),1)}x{round(random.uniform(1,40),1)} cm"

def random_weight():
    return round(random.uniform(0.1, 25.0), 2)

def sometimes_inconsistent_case(s):
    if random.random() < PCT_INCONSISTENT_CASE:
        return random.choice([s.lower(), s.upper(), s.title(), s.capitalize()])
    return s

def inject_missing_values(df: pd.DataFrame, columns: list, fraction=PCT_MISSING):
    for col in columns:
        # Ensure column can hold mixed data
        df[col] = df[col].astype(object)
        mask = np.random.rand(len(df)) < fraction

        # Randomly assign missing values (either np.nan or empty string)
        for idx in df[mask].index:
            df.at[idx, col] = np.nan if np.random.rand() < 0.5 else ""
    return df


def inject_type_mismatches(df: pd.DataFrame, numeric_columns: list, fraction=PCT_TYPE_MISMATCH):
    for col in numeric_columns:
        mask = np.random.rand(len(df)) < fraction
        if mask.any():
            # Ensure col can hold strings mixed into numeric columns
            df[col] = df[col].astype(object)
            repl_choices = ["N/A", "unknown", "one hundred", "—"]
            # Random strings per row (not one broadcasted value)
            df.loc[mask, col] = np.random.choice(repl_choices, size=int(mask.sum()))
    return df

def inject_outliers(df: pd.DataFrame, columns: list, fraction=PCT_OUTLIERS, scale=20):
    for col in columns:
        mask = np.random.rand(len(df)) < fraction
        if mask.any():
            # Convert to numeric, coerce errors to NaN, then fill with 1 to allow multiplication
            base_vals = pd.to_numeric(df.loc[mask, col], errors="coerce").fillna(1)
            df.loc[mask, col] = base_vals * scale * np.random.uniform(5, 50, size=mask.sum())
    return df

def create_inplace_duplicates(df: pd.DataFrame, fraction=PCT_DUPLICATE_ROWS):
    # Duplicate rows by copying randomly selected source rows over random target indices (keeps row count same)
    k = int(len(df) * fraction)
    if k <= 0:
        return df
    src_idx = np.random.choice(df.index, size=k, replace=True)
    dst_idx = np.random.choice(df.index, size=k, replace=False)
    df.loc[dst_idx] = df.loc[src_idx].values
    return df

# -----------------------------
# 1) customers.csv (10,000 rows, ≥12 cols)
# -----------------------------
def build_customers(n=N_ROWS):
    customer_ids = np.arange(1, n + 1)

    first_names = [sometimes_inconsistent_case(random.choice(FIRST_NAMES)) for _ in range(n)]
    last_names = [sometimes_inconsistent_case(random.choice(LAST_NAMES)) for _ in range(n)]
    emails = [f"{fn.split()[0].lower()}.{ln.split()[0].lower()}@example.com" for fn, ln in zip(first_names, last_names)]
    phones = [random_phone() for _ in range(n)]
    genders = [random.choice(["Male", "Female", "Other"]) for _ in range(n)]
    dobs, ages = zip(*[random_dob() for _ in range(n)])
    join_dates = [random_date() for _ in range(n)]
    region_ids = np.random.randint(1, n + 1, size=n)  # share RegionID space with regions.json (1..10000)
    cities = [random.choice(CITIES) for _ in range(n)]
    states = [random.choice(STATES) for _ in range(n)]
    countries = [random.choice(COUNTRIES) for _ in range(n)]
    loyalty = [random.choice(LOYALTY_TIERS) for _ in range(n)]
    is_active = np.random.choice([True, False], size=n, p=[0.85, 0.15])
    income = np.round(np.random.normal(9_00_000, 4_00_000, size=n), 0)  # INR-like incomes
    prefer_pay = [random.choice(PAYMENT_TYPES) for _ in range(n)]

    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "FirstName": first_names,
        "LastName": last_names,
        "Email": emails,
        "Phone": phones,
        "Gender": genders,
        "DOB": dobs,                # inconsistent formats
        "Age": ages,
        "JoinDate": join_dates,     # inconsistent formats
        "RegionID": region_ids,
        "City": cities,
        "State": states,
        "Country": countries,
        "LoyaltyTier": loyalty,
        "IsActive": is_active,
        "AnnualIncome": income,
        "PreferredPaymentMethod": [sometimes_inconsistent_case(x) for x in prefer_pay],
    })

    # Inject issues
    df = inject_missing_values(df, ["Phone", "Email", "PreferredPaymentMethod"])
    df = inject_type_mismatches(df, ["Age", "AnnualIncome"])
    df = inject_outliers(df, ["AnnualIncome"], scale=100)  # create extreme incomes
    df = create_inplace_duplicates(df)

    return df

# -----------------------------
# 2) products.csv (10,000 rows, ≥12 cols)
# -----------------------------
def build_products(n=N_ROWS):
    product_ids = np.arange(1, n + 1)
    skus = [random_sku() for _ in range(n)]
    product_names = [f"{random.choice(['Ultra', 'Pro', 'Lite', 'Max', 'Eco', 'Smart'])} " +
                     f"{random.choice(['Phone', 'Mixer', 'Shoes', 'Watch', 'Headphones', 'Bottle', 'Backpack'])}"
                     for _ in range(n)]
    categories = [random.choice(CATEGORIES) for _ in range(n)]
    subcategories = [random.choice(["Premium", "Budget", "Standard", "Kids", "Outdoor", "Office"]) for _ in range(n)]
    brands = [random.choice(BRANDS) for _ in range(n)]
    unit_price = np.round(np.random.gamma(5, 300, size=n), 2)  # realistic skewed price distribution
    cost_price = np.round(unit_price * np.random.uniform(0.4, 0.85, size=n), 2)
    launch_dates = [random_date(START_DATE - timedelta(days=365*5), END_DATE) for _ in range(n)]
    discontinued = np.random.choice([True, False], size=n, p=[0.1, 0.9])
    colors = [random.choice(["Red", "Blue", "Black", "White", "Green", "Grey", "Pink"]) for _ in range(n)]
    sizes = [random.choice(["XS", "S", "M", "L", "XL", "XXL", ""]) for _ in range(n)]
    weights = [random_weight() for _ in range(n)]
    dims = [random_dims() for _ in range(n)]
    rating = np.round(np.random.uniform(2.0, 5.0, size=n), 2)
    stock_status = [random.choice(["In Stock", "Low Stock", "Out of Stock"]) for _ in range(n)]

    df = pd.DataFrame({
        "ProductID": product_ids,
        "SKU": skus,
        "ProductName": [sometimes_inconsistent_case(x) for x in product_names],
        "Category": categories,
        "SubCategory": subcategories,
        "Brand": brands,
        "UnitPrice": unit_price,
        "CostPrice": cost_price,
        "LaunchDate": launch_dates,   # inconsistent formats
        "DiscontinuedFlag": discontinued,
        "Color": colors,
        "Size": sizes,
        "WeightKg": weights,
        "PackageDimensions": dims,
        "Rating": rating,
        "StockStatus": stock_status,
    })

    # Inject issues
    df = inject_missing_values(df, ["Size", "Color"])
    df = inject_type_mismatches(df, ["UnitPrice", "CostPrice", "WeightKg", "Rating"])
    df = inject_outliers(df, ["UnitPrice", "WeightKg"], scale=50)  # extreme prices/weights
    df = create_inplace_duplicates(df)

    return df

# -----------------------------
# 3) transactions.csv (10,000 rows, ≥12 cols)
# -----------------------------
def build_transactions(n=N_ROWS):
    transaction_ids = np.arange(1, n + 1)
    customer_ids = np.random.randint(1, N_ROWS + 1, size=n)  # aligns with customers
    product_ids = np.random.randint(1, N_ROWS + 1, size=n)   # aligns with products
    region_ids = np.random.randint(1, N_ROWS + 1, size=n)    # aligns with regions
    dates = [random_date() for _ in range(n)]
    qty = np.random.randint(1, 6, size=n)
    unit_price = np.round(np.random.gamma(5, 300, size=n), 2)
    discount_pct = np.round(np.random.choice([0, 0, 0, 5, 10, 15, 20, 25], size=n), 2)
    tax_amt = np.round(unit_price * qty * 0.18, 2)
    total_amt = np.round(unit_price * qty * (1 - discount_pct/100) + tax_amt, 2)
    pay_type = [random.choice(PAYMENT_TYPES) for _ in range(n)]
    channel = [random.choice(CHANNELS) for _ in range(n)]
    status = [random.choice(["Completed", "Pending", "Failed", "Refunded"]) for _ in range(n)]
    promo = [random.choice(["", "NEWUSER", "FEST20", "SAVE10", "BOGO", "CASHBACK"]) for _ in range(n)]
    return_flag = np.random.choice(["Y", "N"], size=n, p=[0.08, 0.92])
    line_status = np.random.choice(["O", "F"], size=n, p=[0.6, 0.4])
    currency = [random.choice(CURRENCIES) for _ in range(n)]
    device = [random.choice(["iOS", "Android", "Desktop", "Tablet"]) for _ in range(n)]

    df = pd.DataFrame({
        "TransactionID": transaction_ids,
        "CustomerID": customer_ids,
        "ProductID": product_ids,
        "RegionID": region_ids,
        "TransactionDate": dates,  # inconsistent formats
        "Quantity": qty,
        "UnitPrice": unit_price,
        "DiscountPercent": discount_pct,
        "TaxAmount": tax_amt,
        "TotalAmount": total_amt,
        "PaymentType": [sometimes_inconsistent_case(x) for x in pay_type],
        "Channel": channel,
        "Status": status,
        "PromoCode": promo,
        "ReturnFlag": return_flag,
        "LineStatus": line_status,
        "Currency": currency,
        "DeviceType": device
    })

    # Inject issues
    df = inject_missing_values(df, ["PromoCode", "PaymentType"])
    df = inject_type_mismatches(df, ["Quantity", "UnitPrice", "DiscountPercent", "TaxAmount", "TotalAmount"])
    df = inject_outliers(df, ["TotalAmount"], scale=30)  # extreme totals
    df = create_inplace_duplicates(df)

    return df

# -----------------------------
# 4) regions.json (10,000 rows, ≥12 cols)
# -----------------------------
def build_regions(n=N_ROWS):
    region_ids = np.arange(1, n + 1)
    region_names = [f"Region-{i}" for i in region_ids]
    countries = [random.choice(COUNTRIES) for _ in range(n)]
    states = [random.choice(STATES) for _ in range(n)]
    cities = [random.choice(CITIES) for _ in range(n)]
    timezones = [random.choice(["Asia/Kolkata", "UTC", "US/Pacific", "Europe/London", "Europe/Berlin"]) for _ in range(n)]
    currency = [random.choice(CURRENCIES) for _ in range(n)]
    manager = [f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" for _ in range(n)]
    sales_target = np.round(np.random.uniform(50_000, 5_000_000, size=n), 2)
    created = [random_date(START_DATE - timedelta(days=365*3), END_DATE) for _ in range(n)]
    updated = [random_date(START_DATE, END_DATE) for _ in range(n)]
    lat = np.round(np.random.uniform(-60, 60, size=n), 5)
    lon = np.round(np.random.uniform(-150, 150, size=n), 5)
    warehouses = [[f"WH-{random.randint(1, 999)}" for _ in range(random.randint(1, 4))] for _ in range(n)]
    tax_rate = np.round(np.random.uniform(0.05, 0.25, size=n), 3)
    parent = np.random.choice(list(range(1, n + 1)) + [None], size=n, p=[*( [1/(n+10)]*n ), 10/(n+10)])

    df = pd.DataFrame({
        "RegionID": region_ids,
        "RegionName": [sometimes_inconsistent_case(x) for x in region_names],
        "Country": countries,
        "State": states,
        "City": cities,
        "Timezone": timezones,
        "Currency": currency,
        "ManagerName": manager,
        "SalesTarget": sales_target,
        "CreatedAt": created,
        "UpdatedAt": updated,
        "GeoCoordinates": list(zip(lat, lon)),
        "Warehouses": warehouses,
        "TaxRate": tax_rate,
        "ParentRegionID": parent
    })

    # Inject issues
    df = inject_missing_values(df, ["ParentRegionID", "ManagerName"])
    df = inject_type_mismatches(df, ["SalesTarget", "TaxRate"])
    df = inject_outliers(df, ["SalesTarget"], scale=40)
    df = create_inplace_duplicates(df)

    return df

# -----------------------------
# 5) support_tickets.json (10,000 rows, ≥12 cols)
# -----------------------------
def build_support_tickets(n=N_ROWS):
    ticket_ids = np.arange(1, n + 1)
    customer_ids = np.random.randint(1, N_ROWS + 1, size=n)   # aligns with customers
    region_ids = np.random.randint(1, N_ROWS + 1, size=n)     # aligns with regions
    created = [random_date(START_DATE, END_DATE) for _ in range(n)]
    # closed at after created (some nulls)
    closed = []
    for c in created:
        try:
            parsed = pd.to_datetime(c, errors="coerce", dayfirst=True)
        except Exception:
            parsed = None
        if random.random() < 0.15:
            closed.append("")  # missing closed time
        else:
            base = parsed if parsed is not None and not pd.isna(parsed) else datetime.utcnow()
            closed.append((base + timedelta(hours=random.randint(1, 240))).strftime(
                random.choice(["%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%b %d, %Y %I:%M %p"])
            ))

    status = [random.choice(STATUSES) for _ in range(n)]
    channel = [random.choice(["Email", "Chat", "Phone", "Portal"]) for _ in range(n)]
    priority = [random.choice(PRIORITIES) for _ in range(n)]
    issue = [random.choice(ISSUE_TYPES) for _ in range(n)]
    sentiment = np.round(np.random.uniform(-1.0, 1.0, size=n), 3)
    agent = [random.choice(AGENTS) for _ in range(n)]
    resolution_mins = np.random.randint(10, 7 * 24 * 60, size=n)
    sla_breached = np.random.choice([True, False], size=n, p=[0.25, 0.75])
    notes = [random.choice([
        "Customer requested callback.", "Refund processed.", "Replacement issued.",
        "Investigating with courier.", "Escalated to L2.", "NA"
    ]) for _ in range(n)]
    satisfaction = np.round(np.random.uniform(1.0, 5.0, size=n), 2)
    related_txn = np.random.choice(list(range(1, N_ROWS + 1)) + [None, None, None], size=n)  # some nulls
    tags = [[random.choice(["refund", "delivery", "priority", "vip", "payment", "replacement"])
            for _ in range(random.randint(1, 3))] for _ in range(n)]

    df = pd.DataFrame({
        "TicketID": ticket_ids,
        "CustomerID": customer_ids,
        "RegionID": region_ids,
        "CreatedAt": created,
        "ClosedAt": closed,
        "Status": status,
        "Channel": channel,
        "Priority": priority,
        "IssueType": [sometimes_inconsistent_case(x) for x in issue],
        "SentimentScore": sentiment,
        "AgentName": agent,
        "ResolutionTimeMins": resolution_mins,
        "SLA_Breached": sla_breached,
        "Notes": notes,
        "SatisfactionRating": satisfaction,
        "RelatedTransactionID": related_txn,
        "Tags": tags
    })

    # Inject issues
    df = inject_missing_values(df, ["Notes", "ClosedAt"])
    df = inject_type_mismatches(df, ["SentimentScore", "ResolutionTimeMins", "SatisfactionRating"])
    df = inject_outliers(df, ["ResolutionTimeMins"], scale=20)
    df = create_inplace_duplicates(df)

    return df

# -----------------------------
# Save helpers
# -----------------------------
def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8")

def save_json_records(df: pd.DataFrame, path: str):
    # Save as an array of records
    df.to_json(path, orient="records", indent=2, date_format="iso")

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(DATA_DIR)

    # Build datasets
    customers = build_customers()
    products = build_products()
    transactions = build_transactions()
    regions = build_regions()
    tickets = build_support_tickets()

    # Save
    customers_path = os.path.join(DATA_DIR, "customers.csv")
    products_path = os.path.join(DATA_DIR, "products.csv")
    transactions_path = os.path.join(DATA_DIR, "transactions.csv")
    regions_path = os.path.join(DATA_DIR, "regions.json")
    tickets_path = os.path.join(DATA_DIR, "support_tickets.json")

    save_csv(customers, customers_path)
    save_csv(products, products_path)
    save_csv(transactions, transactions_path)
    save_json_records(regions, regions_path)
    save_json_records(tickets, tickets_path)

    # Previews
    print("\n=== Sample Previews (first 5 rows) ===")
    with pd.option_context("display.max_columns", None):
        print("\ncustomers.csv")
        print(customers.head(5))
        print("\nproducts.csv")
        print(products.head(5))
        print("\ntransactions.csv")
        print(transactions.head(5))
        print("\nregions.json (first 2 records)")
        print(pd.DataFrame(regions.head(2)).to_dict(orient="records"))
        print("\nsupport_tickets.json (first 2 records)")
        print(pd.DataFrame(tickets.head(2)).to_dict(orient="records"))

    # Summary
    print("\n=== Dataset Creation Summary ===")
    print(f"Saved: {customers_path}  (rows={len(customers)}, cols={customers.shape[1]})")
    print(f"Saved: {products_path}   (rows={len(products)}, cols={products.shape[1]})")
    print(f"Saved: {transactions_path} (rows={len(transactions)}, cols={transactions.shape[1]})")
    print(f"Saved: {regions_path}    (rows={len(regions)}, cols={regions.shape[1]})")
    print(f"Saved: {tickets_path}    (rows={len(tickets)}, cols={tickets.shape[1]})")
    print("\nDone. Datasets with intentional data quality issues are ready in the 'data/' directory.")

if __name__ == "__main__":
    main()
