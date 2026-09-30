"""
prepare_data.py
High-performance streaming data ingestion and preprocessing pipeline for Cafeteria Order Data.
Extracts reference tables, aggregates daily branch orders, hourly metrics, dish analytics, 
and generates compact Parquet datasets for exploratory data analysis and time-series forecasting.
"""

import os
import io
import csv
import time
from collections import Counter, defaultdict
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = "data"
SQL_PATH = "Cafeteria Order Data/Cafeteria Order Data.sql"
USERS_SQL_PATH = "Cafeteria Order Data/users.sql"

os.makedirs(DATA_DIR, exist_ok=True)

def normalize_payment(mode):
    if not mode:
        return "Unknown"
    m = mode.strip().lower()
    if "paytm" in m:
        return "Paytm"
    if "upi" in m:
        return "UPI"
    if "razor" in m:
        return "Razorpay"
    if "cca" in m:
        return "CCAvenue"
    if "cash" in m:
        return "Cash"
    if "card" in m:
        return "Card"
    if "qr" in m:
        return "QR Code"
    return "Other"

def extract_reference_tables():
    print("[1/5] Extracting reference tables (branches, categories, counters, dishes)...")
    
    # 1. Branches
    branches_cols = [
        "id", "name", "branch_code", "branch_merchant_code", "branch_manager", 
        "invoice_prefix", "contact_number", "is_active", "company_has_region_id", 
        "company_id", "is_pos", "is_sok", "is_qrcode", "is_mobile_ordering", 
        "is_table_room", "country_id", "state_id", "city_id", "license_id", 
        "license_no", "gst_no", "tax_ids", "discount_ids", "is_for_app", 
        "dining_page_link", "created_at", "updated_at"
    ]
    
    def parse_table_range(start_byte, end_byte, cols):
        rows = []
        with open(SQL_PATH, "rb") as f:
            f.seek(start_byte)
            while f.tell() < end_byte:
                line = f.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", "ignore").strip()
                if not line_str.startswith("("):
                    continue
                content = line_str[1:]
                if content.endswith(");") or content.endswith("),"):
                    content = content[:-2]
                reader = csv.reader(io.StringIO(content), quotechar="\x27", skipinitialspace=True)
                for r in reader:
                    row_dict = {col: (r[idx] if idx < len(r) and r[idx] != "NULL" else None) for idx, col in enumerate(cols)}
                    rows.append(row_dict)
        return pd.DataFrame(rows)

    branches_df = parse_table_range(1439795, 1445000, branches_cols)
    branches_df.to_csv(os.path.join(DATA_DIR, "branches.csv"), index=False)
    print(f"  ✓ branches.csv saved: {len(branches_df)} rows")

    # 2. Categories
    categories_cols = ["id", "category_name", "images", "branch_id", "counter_id", "discount_ids", "category_type", "is_active", "created_at", "updated_at"]
    categories_df = parse_table_range(7564085, 7575000, categories_cols)
    categories_df.to_csv(os.path.join(DATA_DIR, "categories.csv"), index=False)
    print(f"  ✓ categories.csv saved: {len(categories_df)} rows")

    # 3. Counters
    counters_cols = ["id", "counter_name", "counter_number", "counter_address", "image", "license_no", "tax_no", "license_expiry_date", "branch_id", "area_id", "discount_ids", "template_ids", "vendor_id", "app_vendor_id", "priority", "todays_paragraph", "todays_favorites", "isDigitalDashboard", "created_at", "updated_at"]
    counters_df = parse_table_range(13304950, 13320000, counters_cols)
    counters_df.to_csv(os.path.join(DATA_DIR, "counters.csv"), index=False)
    print(f"  ✓ counters.csv saved: {len(counters_df)} rows")

    # 4. Dishes
    dishes_cols = ["id", "dish_name", "dish_price", "dish_code", "dish_images", "has_variant", "is_tax_inclusive", "is_discount", "category_id", "counter_id", "discount_ids", "chef_preparation", "preparation_time", "dish_hsn", "dish_type", "item_type", "stock_quantity", "stock_type", "edited_at", "edited_by", "is_available", "popular_special", "is_active", "is_edible", "created_at", "updated_at"]
    dishes_df = parse_table_range(13393516, 15820000, dishes_cols)
    dishes_df.to_csv(os.path.join(DATA_DIR, "dishes.csv"), index=False)
    print(f"  ✓ dishes.csv saved: {len(dishes_df)} rows")
    
    return branches_df, categories_df, counters_df, dishes_df

def extract_users():
    print("[2/5] Extracting users summary from users.sql...")
    users_cols = [
        "id", "name", "username", "email", "email_verified_at", "password", 
        "emp_id", "otp", "otp_expire_timestamp", "is_2mfa", "gender", "dob", 
        "address", "remember_token", "fcm_code", "status", "is_lock", 
        "user_type", "deviceId", "app_version", "login_attempt", "created_at"
    ]
    rows = []
    with open(USERS_SQL_PATH, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str.startswith("("):
                continue
            content = line_str[1:]
            if content.endswith(");") or content.endswith("),"):
                content = content[:-2]
            reader = csv.reader(io.StringIO(content), quotechar="\x27", skipinitialspace=True)
            for r in reader:
                if len(r) >= 20:
                    rows.append({
                        "id": r[0],
                        "gender": r[10] if r[10] != "NULL" else None,
                        "status": r[15] if r[15] != "NULL" else None,
                        "user_type": r[17] if r[17] != "NULL" else None,
                        "created_at": r[21] if len(r) > 21 and r[21] != "NULL" else None
                    })
    users_df = pd.DataFrame(rows)
    users_df.to_csv(os.path.join(DATA_DIR, "users_summary.csv"), index=False)
    print(f"  ✓ users_summary.csv saved: {len(users_df)} rows")
    return users_df

def stream_orders_and_aggregate():
    print("[3/5] Streaming and processing 5.96M orders from SQL dump...")
    start_time = time.time()
    
    # Trackers for aggregation
    daily_branch_metrics = defaultdict(lambda: {
        "total_orders": 0, "paid_orders": 0, "cancel_orders": 0, "pending_orders": 0,
        "total_revenue": 0.0, "cash_orders": 0, "digital_orders": 0,
        "mobile_orders": 0, "sok_orders": 0, "pos_orders": 0
    })
    
    hourly_traffic = defaultdict(lambda: {"order_count": 0, "paid_revenue": 0.0})
    user_order_counts = Counter()
    
    # Parquet schema for clean orders
    order_records = []
    total_orders_count = 0
    
    # We will write the full clean orders into parquet in chunks
    parquet_path = os.path.join(DATA_DIR, "orders_clean.parquet")
    writer = None
    
    schema = pa.schema([
        ("order_id", pa.int64()),
        ("user_id", pa.int32()),
        ("order_date", pa.string()),
        ("branch_id", pa.int16()),
        ("channel", pa.string()),
        ("payment_mode", pa.string()),
        ("sub_total", pa.float32()),
        ("tax_amount", pa.float32()),
        ("discount_amount", pa.float32()),
        ("grand_total", pa.float32()),
        ("order_status", pa.int8()),
        ("paid_status", pa.string())
    ])
    
    chunk_size = 500000
    
    with open(SQL_PATH, "rb") as f:
        f.seek(91544735)
        end = 9226282877
        for line in f:
            if f.tell() > end:
                break
            if not line.startswith(b"("):
                continue
            
            s = line.decode("utf-8", "ignore").strip()
            parts = s[1:-2].split(",")
            if len(parts) > 31:
                try:
                    oid = int(parts[0].strip(" \x27"))
                    uid_str = parts[2].strip(" \x27")
                    uid = int(uid_str) if uid_str != "NULL" else -1
                    status_str = parts[3].strip(" \x27")
                    status = int(status_str) if status_str != "NULL" else 0
                    odate = parts[6].strip(" \x27")
                    bid_str = parts[8].strip(" \x27")
                    bid = int(bid_str) if bid_str != "NULL" else -1
                    channel = parts[12].strip(" \x27")
                    subtotal = float(parts[13].strip(" \x27")) if parts[13].strip(" \x27") != "NULL" else 0.0
                    tax = float(parts[14].strip(" \x27")) if parts[14].strip(" \x27") != "NULL" else 0.0
                    discount = float(parts[18].strip(" \x27")) if parts[18].strip(" \x27") != "NULL" else 0.0
                    mode = normalize_payment(parts[19].strip(" \x27"))
                    grand = float(parts[28].strip(" \x27")) if parts[28].strip(" \x27") != "NULL" else 0.0
                    paid = parts[30].strip(" \x27")
                    
                    total_orders_count += 1
                    user_order_counts[uid] += 1
                    
                    # Daily aggregation key: (date_str, branch_id)
                    date_str = odate[:10] if len(odate) >= 10 else "Unknown"
                    m = daily_branch_metrics[(date_str, bid)]
                    m["total_orders"] += 1
                    if paid == "paid":
                        m["paid_orders"] += 1
                        m["total_revenue"] += grand
                    elif paid == "cancel":
                        m["cancel_orders"] += 1
                    elif paid == "pending":
                        m["pending_orders"] += 1
                    
                    if mode == "Cash":
                        m["cash_orders"] += 1
                    else:
                        m["digital_orders"] += 1
                        
                    if channel == "mobile app":
                        m["mobile_orders"] += 1
                    elif channel == "sok":
                        m["sok_orders"] += 1
                    elif channel == "pos":
                        m["pos_orders"] += 1
                        
                    # Hourly aggregation: (branch_id, hour)
                    if len(odate) >= 13:
                        try:
                            hour = int(odate[11:13])
                            ht = hourly_traffic[(bid, hour)]
                            ht["order_count"] += 1
                            if paid == "paid":
                                ht["paid_revenue"] += grand
                        except:
                            pass
                            
                    order_records.append({
                        "order_id": oid,
                        "user_id": uid,
                        "order_date": odate,
                        "branch_id": bid,
                        "channel": channel,
                        "payment_mode": mode,
                        "sub_total": subtotal,
                        "tax_amount": tax,
                        "discount_amount": discount,
                        "grand_total": grand,
                        "order_status": status,
                        "paid_status": paid
                    })
                    
                    # Flush chunk to Parquet
                    if len(order_records) >= chunk_size:
                        table = pa.Table.from_pylist(order_records, schema=schema)
                        if writer is None:
                            writer = pq.ParquetWriter(parquet_path, schema, compression="snappy")
                        writer.write_table(table)
                        order_records.clear()
                        print(f"  ...processed {total_orders_count:,} orders ({time.time()-start_time:.1f}s)")
                        
                except Exception as e:
                    pass

    # Flush remaining orders
    if order_records:
        table = pa.Table.from_pylist(order_records, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, schema, compression="snappy")
        writer.write_table(table)
        order_records.clear()
        
    if writer is not None:
        writer.close()
        
    print(f"  ✓ orders_clean.parquet written: {total_orders_count:,} orders ({os.path.getsize(parquet_path)/(1024*1024):.1f} MB)")
    
    # Save daily metrics
    daily_rows = []
    for (date_str, bid), m in daily_branch_metrics.items():
        if date_str != "Unknown":
            daily_rows.append({
                "date": date_str,
                "branch_id": bid,
                **m
            })
    daily_df = pd.DataFrame(daily_rows)
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
    daily_df = daily_df.sort_values(["branch_id", "date"]).reset_index(drop=True)
    daily_df.to_csv(os.path.join(DATA_DIR, "daily_branch_orders.csv"), index=False)
    print(f"  ✓ daily_branch_orders.csv saved: {len(daily_df)} daily rows across branches")
    
    # Save hourly traffic
    hourly_rows = []
    for (bid, hour), ht in hourly_traffic.items():
        hourly_rows.append({
            "branch_id": bid,
            "hour": hour,
            "order_count": ht["order_count"],
            "paid_revenue": ht["paid_revenue"]
        })
    hourly_df = pd.DataFrame(hourly_rows).sort_values(["branch_id", "hour"]).reset_index(drop=True)
    hourly_df.to_csv(os.path.join(DATA_DIR, "hourly_traffic.csv"), index=False)
    print(f"  ✓ hourly_traffic.csv saved: {len(hourly_df)} hourly bins")
    
    # Save user order frequency distribution
    freq_counter = Counter(user_order_counts.values())
    user_freq_df = pd.DataFrame([{"orders_placed": k, "user_count": v} for k, v in sorted(freq_counter.items())])
    user_freq_df.to_csv(os.path.join(DATA_DIR, "customer_order_frequencies.csv"), index=False)
    print(f"  ✓ customer_order_frequencies.csv saved")
    
    return daily_df, hourly_df

def stream_order_details():
    print("[4/5] Streaming and analyzing 7.42M order items from SQL dump...")
    start_time = time.time()
    
    dish_volume = Counter()
    dish_revenue = defaultdict(float)
    dish_meta = {}
    counter_volume = Counter()
    counter_revenue = defaultdict(float)
    total_items = 0
    
    with open(SQL_PATH, "rb") as f:
        f.seek(9226284480)
        end = 10987630441
        for line in f:
            if f.tell() > end:
                break
            if not line.startswith(b"("):
                continue
            total_items += 1
            s = line.decode("utf-8", "ignore").strip()
            
            # (id, order_id, dish_id, dish_variant_id, dish_variant_price, extras_id, extras_price, addons_id, addons_price, dish_name, order_quantity, order_status, cd_status, prepared_timestamp, delivered_timestamp, dish_price, ...)
            # Extract dish_name cleanly by finding quotes
            q1 = s.find("'")
            if q1 != -1:
                q2 = s.find("'", q1 + 1)
                if q2 != -1:
                    dish_name = s[q1+1:q2]
                    # prefix before quote
                    prefix_parts = s[1:q1].rstrip(", ").split(",")
                    # suffix after quote
                    suffix_parts = s[q2+1:].lstrip(", ").split(",")
                    
                    try:
                        dish_id = int(prefix_parts[2].strip())
                        qty = int(suffix_parts[0].strip())
                        price = float(suffix_parts[5].strip())
                        counter_id = int(suffix_parts[8].strip()) if suffix_parts[8].strip() != "NULL" else -1
                        
                        dish_volume[dish_name] += qty
                        dish_revenue[dish_name] += (qty * price)
                        dish_meta[dish_name] = {"dish_id": dish_id, "price": price, "counter_id": counter_id}
                        
                        counter_volume[counter_id] += qty
                        counter_revenue[counter_id] += (qty * price)
                    except:
                        pass
                        
            if total_items % 1000000 == 0:
                print(f"  ...processed {total_items:,} items ({time.time()-start_time:.1f}s)")
                
    # Save top dishes
    top_dishes = []
    for dish_name, vol in dish_volume.most_common(200):
        rev = dish_revenue[dish_name]
        meta = dish_meta.get(dish_name, {})
        top_dishes.append({
            "dish_name": dish_name,
            "dish_id": meta.get("dish_id"),
            "counter_id": meta.get("counter_id"),
            "unit_price": meta.get("price"),
            "quantity_sold": vol,
            "total_revenue": round(rev, 2)
        })
    top_dishes_df = pd.DataFrame(top_dishes)
    top_dishes_df.to_csv(os.path.join(DATA_DIR, "top_dishes.csv"), index=False)
    print(f"  ✓ top_dishes.csv saved: {len(top_dishes_df)} dishes")
    
    # Save counter performance
    counter_rows = []
    for cid, vol in counter_volume.items():
        counter_rows.append({
            "counter_id": cid,
            "items_sold": vol,
            "total_revenue": round(counter_revenue[cid], 2)
        })
    counter_df = pd.DataFrame(counter_rows).sort_values("items_sold", ascending=False).reset_index(drop=True)
    counter_df.to_csv(os.path.join(DATA_DIR, "counter_performance.csv"), index=False)
    print(f"  ✓ counter_performance.csv saved: {len(counter_df)} counters")
    print(f"  Total items analyzed: {total_items:,} in {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("STARTING DATA INGESTION & PIPELINE ENGINE")
    print("=" * 60)
    extract_reference_tables()
    extract_users()
    stream_orders_and_aggregate()
    stream_order_details()
    print("=" * 60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {time.time()-t0:.1f}s!")
    print("=" * 60)
