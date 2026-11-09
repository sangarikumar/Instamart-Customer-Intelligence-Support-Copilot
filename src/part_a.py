from pathlib import Path
import sqlite3, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; OUT=ROOT/'outputs'; SQL=ROOT/'sql'
CLEAN.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True); SQL.mkdir(exist_ok=True)
def run():
    c=pd.read_csv(RAW/'customers.csv',keep_default_na=False); p=pd.read_csv(RAW/'products.csv'); o=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    # Customers: normalise labels and repair impossible/missing ages with city median.
    c['city']=c.city.astype(str).str.strip().str.title()
    c['gender']=c.gender.astype(str).str.strip().str.lower().map({'male':'male','m':'male','female':'female','f':'female'}).fillna('unknown')
    c['age']=pd.to_numeric(c.age,errors='coerce'); c.loc[~c.age.between(10,100),'age']=np.nan
    c['age']=c['age'].fillna(c.groupby('city').age.transform('median')).round().astype(int)
    c['signup_date']=pd.to_datetime(c.signup_date)
    p['price_inr']=pd.to_numeric(p.price_inr)
    # Orders: exact duplicates first, then keep latest record per order id.
    o['order_ts']=pd.to_datetime(o.order_ts); o=o.drop_duplicates().sort_values('order_ts').drop_duplicates('order_id',keep='last')
    o['quantity']=pd.to_numeric(o.quantity,errors='coerce'); o['discount_pct']=pd.to_numeric(o.discount_pct,errors='coerce')
    o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce'); o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce')
    o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left')
    missing=o.order_amount.isna(); o.loc[missing,'order_amount']=(o.loc[missing,'price_inr']*o.loc[missing,'quantity']*(1-o.loc[missing,'discount_pct']/100)).round(2)
    mask=(o.status=='completed') & o.delivery_minutes.isna()
    med=o[o.status=='completed'].groupby('category').delivery_minutes.transform('median')
    # map category medians to all completed rows, avoiding index alignment problems
    catmed=o[o.status=='completed'].groupby('category').delivery_minutes.median()
    o.loc[mask,'delivery_minutes']=o.loc[mask,'category'].map(catmed)
    o=o.drop(columns=['price_inr','category'])
    for df,name in [(c,'customers'),(p,'products'),(o,'orders')]:
        df.to_csv(CLEAN/f'{name}.csv',index=False)
    db=ROOT/'data/instamart.db'
    with sqlite3.connect(db) as con:
        c.assign(signup_date=c.signup_date.dt.strftime('%Y-%m-%d')).to_sql('customers',con,index=False,if_exists='replace')
        p.to_sql('products',con,index=False,if_exists='replace')
        o.assign(order_ts=o.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')).to_sql('orders',con,index=False,if_exists='replace')
    queries='''-- Q1 WHERE then HAVING\nSELECT customer_id, SUM(order_amount) AS spend_over_500 FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;\n-- Q2 revenue by city/category\nSELECT c.city,p.category,SUM(o.order_amount) AS revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;\n-- Q3 products above average price\nSELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);\n-- Q4 customers with no completed order\nSELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;\n-- Q5 above-average customer spend\nWITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT * FROM spend WHERE total_spend>(SELECT avg_total FROM avg_spend);\n-- Q6 ranking window functions\nWITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) row_num FROM spend;\n-- Q7 running daily revenue\nWITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily ORDER BY day;\n'''
    (SQL/'queries.sql').write_text(queries)
    # Validation checkpoints from the brief.
    assert len(c)==1200 and len(p)==40 and len(o)==12309
    assert set(o.status)=={'completed','cancelled'} and o.order_amount.notna().all()
    assert o.loc[o.status=='completed','delivery_minutes'].notna().all()
    assert o.loc[o.status=='cancelled','delivery_minutes'].isna().all()
    assert round(o.loc[o.status=='completed','order_amount'].sum(),2)==2998668.80
    assert c.age.notna().all() and c.city.nunique()==6
    print('ALL CHECKPOINTS PASSED')
if __name__=='__main__': run()
