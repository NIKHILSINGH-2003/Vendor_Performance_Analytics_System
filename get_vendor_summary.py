import sqlite3
import pandas as pd
import logging
import time
import os
from sqlalchemy import create_engine
from ingestion_db import ingestion_db

logging.basicConfig(
    filename="logs/get_vendor_summary.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"
)

engine = create_engine('sqlite:///inventory.db')
def ingest_db(df, table_name, engine):
    '''this function will ingest the dataframe into the database
     '''
    df.to_sql(table_name, con = engine, if_exists  ='replace', index = False)

def load_raw_data():
    '''this function will load the CSVs as dataframes and ingest into db'''
    start = time.time()
    for file in os.listdir('data'):
        if '.csv' in file:
                df = pd.read_csv('data/'+file)
                logging.info(f'Ingesting {file} in db')
                print (df.shape)
                ingest_db(df, file[:-4], engine)
    end = time.time()
    total_time = (end - start)/60
    logging.info('---------------Ingestion Complete----------------')

    logging.info(f'\nTotal Time Taken: {total_time} minutes')

if __name__ == '__main__':
    load_raw_data()

def create_vendor_summary(conn):
    start = time.time()
    '''this function will merge the different tables to get overall vendor summary and adding new columns in the resultant data'''
    vendor_sales_summary = pd.read_sql_query("""WITH FreightSummary AS (
    SELECT
        VendorNumber,
        SUM(Freight) AS FreightCost
    FROM vendor_invoice
    GROUP BY VendorNumber
),
PurchaseSummary AS (
    SELECT
        p.VendorNumber,
        p.VendorName,
        p.Brand,
        p.Description,
        p.PurchasePrice,
        pp.Price AS ActualPrice,
        pp.Volume,
        SUM(p.Quantity) AS TotalPurchaseQuantity,
        SUM(p.Dollars) AS TotalPurchaseDollars
    FROM purchases p
    JOIN purchase_prices pp
        ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY 
        p.VendorNumber, 
        p.VendorName, 
        p.Brand, 
        p.Description, 
        p.PurchasePrice, 
        pp.Price, 
        pp.Volume
),

SalesSummary AS (
    SELECT
        VendorNo,
        Brand,
        SUM(SalesQuantity) AS TotalSalesQuantity,
        SUM(SalesDollars) AS TotalSalesDollars,
        SUM(SalesPrice) AS TotalSalesPrice,
        SUM(ExciseTax) AS TotalExciseTax
    FROM sales
    GROUP BY VendorNo, Brand
)
SELECT
    ps.VendorNumber,
    ps.VendorName,
    ps.Brand,
    ps.Description,
    ps.PurchasePrice,
    ps.ActualPrice,
    ps.Volume,
    ps.TotalPurchaseQuantity,
    ps.TotalPurchaseDollars,
    ss.TotalSalesQuantity,
    ss.TotalSalesDollars,
    ss.TotalSalesPrice,
    ss.TotalExciseTax,
    fs.FreightCost
FROM PurchaseSummary ps
LEFT JOIN SalesSummary ss
    ON ps.VendorNumber = ss.VendorNo
    AND ps.Brand = ss.Brand
LEFT JOIN FreightSummary fs
    ON ps.VendorNumber = fs.VendorNumber
ORDER BY ps.TotalPurchaseDollars DESC
""", conn)
    end = time.time()
    logging.info(f"SQL Execution Time: {(end - start)/60:.2f} minutes")
    return vendor_sales_summary

def clean_data(vendor_sales_summary):
    '''this function will clean the data'''
    start = time.time()
    # changing datatype to float
    vendor_sales_summary['Volume'] = vendor_sales_summary['Volume'].astype('float')

    # filling missing value with 0
    vendor_sales_summary.fillna(0, inplace=True)

    # removing spaces from categorical columns
    vendor_sales_summary['VendorName'] = vendor_sales_summary['VendorName'].str.strip()
    vendor_sales_summary['Description'] = vendor_sales_summary['Description'].str.strip()

    # creating new columns for better analysis
    vendor_sales_summary['GrossProfit'] = vendor_sales_summary['TotalSalesDollars'] - vendor_sales_summary['TotalPurchaseDollars']
    vendor_sales_summary['ProfitMargin'] = (vendor_sales_summary['GrossProfit'] / vendor_sales_summary['TotalSalesDollars']) * 100
    vendor_sales_summary['StockTurnover'] = vendor_sales_summary['TotalSalesQuantity'] / vendor_sales_summary['TotalPurchaseQuantity']
    vendor_sales_summary['SalesToPurchaseRatio'] = vendor_sales_summary['TotalSalesDollars'] / vendor_sales_summary['TotalPurchaseDollars']
    
    end = time.time()
    logging.info(f"Data Cleaning Time: {(end - start):.2f} seconds")

    return vendor_sales_summary

if __name__ == '__main__':
    total_start = time.time()
    # creating database connection
    conn = sqlite3.connect('inventory.db')

    logging.info('Creating Vendor Summary Table.....')
    summary_df = create_vendor_summary(conn)
    logging.info(summary_df.head())

    logging.info('Cleaning Data......')
    clean_df = clean_data(summary_df)
    logging.info(clean_df.head())

    logging.info('Ingesting data....')
    ingest_db(clean_df, 'vendor_sales_summary', conn)
    logging.info('Completed')
    total_end = time.time()
    logging.info(f"TOTAL PIPELINE TIME: {(total_end - total_start)/60:.2f} minutes")