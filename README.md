# Vendor Performance Analytics System
The Vendor Performance Analytics System is a data-driven project that analyzes vendor-related data to evaluate performance and predict future sales. It integrates multiple datasets, applies machine learning models, and uses time-series forecasting to generate actionable insights.

# Features
Data ingestion from CSV files
Data cleaning and preprocessing
Vendor performance analysis
Feature engineering (profit, margin, turnover)
Machine learning prediction (Linear Regression, Random Forest, XGBoost)
Time-series forecasting using Prophet
Visualization using charts and dashboard
SQLite database storage

# Technologies Used
Programming Language: Python
Database: SQLite
Libraries:
Pandas, NumPy
Scikit-learn
XGBoost
Prophet
Matplotlib
SQLAlchemy

# Project Structure
project/
│
├── data/                
├── models/              
├── forecasts/           
├── logs/                
├── database/            
│
├── data_ingestion.py
├── vendor_summary.py
├── predict_vendor_sales.py
├── forecast_vendor_sales.py
├── dashboard.py
│
└── README.md

# Installation and Setup
1. Install Dependencies
pip install pandas numpy scikit-learn xgboost prophet matplotlib sqlalchemy
How to Run
Step 1: Data Ingestion
python data_ingestion.py
Step 2: Generate Vendor Summary
python vendor_summary.py
Step 3: Train Machine Learning Models
python predict_vendor_sales.py
Step 4: Run Forecasting
python forecast_vendor_sales.py
Step 5: View Dashboard
python dashboard.py

# Outputs
Vendor performance summary tables
Machine learning model evaluation (MAE, RMSE, R²)
Forecast graphs and future predictions
Dashboard visualizations

# Use Cases
Vendor performance evaluation
Business decision-making
Sales prediction and planning
Inventory and purchase optimization

# Future Scope
Real-time data integration
Web-based application (Flask/Django)
Cloud deployment
Advanced dashboards (Power BI/Tableau)
Enhanced machine learning models

Author
Nikhil Bhushan Singh
