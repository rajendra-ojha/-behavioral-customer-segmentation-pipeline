# Behavioral Customer Segmentation

<div align="center">

**RFM + Behavioral Feature Engineering + Unsupervised Clustering**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit Learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)](https://seaborn.pydata.org/)
[![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-6C757D?style=flat-square)](https://joblib.readthedocs.io/)
[![Clustering](https://img.shields.io/badge/ML-Clustering-8A2BE2?style=flat-square)](#methodology)
[![RFM](https://img.shields.io/badge/Analytics-RFM-008080?style=flat-square)](#rfm-analysis)
[![PCA](https://img.shields.io/badge/Explainability-PCA-7952B3?style=flat-square)](#visualization)

</div>

## Overview

**Behavioral Customer Segmentation** is an unsupervised machine learning project that groups customers according to their purchasing behavior.

The project uses transaction-level retail data to build **RFM features** and additional behavioral features, preprocesses the customer-level data, compares multiple clustering algorithms, selects a configuration using clustering quality metrics, profiles the discovered segments, and saves the trained model for later use.

### What this project demonstrates

- Data cleaning and validation
- Exploratory Data Analysis
- RFM analysis
- Behavioral feature engineering
- Outlier handling and robust scaling
- K-Means clustering
- Agglomerative clustering
- Gaussian Mixture Models
- Silhouette, Calinski-Harabasz and Davies-Bouldin evaluation
- PCA-based cluster visualization
- Dynamic customer segment naming
- Model persistence with Joblib
- New-customer segment prediction when the selected model supports `.predict()`

---

## Project Architecture

```text
Customer Transactions
        |
        v
Data Validation & Cleaning
        |
        v
Exploratory Data Analysis
        |
        v
RFM + Behavioral Features
        |
        v
Outlier Handling + Log Transformation
        |
        v
RobustScaler
        |
        v
+-------------------------------+
| Clustering Model Comparison   |
|                               |
|  K-Means                      |
|  Agglomerative Clustering     |
|  Gaussian Mixture Model       |
+-------------------------------+
        |
        v
Best Configuration
        |
        v
PCA Visualization
        |
        v
Customer Segment Profiling
        |
        v
Saved Model + Metadata
        |
        v
New Customer Segment Prediction
```

## Dataset

This project uses the **Online Retail II** dataset from the UCI Machine Learning Repository.

The dataset contains real transaction records from a UK-based registered non-store online retailer covering **01/12/2009 to 09/12/2011**. UCI reports **1,067,371 instances** and 8 transaction variables.

### Main columns

| Column | Description |
|---|---|
| `Invoice` | Invoice/transaction identifier |
| `StockCode` | Product identifier |
| `Description` | Product description |
| `Quantity` | Quantity purchased |
| `InvoiceDate` | Transaction date and time |
| `Price` | Unit price |
| `Customer ID` | Customer identifier |
| `Country` | Customer country |

The dataset is licensed by UCI under **CC BY 4.0**. citeturn0search0

### Download

**Official UCI dataset:**  
https://archive.ics.uci.edu/dataset/502/online+retail+ii

A Kaggle source is also referenced in the original notebook:

https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

After downloading, place the CSV/XLSX file inside `data/` and update `RAW_DATA_PATH`.

---

## RFM Analysis

The project calculates:

- **Recency** — days since the customer's most recent purchase
- **Frequency** — number of distinct orders
- **Monetary** — total revenue generated

Additional behavioral features include:

- Average Order Value
- Total Quantity
- Unique Products
- Average Items per Order
- Customer Lifetime Days
- Average Days Between Purchases
- Purchase Frequency Rate
- Product Diversity Ratio
- Primary Country

---

## Methodology

### 1. Data Cleaning

The pipeline removes or handles:

- Invalid transaction dates
- Missing customer IDs
- Missing descriptions
- Duplicate rows
- Cancelled invoices
- Non-positive quantities
- Non-positive prices

### 2. Feature Engineering

Customer-level features are generated from transaction-level records.

### 3. Outlier Handling

The notebook winsorizes:

- `Monetary`
- `Frequency`
- `TotalQuantity`

using the configured upper quantile.

### 4. Transformation and Scaling

Selected skewed numerical features are transformed with `log1p`, followed by `RobustScaler`.

### 5. Clustering

Three unsupervised approaches are compared:

- **K-Means**
- **Agglomerative Clustering**
- **Gaussian Mixture Model**

The notebook searches across multiple cluster counts and model configurations.

### 6. Model Evaluation

The clustering configurations are compared using:

- **Silhouette Score** — higher is generally better
- **Calinski-Harabasz Score** — higher is generally better
- **Davies-Bouldin Score** — lower is generally better

The implementation sorts candidates by silhouette score and selects the top valid configuration.

### 7. PCA Visualization

PCA reduces the clustering feature space to two components for visual inspection of the discovered customer groups.

### 8. Segment Profiling

Each cluster is profiled using customer count, revenue, recency, frequency, monetary value and average order value.

The notebook dynamically assigns descriptive segment names such as:

- High-Value Frequent Customers
- High-Spending Occasional Customers
- Frequent Low-Spend Customers
- Recently Active Growing Customers
- At-Risk / Low-Engagement Customers
- Dormant / Churned Customers
- Moderate / Average Engagement Customers

The exact segments depend on the data and selected clustering configuration.

---

## Output Files

Running the project creates:

```text
models/
├── preprocessing.joblib
├── clustering_model.joblib
└── metadata.joblib

reports/
├── data_quality_report.csv
├── customer_features.csv
├── model_comparison.csv
├── cluster_profiles.csv
└── figures/
    ├── 01_revenue_and_customers_over_time.png
    ├── 02_top_products_and_countries.png
    ├── 03_monetary_distribution_before_after.png
    ├── 04_kmeans_elbow_silhouette.png
    ├── 05_pca_cluster_projection.png
    ├── 06_segment_customers_and_revenue.png
    └── 07_cluster_profile_heatmap.png
```

---

## Project Structure

```text
behavioral-customer-segmentation/
│
├── data/
│   └── README.md
│
├── models/
│   └── generated model files
│
├── reports/
│   ├── figures/
│   └── generated CSV reports
│
├── src/
│   └── behavioral_customer_segmentation.py
│
├── behavioral_customer_segmentation.ipynb
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <your-repository-url>
cd behavioral-customer-segmentation

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Jupyter Notebook

```bash
jupyter notebook behavioral_customer_segmentation.ipynb
```

Before running, update:

```python
RAW_DATA_PATH = "path/to/online_retail_II.csv"
```

### Python Script

```bash
python src/behavioral_customer_segmentation.py
```

---

## Example New-Customer Prediction

When the selected clustering model supports `.predict()`, the saved model can be used for a new customer:

```python
predict_segment(
    recency=15,
    frequency=8,
    monetary=950.0,
    avg_order_value=118.75,
    total_quantity=64,
    unique_products=18,
    customer_lifetime_days=220,
)
```

The function returns the predicted cluster and its associated business profile.

---

## Business Use Cases

Customer segments generated by this project can support:

- Customer retention campaigns
- High-value customer targeting
- Re-engagement campaigns
- Personalized promotions
- Customer lifecycle analysis
- Marketing budget allocation
- Revenue contribution analysis
- Churn-risk investigation

---

## Key Technical Highlights

```text
Data Engineering
      ↓
RFM Analytics
      ↓
Behavioral Feature Engineering
      ↓
Unsupervised ML
      ↓
Model Evaluation
      ↓
Customer Profiling
      ↓
Model Persistence
      ↓
Inference
```

This makes the project more than a basic K-Means notebook: it includes data quality checks, feature engineering, algorithm comparison, evaluation, reporting, persistence and an inference path.

---

## Limitations

- This is an unsupervised segmentation project, so clusters do not have predefined ground-truth labels.
- Segment names are generated from relative customer behavior and should be interpreted as business-oriented descriptions.
- PCA is used for visualization only; clustering is performed on the full transformed feature space.
- New-customer prediction is available only when the selected final estimator exposes a compatible `.predict()` method.

---

## Dataset Citation

Chen, D. (2012). **Online Retail II**. UCI Machine Learning Repository.  
DOI: `10.24432/C5CG6D`

Official dataset page:  
https://archive.ics.uci.edu/dataset/502/online+retail+ii

---

## Author

**Your Name**

If you use this project in your portfolio, replace the placeholder above with your name and add your GitHub/LinkedIn links.

