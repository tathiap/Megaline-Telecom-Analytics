#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mysql.connector
import pandas as pd


# In[2]:


# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="September97!",  # Your MySQL password
    database="megaline_db"
)

# Define the queries for each view
queries = {
    "features": "SELECT * FROM megaline_features;",
    "ab_metrics": "SELECT * FROM ab_testing_metrics;",
    "plan_comparison": "SELECT * FROM plan_comparison;",
    "avg_usage": "SELECT * FROM average_usage;",
    "total_usage": "SELECT * FROM total_usage_summary;"
}

# Fetch each view into a dictionary of DataFrames
dataframes = {}
for key, query in queries.items():
    dataframes[key] = pd.read_sql(query, conn)
    print(f"{key} DataFrame loaded with {len(dataframes[key])} rows.")

# Access each DataFrame like this:
features_df = dataframes["features"]
ab_metrics_df = dataframes["ab_metrics"]

# Close connection
conn.close()

# Display features DataFrame
features_df.head()


# In[ ]:





# In[ ]:




