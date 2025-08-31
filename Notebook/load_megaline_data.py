#!/usr/bin/env python
# coding: utf-8

# In[22]:

import pandas as pd
import mysql.connector
from pathlib import Path
from mysql.connector import Error


# figure out the repo root no matter where the script runs from
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]     # go up from /Notebook/ to repo root
DATA_DIR = PROJECT_ROOT / "dataset"       


csv_path = DATA_DIR / "users_behavior.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"Couldn't find: {csv_path}")

df = pd.read_csv(csv_path)
df.reset_index(inplace=True)
df.rename(columns={'index': 'user_id'}, inplace=True)

# In[21]:


# Connect to MySQL
try:
    # Connect to MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="September97!",  
        database="megaline_db"
    )
    cursor = conn.cursor()

    # Insert data
    insert_query = """
        INSERT INTO users_behavior (user_id, calls, minutes, messages, mb_used, is_ultra)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            int(row['user_id']),
            int(row['calls']),
            float(row['minutes']),
            int(row['messages']),
            float(row['mb_used']),
            int(row['is_ultra'])
        ))

    conn.commit()
    print("Data successfully uploaded to MySQL!")

except Error as e:
    print(f"Error: {e}")

finally:
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print("MySQL connection closed.")


# In[24]:


# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="September97!",
    database="megaline_db"
)

# Create a cursor
cursor = conn.cursor()

# Verify the data
query = "SELECT * FROM users_behavior LIMIT 10;"
cursor.execute(query)
rows = cursor.fetchall()
for row in rows:
    print(row)


# In[25]:


query = "SELECT * FROM users_behavior LIMIT 10;"
df_sql = pd.read_sql(query, conn)
df_sql.head()

