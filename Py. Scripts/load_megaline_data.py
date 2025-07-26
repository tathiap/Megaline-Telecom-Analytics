#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install mysql-connector-python')


# In[22]:


import pandas as pd
import mysql.connector
from mysql.connector import Error


# In[15]:


# Load dataset
df = pd.read_csv("Datasets/users_behavior.csv")
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

