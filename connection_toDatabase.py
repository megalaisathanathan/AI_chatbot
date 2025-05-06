import pyodbc
import pandas as pd

# Azure SQL connection details
server = 'hrleaveapplication.database.windows.net'
database = 'hrleavemanagementDB'
username = 'mega'
password = 'password@123'
driver = '{ODBC Driver 17 for SQL Server}'  # Ensure this driver is installed

# Create connection string
connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# SQL query
query = "SELECT * FROM Table_2"

# Connect and read data
try:
    with pyodbc.connect(connection_string) as conn:
        df = pd.read_sql(query, conn)
        print("Data read successfully:")
        print(df.head())  # Show first 5 rows
except Exception as e:
    print("Error reading data from Azure SQL:", e)
