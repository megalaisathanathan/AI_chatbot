# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.



import urllib.parse
import urllib.request
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext, CardFactory, MemoryStorage, UserState
from botbuilder.schema import (
    ChannelAccount,
    HeroCard,
    CardAction,
    ActivityTypes,
    Attachment,
    AttachmentData,
    Activity,
    ActionTypes,
)
import requests
import pandas as pd
import logging
import sys
import re
import time
import math
from word2number import w2n
import numpy as np
import bs4
import os
import base64
import json
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest
from azure.ai.formrecognizer import DocumentAnalysisClient  # Updated class name
import uuid
import io
from typing import List  # Add this import

    

class MyBot(ActivityHandler):
    def __init__(self, user_state: UserState):
        super().__init__()
        self.user_state = user_state
        self.user_data_accessor = self.user_state.create_property("user_status")
        self.question_index_accessor = self.user_state.create_property("question_index")
        self.answers_accessor = self.user_state.create_property("answers")
        self.val_list_accessor = self.user_state.create_property("val_list")
        self.processed_messages = set()

        self.API_KEY = "patHbVk2KOxbMFsjf.d9d53905e9aead5ffdb50b411eaccac276bfe3b23fd83628f0660a1700911d5e"
        self.BASE_ID = "appFA2iUehCVrUSdl"
        self.HEADERS = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }
        self.TABLE_NAME_READ_1 = "Table 3"
        self.TABLE_NAME_READ_2 = "Table 4"
        self.BATCH_SIZE = 10
        self.URL1 = f"https://api.airtable.com/v0/{self.BASE_ID}/{self.TABLE_NAME_READ_1}"
        self.URL2 = f"https://api.airtable.com/v0/{self.BASE_ID}/{self.TABLE_NAME_READ_2}"
        
    def fetch_all_airtable_data(self,url, headers):
        records = []
        offset = None
        while True:
            params = {}
            if offset:
                params["offset"] = offset
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break
            data = response.json()
            records.extend(data.get("records", []))
            offset = data.get("offset")
            if not offset:
                break
        return records
    
    def fetch_airtable_data(self):
        # Fetch all records from both tables
        records1 = self.fetch_all_airtable_data(self.URL1, self.HEADERS)
        records2 = self.fetch_all_airtable_data(self.URL2, self.HEADERS)
        # Convert records to DataFrames
        df1 = pd.DataFrame([record.get("fields", {}) for record in records1])
        df2 = pd.DataFrame([record.get("fields", {}) for record in records2])
        debug_info = {
            "Table 3 Records": len(df1),
            "Table 4 Records": len(df2)
        }
        print(debug_info)  # Useful for debugging

        return df1, df2, records2
    
    def extract_field(self, field_name, text):
            pattern = f"'{field_name}': DocumentField\\(.*?value='(.*?)'"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1) if match else None

    def delete_records(self, records, url, headers):
        for record in records:
            record_id = record['id']
            delete_url = f"{url}/{record_id}"
            response = requests.delete(delete_url, headers=headers)

    def write_to_airtable(self, df):
        """
        Writes 'confidences_df' to 'Table 4'.
        """
        try:
            TABLE_NAME_READ_3 = "Table 5"
            url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME_READ_3}"
            records = []

            for _, row in df.iterrows():
                record = {
                    "fields": {
                        "Total": row["Total"],
                        "Quantity": row["Quantity"],
                        "Description": row["Description"],
                        "Unit Price": row["Unit Price"],
                        "Amount": row["Amount"]
                    }
                }
                records.append(record)

            # Airtable API limits batch writes to 10 records at a time
            for i in range(0, len(records), 10):
                batch = {"records": records[i:i + 10]}
                response = requests.post(url, headers=self.HEADERS, json=batch)

                if response.status_code == 200:
                    print(f"Inserted {len(batch['records'])} records into Table 4.")
                else:
                    print(f"Error inserting records: {response.text}")

        except Exception as e:
            print(f"Exception occurred while writing to Table 4: {e}")

    def check_thresholds(self, confidences_df, df2):
        # Initialize validation list to store positions of low confidence values
        validation_list = []

        # Convert all numeric columns to float (avoiding type errors)
        numeric_cols = ['Quantity', 'Unit Price', 'Amount', 'Total']
        confidences_df[numeric_cols] = confidences_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Iterate through rows of the DataFrame
        for index, row in confidences_df.iterrows():
            # Check if 'Total' is empty but other fields have values
            if pd.isna(row.get('Total')) and not pd.isna(row.get('Quantity')) and not pd.isna(row.get('Unit Price')) and not pd.isna(row.get('Amount')):
                # Check confidence values and record positions if below threshold
                for col in ['Quantity', 'Unit Price', 'Amount']:
                    if row.get(col, 1.0) < 0.99:  
                        df2_value = df2.iloc[index][col]  # Get the actual extracted value from df2
                        validation_list.append([index, col, df2_value])

        return validation_list

    def checkConfidentialThreshold(self, data, df2):
        table = df2.copy()
        table = table.to_numpy()
        data['Page Number'] = data['Page Number'].astype(int)
        data['Confidence'] = data['Confidence'].astype(float)
        data['Word ID'] = data['Word ID'].astype(int)
        data = data.sort_values(by="Word ID", ascending=True)
        print(data)
        print(data.info())
        table_with_confidences = []

        word_confidence_mapping = data[['Word Content', 'Confidence']].dropna().set_index('Word Content').to_dict()['Confidence']
        for row in table:
            row_confidences = []
            for cell in row:
                if cell == "NaN" or cell == "None":  # Handle empty or missing cells
                    row_confidences.append(None)
                else:
                    # Split cell content into words to match with CSV data
                    words = str(cell).split()
                    confidences = [word_confidence_mapping.get(word, None) for word in words]
                    # Remove None values for words not in CSV
                    valid_confidences = [conf for conf in confidences if conf is not None]
                    # Calculate average confidence for the cell
                    avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else None
                    row_confidences.append(avg_confidence)
            table_with_confidences.append(row_confidences)

        confidences_df = pd.DataFrame(table_with_confidences, columns=["Total", "Quantity", "Description", "Unit Price", "Amount"])
        return confidences_df

    def has_digits(self, value):
        if pd.isna(value):
            return False
        return any(char.isdigit() for char in str(value))

    def post_processing(self, df, data):
        row_index = 0
        for index, row in df.iterrows():
            if row['Field Name'] == 'invoiceTbl':
                row_index = index
        data_string = df.iloc[row_index, 1]
        split_data = data_string.split("value={")
        salesTurn=False
        # List to hold the rows of the table
        table_data = []
        # Display the results
        for index, match in enumerate(split_data):
            # print(f"Part {index}:\n{match}\n")
            if index > 1:
            # match = "'quantity': DocumentField(value_type=string, value='1', content=1, bounding_regions=[BoundingRegion(page_number=1, polygon=[Point(x=702.0, y=1286.0), Point(x=744.0, y=1287.0), Point(x=743.0, y=1370.0), Point(x=701.0, y=1369.0)])], spans=[DocumentSpan(offset=327, length=1)], confidence=None), 'unitPrice': DocumentField(value_type=string, value='11.72', content=11.72, bounding_regions=[BoundingRegion(page_number=1, polygon=[Point(x=2086.0, y=1256.0), Point(x=2298.0, y=1261.0), Point(x=2296.0, y=1367.0), Point(x=2084.0, y=1362.0)])], spans=[DocumentSpan(offset=329, length=5)], confidence=None), 'amount': DocumentField(value_type=string, value='1672', content=1672, bounding_regions=[BoundingRegion(page_number=1, polygon=[Point(x=2469.0, y=1263.0), Point(x=2711.0, y=1262.0), Point(x=2711.0, y=1370.0), Point(x=2469.0, y=1371.0)])], spans=[DocumentSpan(offset=335, length=4)], confidence=None)}, content=None, bounding_regions=[], spans=[], confidence=None), DocumentField(value_type=dictionary,"
                quantity = self.extract_field('quantity', match)
                description = self.extract_field('description', match)
                unit_price = self.extract_field('unitPrice', match)
                amount = self.extract_field('amount', match)
                total = self.extract_field('total', match)
                table_data.append([total, quantity, description, unit_price, amount])
        
        columns = ['Total','Quantity', 'Description', 'Unit Price', 'Amount']
        df2 = pd.DataFrame(table_data, columns=columns)
        try:
            # Search for the row containing "LESS:" in 'Total' column
            rowTotalSalesLess = df2[df2['Total'].str.contains("LESS", case=False, na=False) &
                                    df2['Total'].str.contains("Total Sales", case=False, na=False)]

            if not rowTotalSalesLess.empty:
                # Index of the row that contains "LESS:"
                idx = rowTotalSalesLess.index[0]

                # Split the 'Total' column into two parts
                total_sales_text = "Total Sales (VAT Inclusive)"
                less_vat_text = "LESS: VAT"

                # Extract the amounts: assuming Amount has two numbers separated by space
                amount_values = df2.at[idx, 'Amount']
                if isinstance(amount_values, str):
                    amounts = [float(x) for x in amount_values.split()]
                    # print(amounts)
                else:
                    raise ValueError("Amount field format is invalid for splitting.")

                if len(amounts) >= 2:
                    # Insert a new row for "LESS: VAT" after the current row
                    df2.loc[idx, 'Total'] = total_sales_text  # Modify the existing row
                    df2.loc[idx, 'Amount'] = amounts[0]

                    # Ensure consistent dtypes when adding the new row
                    new_row = pd.DataFrame([{
                        'Total': less_vat_text,
                        'Quantity': float('nan'),  # Explicitly set as NaN for float dtype
                        'Description': None,       # String dtype
                        'Unit Price': float('nan'),
                        'Amount': amounts[1]
                    }])
                    df2 = pd.concat([df2.iloc[:idx + 1], new_row, df2.iloc[idx + 1:]]).reset_index(drop=True)
                else:
                    raise ValueError("Amount column does not have enough numbers to split.")
        except Exception as e:
            # Handle any exceptions
            print(f"An error occurred: {e}")

        try:
            # Search for the row containing "TOTAL AMOUNT DUE" in 'Total' column
            rowTotalAmountDue = df2[df2['Total'].str.contains("TOTAL AMOUNT DUE", case=False, na=False) &
            df2['Total'].str.contains("ZERO Rated Sales", case=False, na=False)]

            zero_rated_sales_text = "ZERO Rated Sales"
            total_amount_due_text = "TOTAL AMOUNT DUE"
            if not rowTotalAmountDue.empty:
                # Index of the row containing "TOTAL AMOUNT DUE"
                idx = rowTotalAmountDue.index[0]

                # Extract amounts: Assuming multiple numbers are present in 'Amount' column
                amount_values = df2.at[idx, 'Amount']
                if isinstance(amount_values, str):
                    amounts = [float(x) for x in amount_values.split()]
                    # print(f"Extracted amounts: {amounts}")
                else:
                    raise ValueError("Amount field format is invalid for splitting.")

                # Check if at least two amounts exist
                if len(amounts) >= 2:
                    # print("chechk2")
                    # Modify the existing row to represent "ZERO Rated Sales"
                    df2.loc[idx, 'Total'] = zero_rated_sales_text
                    df2.loc[idx, 'Amount'] = amounts[0]

                    # Prepare a new row for "TOTAL AMOUNT DUE"
                    new_row = pd.DataFrame([{
                        'Total': total_amount_due_text,
                        'Quantity': float('nan'),
                        'Description': None,
                        'Unit Price': float('nan'),
                        'Amount': amounts[1]
                    }])
                    # Insert the new row after the current row
                    df2 = pd.concat([df2.iloc[:idx + 1], new_row, df2.iloc[idx + 1:]]).reset_index(drop=True)
                    # print("Row split successfully:")
                    # print(df2)
                elif len(amounts) < 2:
                    # print("check1")
                    df2.loc[idx, 'Total'] = total_amount_due_text
                    df2.loc[idx, 'Amount'] = amounts[0]
                    new_row = pd.DataFrame([{
                            'Total': total_amount_due_text,
                            'Quantity': float('nan'),
                            'Description': None,
                            'Unit Price': float('nan'),
                            'Amount': None
                        }])
                else:
                    raise ValueError("Amount column does not have enough numbers to split.")
        except Exception as e:
            # Handle any exceptions
            print(f"An error occurred: {e}")
        # Drop rows where `Unit Price` and `Amount` have no digits at all
        df2.dropna(how="all", inplace=True)
        df2 = df2[df2["Unit Price"].apply(self.has_digits) | df2["Amount"].apply(self.has_digits)]

        # name = file_ori_name.split("/")[-1]  # Get the last part of the URL
        # file_name = name.split(".")[0]  # Remove the file extension
        confidences_df = self.checkConfidentialThreshold(data, df2)
        print(confidences_df)
        val_list = self.check_thresholds(confidences_df, df2)
        return confidences_df, val_list, df2, df

    def convert_to_number(self, word):
            try:
                return w2n.word_to_num(word)
            except ValueError:
                return None 

    def clean_number(self, value):
        # If the value is a string and contains ',' or '.'
        if isinstance(value, str):
            # Check if there's more than one period
            value = value.replace(' ', '')
            if value.count('.') > 1:
                # Split the string at the last dot
                integer_part, decimal_part = value.rsplit('.', 1)
                # Remove commas from the integer part
                integer_part = integer_part.replace(',', '')
                # Remove any periods that are in the integer part
                integer_part = integer_part.replace('.', '')
                # Combine the cleaned integer part and the decimal part
                return f"{integer_part}.{decimal_part}"
            # If there's only one period (standard decimal), handle it as usual
            elif '.' in value:
                # Split integer and decimal part
                integer_part, decimal_part = value.split('.')
                # Remove commas from the integer part
                integer_part = integer_part.replace(',', '')
                # Check the length of the decimal part
                if len(decimal_part) > 2:
                    # Remove the decimal point if the decimal part is more than 2 digits
                    return integer_part + decimal_part
                else:
                    # Return with the decimal point intact if there are 2 or fewer decimal places
                    return f"{integer_part}.{decimal_part}"
            else:
                # If there's no decimal point, just remove commas
                return value.replace(',', '')
        return value

    def calculate_adjusted_values(self, row, confidences_df, df2):
        try:
            # Check if "Total" column is null
            if pd.isna(row["Total"]):
                # Retrieve and ensure confidences_df contains numeric values
                quantity_conf = pd.to_numeric(confidences_df.at[row.name, "Quantity"], errors='coerce')
                unit_price_conf = pd.to_numeric(confidences_df.at[row.name, "Unit Price"], errors='coerce')
                amount_conf = pd.to_numeric(confidences_df.at[row.name, "Amount"], errors='coerce')
                # Store confidences
                confidences = {
                    "Quantity": quantity_conf,
                    "Unit Price": unit_price_conf,
                    "Amount": amount_conf
                }
                # Drop invalid (NaN) values before proceeding
                confidences = {key: value for key, value in confidences.items() if not pd.isna(value)}
                # Debug printing
                print("Confidences after cleaning:", confidences)
                # Sort variables by confidence in descending order
                sorted_vars = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
                print("Sorted values: ", sorted_vars)

                if len(sorted_vars) < 2:
                    # Not enough valid values to proceed
                    print("Not enough valid values to calculate.")
                    return row
                highest, second_highest = sorted_vars[:2]
                print("Highest:", highest, "Second Highest:", second_highest)
                # Extract column names and row index
                highest_col = highest[0]
                second_highest_col = second_highest[0]
                row_index = row.name
                # Retrieve numeric values from df2 for calculations
                quantity = pd.to_numeric(df2.at[row_index, "Quantity"], errors='coerce')
                unit_price = pd.to_numeric(df2.at[row_index, "Unit Price"], errors='coerce')
                amount = pd.to_numeric(df2.at[row_index, "Amount"], errors='coerce')

                # Calculate missing values in df2 based on highest confidence
                if highest_col == "Quantity" and second_highest_col == "Amount":
                    df2.at[row_index, "Unit Price"] = amount / quantity if quantity != 0 else np.nan
                elif highest_col == "Quantity" and second_highest_col == "Unit Price":
                    df2.at[row_index, "Amount"] = quantity * unit_price
                elif highest_col == "Unit Price" and second_highest_col == "Amount":
                    df2.at[row_index, "Quantity"] = amount / unit_price if unit_price != 0 else np.nan
                elif highest_col == "Unit Price" and second_highest_col == "Quantity":
                    df2.at[row_index, "Amount"] = quantity * unit_price
                elif highest_col == "Amount" and second_highest_col == "Quantity":
                    df2.at[row_index, "Unit Price"] = amount / quantity if quantity != 0 else np.nan
                elif highest_col == "Amount" and second_highest_col == "Unit Price":
                    df2.at[row_index, "Quantity"] = amount / unit_price if unit_price != 0 else np.nan
                print(f"Updated row {row_index} in df2.")
            else:
                print(f"Row {row.name} skipped as Total is not null.")
        except Exception as e:
            print(f"An error occurred while processing row {row.name}: {e}")
        return row

    # Function to correct quantity, unit price, and amount
    def correct_values(self, row):
        # Retrieve values safely
        quantity = row["Quantity"] if pd.notna(row["Quantity"]) else 0
        unit_price = row["Unit Price"] if pd.notna(row["Unit Price"]) else 0
        amount = row["Amount"]

        # If amount is NaN or empty, calculate it
        if pd.isna(amount) and quantity > 0 and unit_price > 0:
            row["Amount"] = round(quantity * unit_price, 2)  # Set calculated amount

        # Check if the amount matches for "Total Sales (VAT Inclusive)"
        if "Total Sales (VAT Inclusive)" in str(row["Total"]):
            calculated_total = round(quantity * unit_price, 2)
            if not math.isclose(calculated_total, row["Amount"], rel_tol=1e-6):
                print(f"Discrepancy detected: Calculated: {calculated_total}, Provided: {row['Amount']}")

        # If current values don't match the formula, try to correct
        if not math.isclose(quantity * unit_price, amount, rel_tol=1e-6):
            # Correct quantity if possible
            if unit_price != 0:
                corrected_quantity = amount / unit_price
                if corrected_quantity.is_integer():
                    row["Quantity"] = corrected_quantity
                    return row

            # Correct unit price if possible
            if quantity != 0:
                corrected_unit_price = amount / quantity
                row["Unit Price"] = corrected_unit_price
                return row

        return row

    def clean_invoice_id(self, row):
        if row["Field Name"] == "invoiceID":
            # Extract digits from the 'Field Value' using regex
            row["Field Value"] = "".join(re.findall(r"\d+", row["Field Value"]))
        return row

    #invoice Type processing
    def processInvoiceType(self, row):
        if row["Field Name"] == "invoiceType":
            invoice_value = row["Field Value"].upper()  # Convert to uppercase for consistency

            # Condition checks
            if "LPG" in invoice_value:
                row["Field Value"] = "CASH SALES INVOICE-LPG"
            elif "GASES" in invoice_value:
                row["Field Value"] = "CASH SALES INVOICE-GASES"
            else:
                row["Field Value"] = "CHARGE SALES INVOICE"

        return row

    def findAndUpdateAddress(self, df):
        address = None  # Placeholder for scraped address

        # Step 1: Find the 'soldTo' row and scrape its address
        for index, row in df.iterrows():
            if row["Field Name"] == "soldTo":
                soldTo = row["Field Value"]  # Extract soldTo value
                url = f"https://google.com/search?q=address+of+{soldTo}+philippines"

                try:
                    # Send HTTP request and scrape the address
                    request_result = requests.get(url)
                    soup = bs4.BeautifulSoup(request_result.text, "html.parser")

                    # Extract address (update tag/class as needed)
                    address = soup.find("div", class_="BNeawe").text
                    print(f"Scraped Address for '{soldTo}': {address}")
                    break  # Exit loop after finding the address

                except Exception as e:
                    print(f"Error finding address for {soldTo}: {e}")
                    address = "Address Not Found"
                    break

        # Step 2: Update the 'address' row's Field Value
        if address:
            for index, row in df.iterrows():
                if row["Field Name"] == "address":
                    df.at[index, "Field Value"] = address  # Update the address row
                    print(f"Updated 'address' Field Value: {address}")
                    break

        return df
    
    def execute_post_processing(self, df2, df, confidences_df):
        """Executes the post-processing logic."""
        df2.loc[:, 'Quantity'] = df2['Quantity'].apply(self.convert_to_number)
        df2.loc[:, 'Amount'] = df2['Amount'].apply(self.clean_number)
        df2.loc[:, 'Unit Price'] = df2['Unit Price'].apply(self.clean_number)

        # Update specific rows
        values_to_update = [
            "Total Sales (VAT Included)",
            "LESS: 12% VAT",
            "Net of VAT/Total",
            "Total Amount Due"
        ]

        if len(df2) >= 4:
            last_indices = df2.index[-4:].tolist()
            for i, value in enumerate(values_to_update[::-1]):
                if df2.loc[last_indices[-1 - i], "Total"] != value:
                    df2.loc[last_indices[-1 - i], "Total"] = value
        else:
            if len(df2) >= 2:
                second_last_index = df2.index[-2]
                last_index = df2.index[-1]

                df2.loc[second_last_index + 0.5] = None
                df2.loc[second_last_index + 1.5] = None
                df2 = df2.sort_index().reset_index(drop=True)

                print("Added two new rows. Updated DataFrame:\n", df2)

        # VAT and Amount calculations
        try:
            rowTotalSales = df2[df2['Total'].str.contains("Total Sales", case=False, na=False)]
            rowTotalDue = df2[df2['Total'].str.contains("Total Amount Due", case=False, na=False)]
            rowTotalLess = df2[df2['Total'].str.contains("Less", case=False, na=False)]
            rowTotalNetOf = df2[df2['Total'].str.contains("Net of", case=False, na=False)]

            confidences_df.apply(self.calculate_adjusted_values, axis=1, args=(confidences_df, df2))
            valid_rows = df2[pd.isna(df2['Total']) & df2['Quantity'].notna() & df2['Unit Price'].notna() & df2['Amount'].notna()]

            valid_rows['Quantity'] = pd.to_numeric(valid_rows['Quantity'], errors='coerce')
            valid_rows['Unit Price'] = pd.to_numeric(valid_rows['Unit Price'], errors='coerce')
            valid_rows['Amount'] = pd.to_numeric(valid_rows['Amount'], errors='coerce')

            valid_rows = valid_rows.dropna(subset=['Quantity', 'Unit Price', 'Amount'])
            total_amount = valid_rows['Amount'].sum()
            print("Total calculated from valid rows:", total_amount)

            if not rowTotalSales.empty:
                df2.loc[rowTotalSales.index[0], 'Amount'] = total_amount
                df2.loc[rowTotalDue.index[0], 'Amount'] = total_amount

                VatPercentage = 0.892856862745098
                NetOfVat = float(total_amount) * VatPercentage
                VAT = float(total_amount) - NetOfVat

                if not rowTotalLess.empty:
                    df2.loc[rowTotalLess.index[0], 'Amount'] = round(VAT, 2)
                else:
                    print("Error: rowTotalLess is empty, cannot update VAT.")

                if not rowTotalNetOf.empty:
                    df2.loc[rowTotalNetOf.index[0], 'Amount'] = round(NetOfVat, 2)
                else:
                    print("Error: rowTotalNetOf is empty, cannot update NetOfVat.")

            else:
                raise ValueError("Row containing 'total sales' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        df2.loc[:, "Quantity"] = df2["Quantity"].astype(float)
        df2.loc[:, "Unit Price"] = df2["Unit Price"].astype(float)
        df2.loc[:, "Amount"] = pd.to_numeric(df2["Amount"], errors='coerce')

        df2 = df2.apply(self.correct_values, axis=1)
        df = df.apply(self.clean_invoice_id, axis=1)
        df = self.findAndUpdateAddress(df)
        df = df.apply(self.processInvoiceType, axis=1)

        print("Final Processed DataFrames:")
        print(df)
        print(df2)

    def correct_values_validated_val(self, row):
        try:
            # Retrieve values safely
            quantity = row["Quantity"] if pd.notna(row["Quantity"]) else 0
            unit_price = row["Unit Price"] if pd.notna(row["Unit Price"]) else 0
            amount = row["Amount"]

            # If amount is NaN or empty, calculate it
            if quantity > 0 and unit_price > 0:
                row["Amount"] = round(quantity * unit_price, 2)  # Set calculated amount
            return row

        except Exception as e:
            print(f"An error occurred while processing row {row.name}: {e}")
            return row

    def execute_post_processing_validated_val(self, df2, df, confidences_df):
        """Executes the post-processing logic."""
        df2.loc[:, 'Quantity'] = df2['Quantity'].apply(self.convert_to_number)
        df2.loc[:, 'Amount'] = df2['Amount'].apply(self.clean_number)
        df2.loc[:, 'Unit Price'] = df2['Unit Price'].apply(self.clean_number)
        df2.loc[:, 'Quantity'] = df2['Quantity'].apply(self.convert_to_number)
        df2.loc[:, "Quantity"] = df2["Quantity"].astype(float)
        df2.loc[:, 'Amount'] = df2['Amount'].apply(self.clean_number)
        df2.loc[:, 'Unit Price'] = df2['Unit Price'].apply(self.clean_number)

        print("before post_processing: ")
        print(df2)
        print(" ")
        # Update specific rows
        values_to_update = [
            "Total Sales (VAT Included)",
            "LESS: 12% VAT",
            "Net of VAT/Total",
            "Total Amount Due"
        ]

        if len(df2) >= 4:
            last_indices = df2.index[-4:].tolist()
            for i, value in enumerate(values_to_update[::-1]):
                if df2.loc[last_indices[-1 - i], "Total"] != value:
                    df2.loc[last_indices[-1 - i], "Total"] = value

        else:
            if len(df2) >= 2:
                second_last_index = df2.index[-2]
                last_index = df2.index[-1]

                df2.loc[second_last_index + 0.5] = None
                df2.loc[second_last_index + 1.5] = None
                df2 = df2.sort_index().reset_index(drop=True)

                print("Added two new rows. Updated DataFrame:\n", df2)

        # VAT and Amount calculations
        try:
            df2 = df2.apply(self.correct_values_validated_val, axis=1)
            print("after post_processing: ")
            print(df2)
            print(" ")

            rowTotalSales = df2[df2['Total'].str.contains("Total Sales", case=False, na=False)]
            rowTotalDue = df2[df2['Total'].str.contains("Total Amount Due", case=False, na=False)]
            rowTotalLess = df2[df2['Total'].str.contains("Less", case=False, na=False)]
            rowTotalNetOf = df2[df2['Total'].str.contains("Net of", case=False, na=False)]

            # confidences_df.apply(self.calculate_adjusted_values, axis=1, args=(confidences_df, df2))
            valid_rows = df2[pd.isna(df2['Total']) & df2['Quantity'].notna() & df2['Unit Price'].notna() & df2['Amount'].notna()]

            valid_rows['Quantity'] = pd.to_numeric(valid_rows['Quantity'], errors='coerce')
            valid_rows['Unit Price'] = pd.to_numeric(valid_rows['Unit Price'], errors='coerce')
            valid_rows['Amount'] = pd.to_numeric(valid_rows['Amount'], errors='coerce')

            valid_rows = valid_rows.dropna(subset=['Quantity', 'Unit Price', 'Amount'])
            total_amount = valid_rows['Amount'].sum()
            print("Total calculated from valid rows:", total_amount)

            if not rowTotalSales.empty:
                df2.loc[rowTotalSales.index[0], 'Amount'] = total_amount
                df2.loc[rowTotalDue.index[0], 'Amount'] = total_amount

                VatPercentage = 0.892856862745098
                NetOfVat = float(total_amount) * VatPercentage
                VAT = float(total_amount) - NetOfVat

                if not rowTotalLess.empty:
                    df2.loc[rowTotalLess.index[0], 'Amount'] = round(VAT, 2)
                else:
                    print("Error: rowTotalLess is empty, cannot update VAT.")

                if not rowTotalNetOf.empty:
                    df2.loc[rowTotalNetOf.index[0], 'Amount'] = round(NetOfVat, 2)
                else:
                    print("Error: rowTotalNetOf is empty, cannot update NetOfVat.")

            else:
                raise ValueError("Row containing 'total sales' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        df2.loc[:, "Quantity"] = df2["Quantity"].astype(float)
        df2.loc[:, "Unit Price"] = df2["Unit Price"].astype(float)
        df2.loc[:, "Amount"] = pd.to_numeric(df2["Amount"], errors='coerce')
        
        
        df = df.apply(self.clean_invoice_id, axis=1)
        df = self.findAndUpdateAddress(df)
        df = df.apply(self.processInvoiceType, axis=1)

        print("Final Processed DataFrames:")
        print(df)
        print(df2)

    

    def docu_processing(self, input_image):
        print(input_image)
        ai_studio_endpoint = 'https://azureaistudioh5078952159.cognitiveservices.azure.com/'
        ai_studio_key = '9BMknek8xn6Le7c9AG3THiBj8MhbN33MtGLjSpmAs5hMWDjWbB27JQQJ99BAACYeBjFXJ3w3AAAAACOGHKb6'
        endpoint = "https://docu-int-demo-prycegas.cognitiveservices.azure.com/"
        key = "376e1adee6124181baf381262ec40ccc"
        model_id = "invoice-prycegas4"
        # Use DocumentAnalysisClient instead of DocumentIntelligenceClient
        document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        
        with open(input_image, "rb") as file:
                poller = document_analysis_client.begin_analyze_document(model_id, file)
        
        invoices = poller.result()
        print("check")
        invoice_data = []
        for page in invoices.pages:
            for word in page.words:
                invoice_data.append([
                    page.page_number, 
                    word.content, 
                    word.confidence
                ])

        invoice_df = pd.DataFrame(invoice_data, columns=['Page Number', 'Word Content', 'Confidence'])
        invoice_df["Word ID"] = invoice_df.index + 1
        field_data = []

        # Iterate through the documents and fields
        for document in invoices.documents:
            for name, field in document.fields.items():
                # Extract the field value, type, and confidence
                field_value = str(field.value) if field.value else str(field.content)
                field_data.append([
                    name,  # Field Name
                    field_value,  # Field Value
                    field.value_type,  # Field Type
                    field.confidence  # Confidence
                ])
        # Convert the list to a pandas DataFrame
        table_df = pd.DataFrame(field_data, columns=['Field Name', 'Field Value', 'Field Type', 'Confidence'])
        # Airtable Credentials
        # Airtable configuration
        API_KEY = "pat5OcW9QNOKiAhag.3ebe39f745213b2186de1ecd4bcaa4140b4d1df7519bafe048bb42e3ea77418e"
        BASE_ID = "appFA2iUehCVrUSdl"
        # Headers
        HEADERS = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        TABLE_NAME = "Table 7"

        # Airtable API URL
        URL = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"

        for _, row in invoice_df.iterrows():
            # confidence_value = row['Confidence']
            # if isinstance(confidence_value, float) and (math.isnan(confidence_value) or math.isinf(confidence_value)):
            #     confidence_value = 0.0  # Replace NaN or Inf with a default value
            # # print(row['Field Name'])
            DATA = {
                "fields": {
                    "Page Number": str(row['Page Number']),
                    "Word Content": str(row['Word Content']),
                    "Confidence": str(row['Confidence']),
                    "Word ID":str(row['Word ID'])
                }
            }
            # print(json.dumps(DATA, indent=4))
            # Send the request to Airtable
            response = requests.post(URL, json=DATA, headers=HEADERS)
            
            # Check the response
            if response.status_code in [200, 201]:
                print(f"Record inserted successfully: {response.json()}")
            else:
                print(f"Error inserting record: {response.status_code}, {response.text}")

        TABLE_NAME_2 = "Table 6"

        # Airtable API URL
        URL2 = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME_2}"

        for _, row in table_df.iterrows():
            confidence_value = row['Confidence']
            if isinstance(confidence_value, float) and (math.isnan(confidence_value) or math.isinf(confidence_value)):
                confidence_value = 0.0  # Replace NaN or Inf with a default value
            # print(row['Field Name'])
            DATA = {
                "fields": {
                    "Field Name": str(row['Field Name']),
                    "Field Value": str(row['Field Value']),
                    "Field Type": str(row['Field Type']),
                    "Confidence": confidence_value  # Ensure a valid number
                }
            }
            # print(json.dumps(DATA, indent=4))
            # Send the request to Airtable
            response2 = requests.post(URL2, json=DATA, headers=HEADERS)
 
            if response2.status_code in [200, 201]:
                print(f"Record inserted successfully: {response.json()}")
            else:
                print(f"Error inserting record: {response.status_code}, {response.text}")
        return table_df, invoice_df

    def ocr_post_processing(self, df, data):
        row_index = 0
        for index, row in df.iterrows():
            if row['Field Name'] == 'invoiceTbl':
                row_index = index
        data_string = df.iloc[row_index, 1]
        split_data = data_string.split("value={")
        salesTurn=False
        # List to hold the rows of the table
        table_data = []
        # Display the results
        for index, match in enumerate(split_data):
            # print(f"Part {index}:\n{match}\n")
            if index > 1:
            # match = "'quantity': DocumentField(value_type=string, value='1', content=1, bounding_regions=[BoundingRegion(page_number=1, polygon=[Point(x=702.0, y=1286.0), Point(x=744.0, y=1287.0), Point(x=743.0, y=1370.0), Point(x=701.0, y=1369.0)])], spans=[DocumentSpan(offset=327, length=1)], confidence=None), 'unitPrice': DocumentField(value_type=string, value='11.72', content=11.72, bounding_regions=[BoundingRegion(page_number=1, polygon=[Point(x=2086.0, y=1256.0), Point(x=2298.0, y=1261.0), Point(x=2296.0, y=1367.0), Point(x=2084.0, y=1362.0)])], spans=[DocumentSpan(offset=329, length=5)], confidence=None), 'amount': DocumentField(value_type=string, value='1672', content=1672, bounding_regions=[BoundingRegion(page_number=1, polygon=[Point(x=2469.0, y=1263.0), Point(x=2711.0, y=1262.0), Point(x=2711.0, y=1370.0), Point(x=2469.0, y=1371.0)])], spans=[DocumentSpan(offset=335, length=4)], confidence=None)}, content=None, bounding_regions=[], spans=[], confidence=None), DocumentField(value_type=dictionary,"
                quantity = self.extract_field('quantity', match)
                description = self.extract_field('description', match)
                unit_price = self.extract_field('unitPrice', match)
                amount = self.extract_field('amount', match)
                total = self.extract_field('total', match)
                table_data.append([total, quantity, description, unit_price, amount])

        columns = ['Total','Quantity', 'Description', 'Unit Price', 'Amount']
        df2 = pd.DataFrame(table_data, columns=columns)

        try:
            # Search for the row containing "LESS:" in 'Total' column
            rowTotalSalesLess = df2[df2['Total'].str.contains("LESS", case=False, na=False) &
                                    df2['Total'].str.contains("Total Sales", case=False, na=False)]

            if not rowTotalSalesLess.empty:
                # Index of the row that contains "LESS:"
                idx = rowTotalSalesLess.index[0]

                # Split the 'Total' column into two parts
                total_sales_text = "Total Sales (VAT Inclusive)"
                less_vat_text = "LESS: VAT"

                # Extract the amounts: assuming Amount has two numbers separated by space
                amount_values = df2.at[idx, 'Amount']
                if isinstance(amount_values, str):
                    amounts = [float(x) for x in amount_values.split()]
                    print(amounts)
                else:
                    raise ValueError("Amount field format is invalid for splitting.")

                if len(amounts) >= 2:
                    # Insert a new row for "LESS: VAT" after the current row
                    df2.loc[idx, 'Total'] = total_sales_text  # Modify the existing row
                    df2.loc[idx, 'Amount'] = amounts[0]

                    # Ensure consistent dtypes when adding the new row
                    new_row = pd.DataFrame([{
                        'Total': less_vat_text,
                        'Quantity': float('nan'),  # Explicitly set as NaN for float dtype
                        'Description': None,       # String dtype
                        'Unit Price': float('nan'),
                        'Amount': amounts[1]
                    }])
                    print(new_row)
                    df2 = pd.concat([df2.iloc[:idx + 1], new_row, df2.iloc[idx + 1:]]).reset_index(drop=True)
                    print(df2)
                else:
                    raise ValueError("Amount column does not have enough numbers to split.")
        except Exception as e:
            # Handle any exceptions
            print(f"An error occurred: {e}")

        try:
            # Search for the row containing "TOTAL AMOUNT DUE" in 'Total' column
            rowTotalAmountDue = df2[df2['Total'].str.contains("TOTAL AMOUNT DUE", case=False, na=False) &
            df2['Total'].str.contains("ZERO Rated Sales", case=False, na=False)]

            zero_rated_sales_text = "ZERO Rated Sales"
            total_amount_due_text = "TOTAL AMOUNT DUE"
            if not rowTotalAmountDue.empty:
                # Index of the row containing "TOTAL AMOUNT DUE"
                idx = rowTotalAmountDue.index[0]

                # Extract amounts: Assuming multiple numbers are present in 'Amount' column
                amount_values = df2.at[idx, 'Amount']
                if isinstance(amount_values, str):
                    amounts = [float(x) for x in amount_values.split()]
                    print(f"Extracted amounts: {amounts}")
                else:
                    raise ValueError("Amount field format is invalid for splitting.")

                # Check if at least two amounts exist
                if len(amounts) >= 2:
                    print("chechk2")
                    # Modify the existing row to represent "ZERO Rated Sales"
                    df2.loc[idx, 'Total'] = zero_rated_sales_text
                    df2.loc[idx, 'Amount'] = amounts[0]

                    # Prepare a new row for "TOTAL AMOUNT DUE"
                    new_row = pd.DataFrame([{
                        'Total': total_amount_due_text,
                        'Quantity': float('nan'),
                        'Description': None,
                        'Unit Price': float('nan'),
                        'Amount': amounts[1]
                    }])
                    # Insert the new row after the current row
                    df2 = pd.concat([df2.iloc[:idx + 1], new_row, df2.iloc[idx + 1:]]).reset_index(drop=True)
                    print("Row split successfully:")
                    print(df2)
                elif len(amounts) < 2:
                    print("check1")
                    df2.loc[idx, 'Total'] = total_amount_due_text
                    df2.loc[idx, 'Amount'] = amounts[0]
                    new_row = pd.DataFrame([{
                            'Total': total_amount_due_text,
                            'Quantity': float('nan'),
                            'Description': None,
                            'Unit Price': float('nan'),
                            'Amount': None
                        }])
                else:
                    raise ValueError("Amount column does not have enough numbers to split.")
        except Exception as e:
            # Handle any exceptions
            print(f"An error occurred: {e}")

        # Drop rows where `Unit Price` and `Amount` have no digits at all
        df2.dropna(how="all", inplace=True)

        df2 = df2[df2["Unit Price"].apply(self.has_digits) | df2["Amount"].apply(self.has_digits)]
        confidences_df = self.checkConfidentialThreshold(data, df2)
        
        df2.loc[:, 'Quantity'] = df2['Quantity'].apply(self.convert_to_number)
        df2.loc[:, 'Amount'] = df2['Amount'].apply(self.clean_number)
        df2.loc[:, 'Unit Price'] = df2['Unit Price'].apply(self.clean_number)
        values_to_update = [
            "Total Sales (VAT Included)",
            "LESS: 12% VAT",
            "Net of VAT/Total",
            "Total Amount Due"
        ]

        # Ensure df2 has at least 4 rows
        if len(df2) >= 4:
            # Find the indices of the last 4 rows dynamically
            last_indices = df2.index[-4:].tolist()

            # Update only the rows that need changes
            for i, value in enumerate(values_to_update[::-1]):  # Reverse the list to match order
                if df2.loc[last_indices[-1 - i], "Total"] != value:
                    df2.loc[last_indices[-1 - i], "Total"] = value
        else:
            if len(df2) >= 2:
                second_last_index = df2.index[-2]
                last_index = df2.index[-1]

                # Insert two new rows
                df2.loc[second_last_index + 0.5] = None
                df2.loc[second_last_index + 1.5] = None

                # Re-sort the index to maintain order
                df2 = df2.sort_index().reset_index(drop=True)

                # Debugging output
                print("Added two new rows. Updated DataFrame:\n", df2)

            # print("Not enough rows in the DataFrame to update the last four rows.")

        if len(df2) >= 4:
            # Find the indices of the last 4 rows dynamically
            last_indices = df2.index[-4:].tolist()

            # Update only the rows that need changes
            for i, value in enumerate(values_to_update[::-1]):  # Reverse the list to match order
                if df2.loc[last_indices[-1 - i], "Total"] != value:
                    df2.loc[last_indices[-1 - i], "Total"] = value

        flags = []
        totalSales = None
        try:
            rowTotalSales = df2[df2['Total'].str.contains("Total Sales", case=False, na=False)]
            print("check1")
            rowTotalDue = df2[df2['Total'].str.contains("Total Amount Due", case=False, na=False)]
            # print("check2")
            rowTotalLess = df2[df2['Total'].str.contains("Less", case=False, na=False)]
            # print("check3")
            rowTotalNetOf = df2[df2['Total'].str.contains("Net of", case=False, na=False)]
            confidences_df.apply(self.calculate_adjusted_values, axis=1, args=(confidences_df, df2))
            valid_rows = df2[pd.isna(df2['Total']) &
                            df2['Quantity'].notna() &
                            df2['Unit Price'].notna() &
                            df2['Amount'].notna()]

            # Convert values to numeric
            valid_rows['Quantity'] = pd.to_numeric(valid_rows['Quantity'], errors='coerce')
            valid_rows['Unit Price'] = pd.to_numeric(valid_rows['Unit Price'], errors='coerce')
            valid_rows['Amount'] = pd.to_numeric(valid_rows['Amount'], errors='coerce')

            # Ensure all values in each row are numeric
            valid_rows = valid_rows.dropna(subset=['Quantity', 'Unit Price', 'Amount'])

            # Sum the amounts
            total_amount = valid_rows['Amount'].sum()
            print("Total calculated from valid rows:", total_amount)

            # Update the row containing "Total Sales"
            if not rowTotalSales.empty:
                print("chck 3")
                df2.loc[rowTotalSales.index[0], 'Amount'] = total_amount
                df2.loc[rowTotalDue.index[0], 'Amount'] = total_amount
                print("chck 2")
                print(f"Updated Total Sales row with total amount: {total_amount}")
                # df2.loc[rowTotalDue.index[0], 'Amount'] = total_amount
                VatPercentage = 0.892856862745098
                NetOfVat = float(total_amount) * VatPercentage
                VAT = float(total_amount) - NetOfVat
                print("chck 4")
                if not rowTotalLess.empty:
                    df2.loc[rowTotalLess.index[0], 'Amount'] = round(VAT, 2)
                else:
                    print("Error: rowTotalLess is empty, cannot update VAT.")

                if not rowTotalNetOf.empty:
                    df2.loc[rowTotalNetOf.index[0], 'Amount'] = round(NetOfVat, 2)
                else:
                    print("Error: rowTotalNetOf is empty, cannot update NetOfVat.")

            else:
                raise ValueError("Row containing 'total sales' not found.")
            print(f"Total Sales: {totalSales}")
        except Exception as e:
            # Handle any exceptions
            print(f"An error occurred: {e}")
        df2.loc[:, "Quantity"] = df2["Quantity"].astype(float)
        df2.loc[:, "Unit Price"] = df2["Unit Price"].astype(float)
        df2.loc[:, "Amount"] = pd.to_numeric(df2["Amount"], errors='coerce')  # Convert to float, invalid entries as NaN
        # Apply corrections to each row
        df2 = df2.apply(self.correct_values, axis=1)
        # Apply the function to clean the invoiceID row
        df = df.apply(self.clean_invoice_id, axis=1)

        # Apply the function to the DataFrame
        df = self.findAndUpdateAddress(df)
        # Apply the function to the DataFrame
        df = df.apply(self.processInvoiceType, axis=1)
        return df, df2

    async def _handle_incoming_attachment(self, turn_context: TurnContext):
        for attachment in turn_context.activity.attachments:
            attachment_info = await self._download_attachment_and_write(attachment)
            if "filename" in attachment_info:
                file_path = attachment_info["local_path"]
                # Send confirmation message
                await turn_context.send_activity(
                    f"Attachment **{attachment_info['filename']}** has been received and saved to **{file_path}**."
                )
                image = file_path
                # Send the uploaded image back for visualization
                reply = MessageFactory.attachment(
                    Attachment(
                        name=attachment_info["filename"],
                        content_type=attachment.content_type,  # Ensure the correct content type
                        content_url=attachment.content_url  # Display image from the URL
                    )
                )
                await turn_context.send_activity(reply)
                return image
        return None

    async def _download_attachment_and_write(self, attachment: Attachment) -> dict:
        """
        Retrieve the attachment via the attachment's contentUrl.
        :param attachment:
        :return: Dict: keys "filename", "local_path"
        """
        try:
            response = urllib.request.urlopen(attachment.content_url)
            headers = response.info()

            # If user uploads JSON file, this prevents it from being written as
            # "{"type":"Buffer","data":[123,13,10,32,32,34,108..."
            if headers["content-type"] == "application/json":
                data = bytes(json.load(response)["data"])
            else:
                data = response.read()

            local_filename = os.path.join(os.getcwd(), attachment.name)
            with open(local_filename, "wb") as out_file:
                out_file.write(data)

            return {"filename": attachment.name, "local_path": local_filename}
        except Exception as exception:
            print(exception)
            return {}

    async def _handle_outgoing_attachment(self, turn_context: TurnContext):
        user_choice = turn_context.activity.text  # Get user input
        reply = Activity(type=ActivityTypes.message)
        attachments = []

        if user_choice == "1":
            reply.text = "This is an inline attachment."
            attachments = [self._get_inline_attachment()]

        elif user_choice == "2":
            reply.text = "This is an internet attachment."
            attachments = [self._get_internet_attachment()]

        elif user_choice == "3":
            reply.text = "This is an uploaded attachment."
            uploaded_attachment = await self._get_upload_attachment(turn_context)
            attachments = [uploaded_attachment]

        else:
            reply.text = "Your input was not recognized, please try again."

        reply.attachments = attachments
        await turn_context.send_activity(reply)

        return attachments  # Return attachments list

    async def _display_options(self, turn_context: TurnContext):
        """
        Create a HeroCard with options for the user to interact with the bot.
        :param turn_context:
        :return:
        """

        # Note that some channels require different values to be used in order to get buttons to display text.
        # In this code the emulator is accounted for with the 'title' parameter, but in other channels you may
        # need to provide a value for other parameters like 'text' or 'displayText'.
        card = HeroCard(
            text="You can upload an image or select one of the following choices",
            buttons=[
                CardAction(
                    type=ActionTypes.im_back, title="1. Inline Attachment", value="1"
                ),
                CardAction(
                    type=ActionTypes.im_back, title="2. Internet Attachment", value="2"
                ),
                CardAction(
                    type=ActionTypes.im_back, title="3. Uploaded Attachment", value="3"
                ),
            ],
        )

        reply = MessageFactory.attachment(CardFactory.hero_card(card))
        await turn_context.send_activity(reply)

    def _get_inline_attachment(self) -> Attachment:
        """
        Creates an inline attachment sent from the bot to the user using a base64 string.
        Using a base64 string to send an attachment will not work on all channels.
        Additionally, some channels will only allow certain file types to be sent this way.
        For example a .png file may work but a .pdf file may not on some channels.
        Please consult the channel documentation for specifics.
        :return: Attachment
        """
        file_path = os.path.join(os.getcwd(), "resources/architecture-resize.png")
        with open(file_path, "rb") as in_file:
            base64_image = base64.b64encode(in_file.read()).decode()

        return Attachment(
            name="architecture-resize.png",
            content_type="image/png",
            content_url=f"data:image/png;base64,{base64_image}",
        )

    async def _get_upload_attachment(self, turn_context: TurnContext) -> Attachment:
        """
        Creates an "Attachment" to be sent from the bot to the user from an uploaded file.
        :param turn_context:
        :return: Attachment
        """
        with open(
            os.path.join(os.getcwd(), "Template24.png"), "rb"
        ) as in_file:
            image_data = in_file.read()

        connector = await turn_context.adapter.create_connector_client(
            turn_context.activity.service_url
        )
        conversation_id = turn_context.activity.conversation.id
        response = await connector.conversations.upload_attachment(
            conversation_id,
            AttachmentData(
                name="Template24.png",
                original_base64=image_data,
                type="image/png",
            ),
        )

        base_uri: str = connector.config.base_url
        attachment_uri = (
            base_uri
            + ("" if base_uri.endswith("/") else "/")
            + f"v3/attachments/{response.id}/views/original"
        )

        return Attachment(
            name="Template24.png",
            content_type="image/png",
            content_url=attachment_uri,
        )

    def _get_internet_attachment(self) -> Attachment:
        """
        Creates an Attachment to be sent from the bot to the user from a HTTP URL.
        :return: Attachment
        """
        return Attachment(
            name="Template1.png",
            content_type="image/png",
            content_url="https://aiautomationbot001.blob.core.windows.net/input/Template1.png",
        )

    async def on_members_added_activity(self, members_added: List[ChannelAccount], turn_context: TurnContext):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello! Please upload your invoice(s) here.")
    

    async def on_message_activity(self, turn_context: TurnContext):
        try:
            # Prevent the bot from processing its own messages
            if turn_context.activity.from_property.id == turn_context.activity.recipient.id:
                return  # Ignore bot's own messages
            
            if turn_context.activity.id in self.processed_messages:
                return  # Ignore duplicate messages
            self.processed_messages.add(turn_context.activity.id)
            
            if turn_context.activity.attachments and len(turn_context.activity.attachments) > 0:
                await turn_context.send_activity("Invoice received! Processing your data...")
                
                for attachment in turn_context.activity.attachments:
                    print(f"Attachment content URL: {attachment.content_url}")

                # Capture the returned image file path from _handle_incoming_attachment
                image = await self._handle_incoming_attachment(turn_context)
                print(image)
                if image:
                    # Pass the image file path to docu_processing
                    table_df, invoice_df = self.docu_processing(image)
                    df, df2 = self.ocr_post_processing(table_df, invoice_df)
                    df_text = df.to_markdown() if not df.empty else "No data found in df."
                    df2_text = df2.to_markdown() if not df2.empty else "No data found in df2."

                    # Send results back to the user
                    await turn_context.send_activity(f"**Processed Invoice Table:**\n```\n{df_text}\n```")
                    await turn_context.send_activity(f"**Processed Invoice Summary:**\n```\n{df2_text}\n```")
   

                # # invoice_df, table_df, debug_info = self.fetch_airtable_data()
                # df, df2 = self.ocr_post_processing(table_df, invoice_df)

                # # Convert DataFrames to text
                # df_text = df.to_markdown() if not df.empty else "No data found in df."
                # df2_text = df2.to_markdown() if not df2.empty else "No data found in df2."

                # # Send results back to the user
                # await turn_context.send_activity(f"**Processed Invoice Table:**\n```\n{df_text}\n```")
                # await turn_context.send_activity(f"**Processed Invoice Summary:**\n```\n{df2_text}\n```")

        except Exception as e:
            print(f"An error occurred: {e}")    
            await turn_context.send_activity("An error occurred while processing the invoice.")

    
    # async def on_message_activity(self, turn_context: TurnContext):
    #     try:
    #         if turn_context.activity.attachments and len(turn_context.activity.attachments) > 0:
    #             await turn_context.send_activity("Invoice received! Processing your data...")
    #             for attachment in turn_context.activity.attachments:
    #                 print(f"Attachment content URL: {attachment.content_url}")

    #             # Capture the returned image file path from _handle_incoming_attachment
    #             image = await self._handle_incoming_attachment(turn_context)
    #             print(image)
    #             if image:
    #                 # Pass the image file path to docu_processing
    #                 result = self.docu_processing(image)
    #                 print(result)

    #             table_df = pd.DataFrame()
    #             invoice_df = pd.DataFrame()
    #             invoice_df, table_df, debug_info = self.fetch_airtable_data()
    #             df, df2 = self.ocr_post_processing(table_df, invoice_df)

    #             # Convert DataFrames to text
    #             df_text = df.to_markdown() if not df.empty else "No data found in df."
    #             df2_text = df2.to_markdown() if not df2.empty else "No data found in df2."

    #             # Send results back to the user
    #             await turn_context.send_activity(f"**Processed Invoice Table:**\n```\n{df_text}\n```")
    #             await turn_context.send_activity(f"**Processed Invoice Summary:**\n```\n{df2_text}\n```")

    #             '''actual code end here'''
            # else:
            #     await self._display_options(turn_context)  # Show options
            #     attachments = await self._handle_outgoing_attachment(turn_context)  # Get selected attachments
            #     # Extract content URLs if attachments exist
            #     if attachments:
            #         response_urls = [attachment.content_url for attachment in attachments]
            #         print(f"Selected Attachment URLs: {response_urls}")
            #         result = self.docu_processing(response_urls)
            #         print(result)  # Debug output
            #     # print(f"Status Code: {response.status_code}")
            #     # print(f"Headers: {response.headers}")
            #     # print(f"Response Text: {response.text}")  # Log the full response body
            #     # image_data = response.content  # Get the raw image bytes
            #     # print("check2")
            #     # Convert the image data to a PFImage
            #     # pf_image = PFImage(image_data, mime_type=attachment.content_type)
            #     # print("check3")
            #     # print(pf_image)
            #     # Pass the PFImage to the OCR function


            #     # result = self.docu_processing(response)
            #     # print(result)  # Debug output
        except Exception as e:
            print(f"An error occurred: {e}")    
            await turn_context.send_activity("An error occurred while processing the invoice.")

        

        

        # user_status = await self.user_data_accessor.get(turn_context, lambda: {"first_message": False})
        # question_index = await self.question_index_accessor.get(turn_context, lambda: 0)
        # answers = await self.answers_accessor.get(turn_context, lambda: [])

        # table_df = pd.DataFrame()
        # invoice_df = pd.DataFrame()
        # invoice_df, table_df, debug_info = self.fetch_airtable_data()
        # confidences_df, val_list, df2, df = self.post_processing(table_df, invoice_df)
        

        # if not user_status["first_message"]:
        #     df2_preview = df2.to_markdown(index=False)  
        #     await turn_context.send_activity(f"Here is the extracted data before validation:\n```\n{df2_preview}\n```")

        
        # # Save val_list to state if it's the first interaction
        # if not user_status["first_message"]:
        #     await self.val_list_accessor.set(turn_context, val_list)
        # else:
        #     val_list = await self.val_list_accessor.get(turn_context, lambda: [])

        
        # if val_list:  # If there are validation questions
        #     # Check if this is the first question
        #     if "question_index" not in user_status:
        #         user_status["question_index"] = 0  # Initialize index for tracking questions
        #         user_status["updated_val_list"] = val_list.copy()  # Keep a copy of original values

        #     question_index = user_status["question_index"]

        #     # If receiving a response from user
        #     if user_status["first_message"] and question_index > 0:
        #         user_input = turn_context.activity.text.strip()
                
        #         # Process user input
        #         if user_input.lower() != "yes":
        #             try:
        #                 new_value = float(user_input)  # Convert to float
        #                 user_status["updated_val_list"][question_index - 1][2] = new_value  # Update value
        #             except ValueError:
        #                 await turn_context.send_activity("Invalid input. Please enter a valid number.")
        #                 return  # Re-ask the same question

        #     # Ask next question if available
        #     if question_index < len(val_list):
        #         row_num, col_name, cell_value = val_list[question_index]
        #         question_text = (
        #             f"Is the extracted value '{cell_value}' in row {row_num} and column '{col_name}' correct? "
        #             f"Type 'yes' if correct, or type the actual number if it's wrong."
        #         )
        #         await turn_context.send_activity(question_text)
                
        #         # Move to the next question
        #         user_status["question_index"] += 1
        #         user_status["first_message"] = True  

        #     else:
        #         # All questions answered, update df2
        #         await turn_context.send_activity("All questions answered. Updating values...")
        #         print("Updated val_list:", user_status["updated_val_list"])

        #         for index, col, new_value in user_status["updated_val_list"]:
        #             print("+==============+")
        #             print(index, " ", col, " ", new_value)
        #             df2.at[index, col] = new_value  # Apply updates to df2

        #         # Reset tracking variables
        #         user_status["first_message"] = False
        #         user_status.pop("question_index", None)
        #         user_status.pop("updated_val_list", None)
        #         print(df2)

        #         # Proceed with post-processing
        #         self.execute_post_processing_validated_val(df2, df, confidences_df)
        #         df2_preview = df2.to_markdown(index=False)  # Convert DataFrame to Markdown format
        #         await turn_context.send_activity(f"Here is the extracted data before validation:\n```\n{df2_preview}\n```")

        # else:  # If val_list is empty, directly proceed with post-processing
        #     self.execute_post_processing(df2, df, confidences_df)
        #     df2_preview = df2.to_markdown(index=False)  # Convert DataFrame to Markdown format
        #     await turn_context.send_activity(f"Here is the extracted data before validation:\n```\n{df2_preview}\n```")

        # # Save state
        # await self.user_data_accessor.set(turn_context, user_status)
        # await self.question_index_accessor.set(turn_context, question_index)
        # await self.answers_accessor.set(turn_context, answers)
        # await self.val_list_accessor.set(turn_context, val_list)
        # await self.user_state.save_changes(turn_context)

