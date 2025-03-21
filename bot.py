# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount


# class MyBot(ActivityHandler):
#     # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.

#     async def on_message_activity(self, turn_context: TurnContext):
#         await turn_context.send_activity(f"You said '{ turn_context.activity.text }'")

#     async def on_members_added_activity(
#         self,
#         members_added: ChannelAccount,
#         turn_context: TurnContext
#     ):
#         for member_added in members_added:
#             if member_added.id != turn_context.activity.recipient.id:
#                 await turn_context.send_activity("Hello and welcome!")

from botbuilder.core import ActivityHandler, TurnContext, MemoryStorage, UserState


#from promptflow import tool
import requests
import pandas as pd
import logging
import sys
import re
import requests
import time
import pandas as pd
import requests
import math
from word2number import w2n
import numpy as np
import bs4
import json
from botbuilder.core import MessageFactory
from botbuilder.schema import Attachment
    

class MyBot(ActivityHandler):
    def __init__(self, user_state: UserState):
        super().__init__()
        self.user_state = user_state
        self.user_data_accessor = self.user_state.create_property("user_status")
        self.question_index_accessor = self.user_state.create_property("question_index")
        self.answers_accessor = self.user_state.create_property("answers")
        self.val_list_accessor = self.user_state.create_property("val_list")

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

    async def on_message_activity(self, turn_context: TurnContext):
        user_status = await self.user_data_accessor.get(turn_context, lambda: {"first_message": False})
        question_index = await self.question_index_accessor.get(turn_context, lambda: 0)
        answers = await self.answers_accessor.get(turn_context, lambda: [])

        table_df = pd.DataFrame()
        invoice_df = pd.DataFrame()
        invoice_df, table_df, debug_info = self.fetch_airtable_data()
        confidences_df, val_list, df2, df = self.post_processing(table_df, invoice_df)
        print("after post_processing: ")
        print(df2)
        print(" ")

        if not user_status["first_message"]:
            df2_preview = df2.to_markdown(index=False)  
            await turn_context.send_activity(f"Here is the extracted data before validation:\n```\n{df2_preview}\n```")

        
        # Save val_list to state if it's the first interaction
        if not user_status["first_message"]:
            await self.val_list_accessor.set(turn_context, val_list)
        else:
            val_list = await self.val_list_accessor.get(turn_context, lambda: [])

        
        if val_list:  # If there are validation questions
            
            # Generate questions
            questions = [
                f"Is the extracted value '{cell_value}' in row {row_num} and column '{col_name}' correct? "
                f"Type 'yes' if correct, or type the actual number if it's wrong."
                for row_num, col_name, cell_value in val_list
            ]

            # If receiving a response, store the answer and update val_list
            if user_status["first_message"]:
                user_input = turn_context.activity.text.strip()

                if user_input.lower() != "yes":
                    try:
                        new_value = float(user_input)  # Convert to float
                        val_list[question_index][2] = new_value  # Update val_list
                    except ValueError:
                        await turn_context.send_activity("Invalid input. Please enter a valid number.")
                        return  # Ask the same question again

                answers.append(user_input)  # Store the response
                question_index += 1  # Move to the next question

            # Ask the next question if there are more
            if question_index < len(questions):
                await turn_context.send_activity(questions[question_index])
                user_status["first_message"] = True
            else:
                await turn_context.send_activity("All questions answered. Thank you!")
                print("Updated val_list:", val_list)
                user_status["first_message"] = False
                question_index = 0
                answers = []

                # Update df2 with new values
                for index, col, new_value in val_list:
                    df2.at[index, col] = new_value

                # Proceed with post-processing
                self.execute_post_processing_validated_val(df2, df, confidences_df)
                df2_preview = df2.to_markdown(index=False)  # Convert DataFrame to Markdown format
                await turn_context.send_activity(f"Here is the extracted data before validation:\n```\n{df2_preview}\n```")


        else:  # If val_list is empty, directly proceed with post-processing
            self.execute_post_processing(df2, df, confidences_df)
            df2_preview = df2.to_markdown(index=False)  # Convert DataFrame to Markdown format
            await turn_context.send_activity(f"Here is the extracted data before validation:\n```\n{df2_preview}\n```")

        # Save state
        await self.user_data_accessor.set(turn_context, user_status)
        await self.question_index_accessor.set(turn_context, question_index)
        await self.answers_accessor.set(turn_context, answers)
        await self.val_list_accessor.set(turn_context, val_list)
        await self.user_state.save_changes(turn_context)