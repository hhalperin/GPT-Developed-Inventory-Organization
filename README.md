# GPT-Developed-Inventory-Organization
Script for classification of inventory item categories using GPT-4 for optimized organization. 

Introduction
This standalone Python script, gpt_developed_inventory_organization.py, automates the process of organizing and categorizing warehouse inventory. 
The script makes API calls to enrich product descriptions, remove empty rows, and categorize items based on various attributes like CatalogNo.

Features
Remove Empty Rows: Cleans up the data by removing rows where CatalogNo is empty.
Replace Manufacturer Codes: Replaces manufacturer codes with their full names based on a predefined dictionary.
Categorization with GPT: Uses GPT (Generative Pre-trained Transformer) models to categorize items and enrich product descriptions.
Rate-Limited API Calls: Incorporates time management between API calls to prevent rate-limiting issues.
Excel Output: Generates an organized inventory Excel file detailing how the inventory should be reorganized.

Dependencies
Python 3.x
Pandas
Requests (For API Calls)
[Any other specific dependencies]

How to Run
Clone the GitHub repository to your local machine.
Navigate to the project folder in your terminal.
Run the following command to execute the script:
Copy code
python gpt_developed_inventory_organization.py

Output
The script will generate an Excel file with the reorganized inventory, including enriched product descriptions and categorized items.

Limitations
Rate-limited API calls may slow down the process.
[Any other specific limitations]
