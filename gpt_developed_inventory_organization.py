# -*- coding: utf-8 -*-
"""GPT Developed Inventory Organization

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mrT-tymKAo80GPjpjjjJeYnOlsnSZN7j

API Playground:
You are a professional electrician and expert in electrical supply parts. Your expertise includes knowledge of the parts use cases and functionality. Your job is turn an abbreviated part description into a detailed, concise description in 10 words or less.

Part: LT400I
Description: 4-IN INS L/T CONN
Manufacturer: Dottie

Using the description created with the information provided, determine two categories for the item that are industry relevant: Main and Sub categories.

Load API Key and Imports
---
---
"""

!pip install --upgrade openai
!pip install fuzzywuzzy

import pandas as pd, numpy as np, openai, os, json, re, requests, time, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from fuzzywuzzy import fuzz

warnings.filterwarnings('ignore', category=FutureWarning)

"""Load Pandas Data Frame
---
---
* Read excel file of updated bin locations
"""

bin_location_data = pd.read_excel('/content/HH_Updated_Data_Bin_Locations.xlsx')

"""Clean Data Frame
---

---
* Clean data frame by removing empty rows and fully naming manufacturers

"""

def clean_data_frame(bin_location_data):
    """ Cleans data frame by removing empty rows and fully naming manufacturers. """
    mfr_dict = {
        "DOT" : "Dottie",
        "CH" : "Eaton",
        "BLINE" : "Cooper B-Line",
        "MIL" : "Milbank",
        "LEV" : "Leviton",
        "ITE" : "Siemens",
        "GEIND" : "General Electric Industrial",
        "UNIPA" : "Union Pacific",
        "GARV" : "Garvin Industries",
        "FIT" : "American Fittings",
        "TAY" : "TayMac",
        "ARL" : "Arlington",
        "AMFI" : "American Fittings",
        "BPT" : "Bridgeport",
        "CCHO" : "Eaton Course-Hinds",
        "HARGR" : "Harger",
        "CARLN" : "Carlon",
        "MULB" : "Mulberry",
        "SOLAR" : "Solarline",
        "ENERL" : "Enerlites",
        "HUBWD" : "Hubble Wiring Device",
        "DMC" : "DMC Power",
        "INT" : "Intermatic",
        "LUT" : "Lutron",
        "LITTE" : "Littelfuse",
        "GRNGA" : "GreenGate",
        "WATT" : "Wattstopper",
        "SENSO" : "Sensor Switch",
        "CHE" : "Eaton Crouse Hinds",
        "OZ" : "OZ Gedney",
    }

    # Remove empty rows for CatalogNo
    cleaned_bin_location_data = bin_location_data[bin_location_data["CatalogNo"] != "EMPTY"].copy()

    # Replace MfrCode with full names if they exist in mfr_dict
    cleaned_bin_location_data.loc[:, 'MfrCode'] = cleaned_bin_location_data['MfrCode'].apply(lambda x: mfr_dict.get(x, x))

    return cleaned_bin_location_data

cleaned_bin_location_data = clean_data_frame(bin_location_data)

"""Set up api for prompt
---
---
* Uses bin locations data to categorize prompt

"""

def get_bin_data(cleaned_bin_location_data):
    """ Sets up bin location data from pandas data frame to dictionary in order to integrate with API. """
    dict_bin_data = cleaned_bin_location_data.to_dict(orient='index')  # Convert DataFrame to dictionary

    return dict_bin_data

pretty_json = get_bin_data(cleaned_bin_location_data)
print(pretty_json)

"""API KEY"""

# Set the environment variable
os.environ['API_KEY'] = "####"

# Access it later in the code
api_key = os.environ.get('API_KEY')

"""WarehouseInventory Class - Item Categorization Classification using ChatCompletion Engine GPT-4
---
---
- Set up class for determining Warehouse Inventory item categories (main and sub categories).
    - Connect to GPT-4 model via OpenAI API
    - Create prompt to enrich item description by using Catalog number, Manufacturer, and Description
    - Use enriched item description to gather main/sub categorization for items
    - Considere similar items before categorization, to see if items should be categorized together
    - Update data frame with category classifications and full GPT response data
"""

# Testing new version 10/3 (Not working)

import time
import requests
import pandas as pd
from typing import List, Tuple
import logging

class CategorizedWarehouseInventory:
    """
    A class used to represent a categorized warehouse inventory.

    ...

    Attributes
    ----------
    data : pd.DataFrame
        A DataFrame containing the warehouse inventory data.
    existing_categories : dict
        A dictionary to store existing categories.

    Methods
    -------
    call_gpt_api(prompt: str) -> str:
        Sends a request to the GPT-4 API and returns the response content.

    batch_gpt_api_request(prompts: List[str]) -> List[str]:
        Sends a batch request to the GPT-4 API and returns a list of response content.

    calculate_similarity(item1: str, item2: str) -> float:
        Calculates the similarity between two text strings or Catalog Numbers.

    find_similar_items(catalog_no: str) -> Tuple[str, str]:
        Finds similar items based on the Catalog Number.

    enrich_description(catalog_no: str, description: str, mfr_code: str) -> str:
        Enriches the product description using the GPT-4 API.

    enrich_all_descriptions():
        Iterates through the DataFrame enriching each product's description.

    determine_category(catalog_no: str, enriched_description: str, category_type: str, main_category: str = '') -> str:
        Determines the category of an item using the GPT-4 API or existing categories.

    categorize_item(enriched_description: str, catalog_no: str) -> Tuple[str, str]:
        Categorizes an item into a main and sub-category.

    update_existing_categories(main_category: str, sub_category: str, catalog_no: str, enriched_description: str):
        Updates the dictionary of existing categories with new categorizations.

    update_dataframe():
        Iterates through the DataFrame, enriching descriptions, and assigning categories.

    reevaluate_categories():
        Re-evaluates the categorizations of items post-categorization.

    show_dataframe() -> pd.DataFrame:
        Returns the first five rows of the DataFrame for inspection.

    gather_data():
        Enriches all descriptions, classifies main and sub-categories, and saves the DataFrame.
    """

    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    model = "gpt-4"
    max_tokens = 35
    top_p = 0.05
    temperature = 0.1

    last_request_time = 0
    min_interval = 1 / (200 / 60)  # minimal interval between requests to stay within the limit

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the CategorizedWarehouseInventory object with the given data.

        Parameters:
        ----------
        data : pd.DataFrame
            The warehouse inventory data. Expected columns include 'CatalogNo', 'Description', and 'MfrCode'.

        Raises:
        -------
        ValueError
            If the provided data is not a DataFrame or is missing required columns.
        """
        self.data = data
        self.data['Main Category'] = None
        self.data['Sub-category'] = None
        self.data['Max Similarity Score'] = None
        self.data['Most Similar Item'] = None
        self.data['Enriched Description'] = None
        self.existing_categories = {}

    def call_gpt_api(self, prompt: str):
        """
        Sends a request to the GPT-4 API based on the given prompt and returns the response content.

        Parameters:
        ----------
        prompt : str
            The prompt to be sent to the GPT-4 API.

        Returns:
        -------
        str
            The response content from the GPT-4 API.

        Raises:
        -------
        requests.exceptions.RequestException
            If the API call fails for any reason.
        """

        # Calculate the time since the last request
        time_since_last_request = time.time() - self.last_request_time

        # If the time since the last request is less than the minimal interval,
        # sleep for the remainder of the interval
        if time_since_last_request < self.min_interval:
            time.sleep(self.min_interval - time_since_last_request)

        try:
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional electrician and expert in electrical supply parts. Your expertise includes knowledge of the parts use cases and functionality."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            }

            gpt_response = requests.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json=data)
            gpt_response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

            response_dict = gpt_response.json()
            response_content = response_dict['choices'][0]['message']['content'].strip()

            return json.loads(response_content)

        except requests.exceptions.RequestException as e:
            logging.error(f"API Call Failed. Reason: {e}")
            return ""
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON. Reason: {e}")
            return ""

    def batch_gpt_api_request(self, items: list[dict]) -> list[str]:
        """
        Sends a batch request to the GPT-4 API with multiple prompts and returns a list of response content.

        Parameters:
        ----------
        items : List[dict]
            A list of dictionaries, each containing the 'CatalogNo', 'MfrCode', and 'Description' of an item.

        Returns:
        -------
        List[str]
            A list of response content from the GPT-4 API corresponding to each prompt.

        Raises:
        -------
        requests.exceptions.RequestException
            If the API call fails for any reason.
        """
        prompt_parts = [f"Part {i + 1}: Part number = {item['CatalogNo']}, manufacturer = {item['MfrCode']}, description = {item['Description']}"
                        for i, item in enumerate(items)]
        joined_prompt_parts = '\n'.join(prompt_parts)
        prompt = (
            f"Your job is to create a detailed, concise part description for each part that's 10 words or less. "
            f"Use any available information found from Part number, manufacturer, and description. "
            f"Your output should be formatted as a JSON array, with each element being an object containing the part number and enriched description. "
            f"For example: [{'EC200': 'enriched description 1'}, {'EC250': 'enriched description 2'}, ...]\n"
            f"{joined_prompt_parts}"
            )

        response_content = self.call_gpt_api(prompt)

        return response_content

    @staticmethod
    def validate_string_input(*args) -> None:
        for arg in args:
            if not isinstance(arg, str):
                raise ValueError("Expected a string input.")


    @staticmethod
    def calculate_similarity(item1: str, item2: str) -> float:
        """
        Calculates the similarity between two text strings or Catalog Numbers based on a defined metric.

        Parameters:
        ----------
        item1 : str
            The first text string or Catalog Number.
        item2 : str
            The second text string or Catalog Number.

        Returns:
        -------
        float
            The similarity score between the two items.

        Raises:
        -------
        ValueError
            If either item1 or item2 is not a string.
        """

        CategorizedWarehouseInventory.validate_string_input(item1, item2)
        vectorizer = TfidfVectorizer().fit_transform([item1, item2])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors)
        return cosine_sim[0, 1]

    def find_similar_items(self, catalog_no: str) -> Tuple[str, str]:
        """
        Finds similar items based on the Catalog Number.

        Parameters:
        ----------
        catalog_no : str
            The Catalog Number of the item.

        Returns:
        -------
        Tuple[str, str]
            A tuple containing the most common Manufacturer Code and Description among similar items.

        Raises:
        -------
        ValueError
            If catalog_no is not a string.
        """
        self.validate_string_input(catalog_no)
        similarity_scores = []
        for index, item in self.data.iterrows():
            score = self.calculate_similarity(catalog_no, item['CatalogNo'])
            similarity_scores.append((index, score))
        similarity_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity score
        most_similar_index = similarity_scores[1][0]  # Index 1 since index 0 would be the item itself
        most_similar_item = self.data.loc[most_similar_index]
        return most_similar_item['MfrCode'], most_similar_item['Description']


    def enrich_description(self, catalog_no: str, description: str, mfr_code: str) -> str:
        """
        Enriches the product description using the GPT-4 API.

        Parameters:
        ----------
        catalog_no : str
            The Catalog Number of the item.
        description : str
            The current description of the item.
        mfr_code : str
            The Manufacturer Code of the item.

        Returns:
        -------
        str
            The enriched description obtained from the GPT-4 API.

        Raises:
        -------
        ValueError
            If any of the input parameters are not strings.
        """
        prompt = (
            f"Provide a detailed and concise description for the part with "
            f"Part number: {catalog_no}, Manufacturer: {mfr_code}, "
            f"Existing Description: {description}. The description should be 10 words or less."
        )
        enriched_description = self.call_gpt_api(prompt)
        return enriched_description

    def determine_category(self, catalog_no: str, enriched_description: str, category_type: str, main_category: str = '') -> str:
        """
        Determines the category of an item using the GPT-4 API or existing categories.

        Parameters:
        ----------
        catalog_no : str
            The Catalog Number of the item.
        enriched_description : str
            The enriched description of the item.
        category_type : str
            The type of category to determine ('main' or 'sub').
        main_category : str, optional
            The main category, if already determined (default is '').

        Returns:
        -------
        str
            The determined category.

        Raises:
        -------
        ValueError
            If any of the input parameters are not strings or category_type is not 'main' or 'sub'.
        """
        prompt = (
            f"Determine the {category_type} of the part with "
            f"Part number: {catalog_no}, Enriched Description: {enriched_description}"
        )
        if main_category:
            prompt += f", Main Category: {main_category}"
        category = self.call_gpt_api(prompt)
        return category


    def categorize_item(self, enriched_description: str, catalog_no: str) -> Tuple[str, str]:
        """
        Categorizes an item into a main and sub-category.

        Parameters:
        ----------
        enriched_description : str
            The enriched description of the item.
        catalog_no : str
            The Catalog Number of the item.

        Returns:
        -------
        Tuple[str, str]
            A tuple containing the main and sub-category.

        Raises:
        -------
        ValueError
            If either enriched_description or catalog_no is not a string.
        """
        pass  # Implementation goes here


    def update_existing_categories(self, main_category: str, sub_category: str, catalog_no: str, enriched_description: str):
        """
        Updates the dictionary of existing categories with new categorizations.

        Parameters:
        ----------
        main_category : str
            The main category of the item.
        sub_category : str
            The sub-category of the item.
        catalog_no : str
            The Catalog Number of the item.
        enriched_description : str
            The enriched description of the item.

        Raises:
        -------
        ValueError
            If any of the input parameters are not strings.
        """
        pass  # Implementation goes here

    def update_dataframe(self):
        """
        Iterates through the DataFrame, enriching descriptions, and assigning categories.

        Raises:
        -------
        Exception
            If an error occurs during the update process.
        """
        pass  # Implementation goes here

    def reevaluate_categories(self):
        """
        Re-evaluates the categorizations of items post-categorization.

        Raises:
        -------
        Exception
            If an error occurs during the reevaluation process.
        """
        pass  # Implementation goes here

    def show_dataframe(self) -> pd.DataFrame:
        """
        Returns the first five rows of the DataFrame for inspection.

        Returns:
        -------
        pd.DataFrame
            The first five rows of the DataFrame.
        """
        pass  # Implementation goes here

    def gather_data(self):
        """
        Enriches all descriptions, classifies main and sub-categories, and saves the DataFrame.

        Raises:
        -------
        Exception
            If an error occurs during the data gathering process.
        """
        pass  # Implementation goes here

# Testing new version #WORKING VERSION

# Set up WarehouseInventory class and methods for classifying item categories
class WarehouseInventory:

    # Class variables for API headers and other constant parameters
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    model = "gpt-4"
    max_tokens = 35
    top_p = 0.05
    temperature = 0.1

    def __init__(self, data):
        # Initialize instance variables and new DataFrame columns
        self.data = data
        self.data['Main Category'] = None
        self.data['Sub-category'] = None
        self.existing_categories = {}

    @staticmethod
    def call_gpt_api(prompt):
        time.sleep(1 / 190 * 60)  # Sleep for approximately 316 milliseconds
        try:
            data = {
                "model": WarehouseInventory.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional electrician and expert in electrical supply parts. Your expertise includes knowledge of the parts use cases and functionality."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                "max_tokens": WarehouseInventory.max_tokens,
                "top_p": WarehouseInventory.top_p,
                "temperature": WarehouseInventory.temperature
            }

            gpt_response = requests.post("https://api.openai.com/v1/chat/completions", headers=WarehouseInventory.headers, json=data)
            gpt_response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

            response_dict = gpt_response.json()
            response_content = response_dict['choices'][0]['message']['content'].strip()

            #print(f"API Call Successful. Response: {response_content}")  # Debugging line
            return response_content

        except requests.exceptions.RequestException as e:
            print(f"API Call Failed. Reason: {e}")  # Debugging line
            return ""

    # Vectorizing the CatalogNo check
    def has_common_substring(self, str1, str2, length=3):
        return any(sub in str1 for sub in [str2[i:i+length] for i in range(len(str2) - length + 1)])

    def find_similarity(self, text1, text2):
        # Compute the cosine similarity between two text strings
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        return cosine_similarity(vectors)[0, 1]

    def enrich_description(self, catalog_no, description, mfr_code):
        # Generate a prompt and call GPT API to enrich the product description
        prompt = f"Your job is create a detailed, concise part description that's 10 words or less.  Use any available information found from Part number: {catalog_no}, manufacturer: {mfr_code}, and description: {description}."
        enriched_description = self.call_gpt_api(prompt)
        print(f'{enriched_description} compared to original description: {description}')
        return enriched_description

    def enrich_all_descriptions(self):
        with tqdm(total=len(self.data), desc="Enriching Descriptions", unit=" Items") as pbar:
            for index, row in self.data.iterrows():
                catalog_no = row['CatalogNo']
                description = row['Description']
                mfr_code = row['MfrCode']
                enriched_description = self.enrich_description(catalog_no, description, mfr_code)
                self.data.at[index, 'Enriched Description'] = enriched_description
                pbar.update(1)

    def determine_category_gpt(self, catalog_no, enriched_description, category_type, main_category=''):
        # Generate a prompt and call GPT API to determine the category of an item
        if category_type == 'main':
            prompt = f"Review the description for {catalog_no}: Description - '{enriched_description}'. Determine the broad {category_type} category in one word or a short phrase."
        else:
            prompt = f"Review the description for {catalog_no}: Description - '{enriched_description}'. Determine a {category_type} category in one word or a short phrase more specific than {main_category}."
        category = self.call_gpt_api(prompt)
        category = category.strip().capitalize()
        return category

    def find_similar_catalogs(self, catalog_no):
        def is_similar(str1, str2):
            return fuzz.ratio(str1, str2) > 80  # Change 80 to a threshold you find appropriate

        similar_catalogs = self.data[self.data['CatalogNo'].apply(lambda x: is_similar(x, catalog_no))]
        if not similar_catalogs.empty:
            return similar_catalogs['MfrCode'].mode()[0], similar_catalogs['Description'].mode()[0]
        return None, None


    def find_similar_catalogs(self, catalog_no):
        return self.data[self.data['CatalogNo'].apply(lambda x: fuzz.ratio(x, catalog_no) > 80)]

    def assign_category_from_similar_items(self, catalog_no):
        similar_catalogs = self.find_similar_catalogs(catalog_no)
        if not similar_catalogs.empty:
            main_category = similar_catalogs['Main Category'].mode()[0]
            sub_category = similar_catalogs['Sub-category'].mode()[0]
            return main_category, sub_category
        return None, None

    def use_gpt_for_categorization(self, enriched_description, catalog_no):
        # Separate logic for GPT-based categorization
        main_category = self.determine_category_gpt(catalog_no, enriched_description, 'main')
        sub_category = self.determine_category_gpt(catalog_no, enriched_description, 'sub', main_category)
        return main_category, sub_category

    def categorize_item(self, enriched_description, catalog_no):
        # Use helper methods to simplify this function

        # Try to categorize using similar items first
        main_category, sub_category = self.assign_category_from_similar_items(catalog_no)

        # If unsuccessful, try using GPT for categorization
        if main_category is None or sub_category is None:
            main_category, sub_category = self.use_gpt_for_categorization(enriched_description, catalog_no)

        # Update existing categories
        self.update_existing_categories(main_category, sub_category, catalog_no, enriched_description)

        return main_category, sub_category

    def update_existing_categories(self, main_category, sub_category, catalog_no, enriched_description):
        # Function to update existing categories
        new_row = pd.DataFrame({
            'CatalogNo': [catalog_no],
            'Enriched Description': [enriched_description],
            'Sub-category': [sub_category]
        })

        if main_category not in self.existing_categories:
            self.existing_categories[main_category] = pd.DataFrame(columns=['CatalogNo', 'Enriched Description', 'Sub-category'])

        self.existing_categories[main_category] = pd.concat([self.existing_categories[main_category], new_row], ignore_index=True)

    def update_dataframe(self):
        try:
            with tqdm(total=len(self.data), desc="Updated dataframe", unit="Parts") as pbar:
                for index, row in self.data.iterrows():
                    catalog_no = row.get('CatalogNo', 'Unknown')
                    description = row.get('Description', 'Unknown')
                    mfr_code = row.get('MfrCode', 'Unknown')

                    enriched_description = self.enrich_description(catalog_no, description, mfr_code)
                    self.data.at[index, 'Enriched Description'] = enriched_description

                    main_category, sub_category = self.categorize_item(enriched_description, catalog_no)

                    self.data.at[index, 'Main Category'] = main_category
                    self.data.at[index, 'Sub-category'] = sub_category

                    pbar.update(1)

                    if index % 100 == 0:
                        self.data.to_excel('categorized_bin_location_data_checkpoint.xlsx', index=False)

        except Exception as e:
            print(f'An error occurred: {e}')
            self.data.to_excel('categorized_bin_location_data_checkpoint.xlsx', index=False)
            pass
    def reevaluate_categories(self):
        # Create a copy of existing categories as we'll be modifying the original dict
        with tqdm(total=len(self.existing_categories), desc="Reevaluating Categories", unit="categories") as pbar:
            existing_categories_copy = self.existing_categories.copy()
            for main_category, items_df in existing_categories_copy.items():
                for index, row in items_df.iterrows():
                    # Call find_main_category to get the most similar main category for each item
                    new_main_category, _, most_similar_item, _, _ = self.find_main_category(row['Enriched Description'], row['CatalogNo'])

                    if new_main_category != main_category:
                        # Move item to new_main_category in self.existing_categories
                        self.existing_categories[new_main_category] = self.existing_categories[new_main_category].append(row)

                        # Update the main category in the main data frame as well
                        self.data.loc[self.data['CatalogNo'] == row['CatalogNo'], 'Main Category'] = new_main_category

                        # Remove the item from its old category
                        self.existing_categories[main_category].drop(index, inplace=True)
                    pbar.update(1)


    def show_dataframe(self):
        return self.data.head()

    def test_method(self, start_idx, end_idx):
        subset_data = self.data.iloc[start_idx:end_idx].copy()
        test_warehouse = WarehouseInventory(subset_data)
        test_warehouse.update_dataframe()
        test_warehouse.data.to_excel('categorized_bin_location_test_data.xlsx')
        return test_warehouse.show_dataframe()

    def gather_data(self):
        self.enrich_all_descriptions()
        self.classify_main_categories()
        self.classify_sub_categories()
        self.data.to_excel('categorized_bin_location_data.xlsx', index=False)

"""Running the Warehouse Inventory Category Classifier to Gather Data
---
---
"""

# Initialize the WarehouseInventory object
#warehouse = WarehouseInventory(cleaned_bin_location_data, api_key)
warehouse = WarehouseInventory(cleaned_bin_location_data)

warehouse.gather_data()

# Use the method
warehouse.reevaluate_categories()

# Test the method on lines 33-48 (which are at 0-based indices 33-35)
#test_result = warehouse.test_method(25, 30)

# Show the result
#print(test_result)