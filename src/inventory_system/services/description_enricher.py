"""
Description enrichment module for the inventory system.
Handles GPT-based description enrichment and management.
"""

import time
import logging
from typing import Dict, Any
import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI

# Configure logging to suppress OpenAI client messages
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("http.client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DescriptionEnricher:
    """Handles GPT-based description enrichment for inventory items."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the description enricher.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = 35
        self.top_p = 0.05
        self.temperature = 0.1

    def call_gpt_api(self, prompt: str) -> str:
        """
        Call the GPT API with a given prompt.
        
        Args:
            prompt: The prompt to send to GPT
            
        Returns:
            str: GPT's response
        """
        time.sleep(0.01)  # Rate limiting
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional electrician and expert in electrical supply parts. Provide short, plain-English responses."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API call error: {e}")
            return ""

    def enrich_description(self, catalog_no: str, description: str, mfr_code: str) -> str:
        """
        Enrich a single item's description using GPT.
        
        Args:
            catalog_no: Item's catalog number
            description: Original description
            mfr_code: Manufacturer code
            
        Returns:
            str: Enriched description
        """
        prompt = (
            f"Create an enriched part description in plain English (10 words or less). "
            f"CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, "
            f"Abbreviated Description: {description}."
        )
        return self.call_gpt_api(prompt)

    def enrich_all_descriptions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich descriptions for all items in the dataset.
        
        Args:
            data: DataFrame containing items to enrich
            
        Returns:
            pd.DataFrame: DataFrame with enriched descriptions
        """
        enriched_data = data.copy()
        enriched_data['Enriched Description'] = None
        
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Enriching Descriptions"):
            catalog_no = row['CatalogNo']
            description = row['Description']
            mfr_code = row['MfrCode']
            
            enriched_description = self.enrich_description(catalog_no, description, mfr_code)
            enriched_data.at[index, 'Enriched Description'] = enriched_description
            
        return enriched_data 