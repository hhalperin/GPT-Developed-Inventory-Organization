"""OpenAI GPT API client for the inventory tool."""

import logging
import time
import requests
from typing import Optional, Dict, Any, List
import pandas as pd
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

class GPTClient:
    """Client for interacting with OpenAI's GPT API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT client.
        
        Args:
            api_key: Optional OpenAI API key. If None, uses key from environment.
        """
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            
        self.api_key = api_key
        self.model = "gpt-4.1-nano-2025-04-14"
        self.max_tokens = 150
        self.temperature = 0.7
        self.top_p = 1.0
        self.api_url = "https://api.openai.com/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables or .env file")
            
    def call_gpt_api(self, prompt: str) -> str:
        """Call the OpenAI GPT API with a given prompt.
        
        Args:
            prompt: The prompt to send to the GPT API.
            
        Returns:
            The response from the GPT API.
            
        Raises:
            requests.RequestException: If API call fails.
        """
        time.sleep(0.01)  # Simple rate limiting
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": ("You are a professional electrician and expert in electrical supply parts. "
                              "Provide short, plain-English responses.")
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "temperature": self.temperature
        }
        
        try:
            logger.debug(f"Sending prompt to GPT API: {prompt[:100]}...")
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            logger.error(f"API call error: {str(e)}")
            raise
            
    def enrich_description(self, catalog_no: str, mfr_code: str, description: str) -> str:
        """Use GPT to produce an enriched, short description for a part, including CatalogNo and Manufacturer in the prompt.
        Args:
            catalog_no: Catalog number.
            mfr_code: Manufacturer code (expanded).
            description: The part description to enrich.
        Returns:
            Enriched description, or 'Unknown' if failed.
        """
        prompt = (
            f"Create an enriched part description in plain English (10 words or less). "
            f"CatalogNo: {catalog_no}, Manufacturer: {mfr_code}, Abbreviated Description: {description}."
        )
        try:
            return self.call_gpt_api(prompt)
        except Exception as e:
            logger.error(f"GPT enrichment failed for CatalogNo {catalog_no}: {e}")
            return "Unknown"

    def enrich_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich descriptions for all items in a DataFrame, logging progress and timing."""
        import time
        enriched_df = df.copy()
        enriched_col = 'Enriched Description'
        start = time.time()
        enriched = []
        for idx, row in enriched_df.iterrows():
            catalog_no = row.get('CatalogNo', '')
            mfr_code = row.get('MfrCode', '')
            description = row.get('Description', '')
            enriched_desc = self.enrich_description(catalog_no, mfr_code, description)
            enriched.append(enriched_desc)
            if (idx+1) % 10 == 0 or (idx+1) == len(enriched_df):
                logger.info(f"Enriched {idx+1}/{len(enriched_df)} descriptions...")
        enriched_df[enriched_col] = enriched
        elapsed = time.time() - start
        num_enriched = sum(x != "Unknown" for x in enriched)
        logger.info(f"Enrichment complete: {num_enriched}/{len(enriched_df)} items enriched successfully in {elapsed:.1f}s.")
        return enriched_df

    def validate_api_key(self) -> bool:
        """Validate the API key by making a test request.
        
        Returns:
            bool: True if API key is valid, False otherwise.
        """
        if not self.api_key:
            return False
            
        try:
            self.call_gpt_api("Test request")
            return True
        except requests.RequestException:
            return False

    def handle_api_error(self, error: requests.RequestException) -> None:
        """Handle API errors gracefully.
        
        Args:
            error: The RequestException that occurred.
        """
        if error.response is not None:
            status_code = error.response.status_code
            if status_code == 401:
                logger.error("Invalid API key")
            elif status_code == 429:
                logger.error("Rate limit exceeded")
            elif status_code == 500:
                logger.error("OpenAI server error")
            else:
                logger.error(f"API error: {error}")
        else:
            logger.error(f"Network error: {error}")

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics.
        
        Returns:
            Dictionary containing usage statistics.
        """
        try:
            response = requests.get(
                "https://api.openai.com/v1/usage",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            return {
                "total_requests": data.get("total_requests", 0),
                "total_tokens": data.get("total_tokens", 0),
                "successful_requests": data.get("successful_requests", 0),
                "failed_requests": data.get("failed_requests", 0)
            }
        except requests.RequestException as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "successful_requests": 0,
                "failed_requests": 0
            } 