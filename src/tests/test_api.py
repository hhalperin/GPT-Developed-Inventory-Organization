"""Tests for the API module."""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import requests

from src.api import GPTClient

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        "CatalogNo": ["SKU1", "SKU2", "SKU3"],
        "Description": ["Item 1", "Item 2", "Item 3"],
        "Main Category": ["Cat1", "Cat1", "Cat2"],
        "Sub-category": ["Sub1", "Sub2", "Sub3"]
    })

@pytest.fixture
def gpt_client():
    """Create a GPTClient instance for testing."""
    return GPTClient(api_key="test_key")

def test_init(gpt_client):
    """Test GPTClient initialization."""
    assert gpt_client.api_key == "test_key"
    assert gpt_client.model == "gpt-4.1-nano-2025-04-14"
    assert gpt_client.max_tokens == 150
    assert gpt_client.temperature == 0.7

@patch('requests.post')
def test_enrich_description(mock_post, gpt_client):
    """Test description enrichment."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Enhanced description"}}]
    }
    mock_post.return_value = mock_response
    
    # Test enrichment
    enriched = gpt_client.enrich_description("TestNo", "TestMfr", "Test item")
    assert enriched == "Enhanced description"
    mock_post.assert_called_once()

@patch('requests.post')
def test_enrich_descriptions(mock_post, gpt_client, sample_data):
    """Test batch description enrichment."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Enhanced description"}}]
    }
    mock_post.return_value = mock_response
    
    # Test batch enrichment
    enriched_data = gpt_client.enrich_descriptions(sample_data)
    assert isinstance(enriched_data, pd.DataFrame)
    assert len(enriched_data) == len(sample_data)
    assert "Enriched Description" in enriched_data.columns
    assert mock_post.call_count == len(sample_data)

@patch('requests.post')
def test_validate_api_key(mock_post, gpt_client):
    """Test API key validation."""
    # Mock successful API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}]
    }
    mock_post.return_value = mock_response
    assert gpt_client.validate_api_key()
    
    # Test invalid key
    mock_post.side_effect = requests.RequestException()
    assert not gpt_client.validate_api_key()

@patch('requests.post')
def test_handle_api_error(mock_post, gpt_client):
    """Test API error handling."""
    # Mock API error
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_post.side_effect = requests.RequestException(response=mock_response)
    
    # Test error handling
    with pytest.raises(requests.RequestException):
        gpt_client.enrich_description("TestNo", "TestMfr", "Test item")

@patch('requests.get')
def test_get_usage_statistics(mock_get, gpt_client):
    """Test getting API usage statistics."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_requests": 100,
        "total_tokens": 1000,
        "successful_requests": 95,
        "failed_requests": 5
    }
    mock_get.return_value = mock_response
    
    stats = gpt_client.get_usage_statistics()
    
    assert isinstance(stats, dict)
    assert stats["total_requests"] == 100
    assert stats["total_tokens"] == 1000
    assert stats["successful_requests"] == 95
    assert stats["failed_requests"] == 5 