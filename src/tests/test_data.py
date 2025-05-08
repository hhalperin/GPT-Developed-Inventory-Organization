"""Tests for data loading module."""

import pytest
import pandas as pd
from pathlib import Path

def test_init(data_loader):
    """Test DataLoader initialization."""
    assert data_loader.data is None
    assert data_loader.raw_data is None
    assert data_loader.cleaned_data is None

def test_load_data(data_loader, sample_data, tmp_path):
    """Test data loading."""
    # Create test files
    csv_file = tmp_path / "test.csv"
    xlsx_file = tmp_path / "test.xlsx"
    sample_data.to_csv(csv_file, index=False)
    sample_data.to_excel(xlsx_file, index=False)
    
    # Test CSV loading
    loaded_csv = data_loader.load_data(csv_file)
    assert isinstance(loaded_csv, pd.DataFrame)
    assert data_loader.raw_data is not None
    assert data_loader.data is not None
    
    # Test Excel loading
    loaded_xlsx = data_loader.load_data(xlsx_file)
    assert isinstance(loaded_xlsx, pd.DataFrame)
    
    # Test invalid file
    with pytest.raises(FileNotFoundError):
        data_loader.load_data(tmp_path / "nonexistent.csv")
        
    # Test invalid format (create empty .txt file first)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("")
    with pytest.raises(ValueError):
        data_loader.load_data(txt_file)

def test_clean_data_frame(data_loader, sample_data):
    """Test data frame cleaning."""
    cleaned_df = data_loader.clean_data_frame(sample_data)
    assert isinstance(cleaned_df, pd.DataFrame)
    assert all(col in cleaned_df.columns for col in [
        'CatalogNo', 'Description', 'Main Category', 'Sub-category', 'MfrCode'
    ])
    assert not cleaned_df['Description'].isna().any()
    assert not cleaned_df['Main Category'].isna().any()
    assert not cleaned_df['Sub-category'].isna().any()

def test_clean_data(data_loader, sample_data):
    """Test data cleaning."""
    # Test with no data loaded
    with pytest.raises(ValueError):
        data_loader.clean_data()
    
    # Test with data loaded
    data_loader.raw_data = sample_data
    cleaned_data = data_loader.clean_data()
    assert isinstance(cleaned_data, pd.DataFrame)
    assert data_loader.cleaned_data is not None

def test_get_manufacturer_code(data_loader):
    """Test manufacturer code extraction."""
    assert data_loader.get_manufacturer_code("ABB123") == "ABB"
    assert data_loader.get_manufacturer_code("1SDA123456") == "ABB"
    assert data_loader.get_manufacturer_code("3VA1234") == "SIEMENS"
    assert data_loader.get_manufacturer_code("Unknown123") == "Unknown"
    assert data_loader.get_manufacturer_code(None) == "Unknown"

def test_get_bin_data(data_loader, sample_data):
    """Test bin data conversion."""
    # Test with no cleaned data
    with pytest.raises(ValueError):
        data_loader.get_bin_data()
    
    # Test with cleaned data
    data_loader.cleaned_data = sample_data
    bin_data = data_loader.get_bin_data()
    assert isinstance(bin_data, dict)
    assert all(key in bin_data for key in [
        'catalog_numbers', 'descriptions', 'manufacturer_codes',
        'main_categories', 'sub_categories'
    ])

def test_save_data(data_loader, sample_data, tmp_path):
    """Test data saving."""
    data_loader.cleaned_data = sample_data
    
    # Test CSV saving
    csv_file = tmp_path / "test.csv"
    data_loader.save_data(csv_file)
    assert csv_file.exists()
    
    # Test Excel saving
    xlsx_file = tmp_path / "test.xlsx"
    data_loader.save_data(xlsx_file)
    assert xlsx_file.exists()
    
    # Test invalid format
    with pytest.raises(ValueError):
        data_loader.save_data(tmp_path / "test.txt")

def test_load_categorized_data(data_loader, sample_data, tmp_path):
    """Test loading categorized data."""
    # Test with no file
    assert data_loader.load_categorized_data() is None
    
    # Create test file
    cat_file = tmp_path / "categorized.xlsx"
    sample_data.to_excel(cat_file, index=False)
    
    # Test loading
    loaded_data = data_loader.load_categorized_data(cat_file)
    assert isinstance(loaded_data, pd.DataFrame)
    
    # Test invalid format (create empty .txt file first)
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("")
    with pytest.raises(ValueError):
        data_loader.load_categorized_data(txt_file)

def test_get_statistics(data_loader, sample_data):
    """Test getting statistics."""
    # Test with no cleaned data
    with pytest.raises(ValueError):
        data_loader.get_statistics()
    
    # Test with cleaned data
    data_loader.cleaned_data = sample_data
    stats = data_loader.get_statistics()
    assert isinstance(stats, dict)
    assert all(key in stats for key in [
        'total_items', 'unique_manufacturers', 'unique_categories',
        'unique_subcategories', 'missing_descriptions', 'missing_categories',
        'manufacturer_distribution', 'category_distribution',
        'subcategory_distribution'
    ])

def test_validate_data(data_loader, sample_data):
    """Test data validation."""
    assert data_loader.validate_data(sample_data) is True
    
    # Test invalid data
    invalid_data = pd.DataFrame({'Invalid': [1, 2, 3]})
    assert data_loader.validate_data(invalid_data) is False 