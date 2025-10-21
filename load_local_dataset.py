"""
Helper functions to load the local agriculture dataset
Use this instead of downloading from HuggingFace every time
"""

import pandas as pd
import os

def load_local_dataset(format='parquet'):
    """
    Load the agriculture dataset from local storage
    
    Args:
        format (str): File format to load - 'parquet' (recommended), 'csv', or 'json'
    
    Returns:
        pd.DataFrame: Dataset with 'question' and 'answer' columns
    
    Example:
        >>> df = load_local_dataset()
        >>> print(f"Loaded {len(df)} examples")
        >>> print(df.head())
    """
    
    # Define file paths
    file_paths = {
        'parquet': 'data/agriculture_qa.parquet',
        'csv': 'data/agriculture_qa.csv',
        'json': 'data/agriculture_qa.json'
    }
    
    if format not in file_paths:
        raise ValueError(f"Format must be one of: {list(file_paths.keys())}")
    
    file_path = file_paths[format]
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            f"Please run: python download_dataset.py"
        )
    
    # Load based on format
    print(f"📦 Loading dataset from local storage ({format})...")
    
    if format == 'parquet':
        df = pd.read_parquet(file_path)
    elif format == 'csv':
        df = pd.read_csv(file_path)
    elif format == 'json':
        df = pd.read_json(file_path)
    
    print(f"✅ Loaded {len(df):,} examples")
    print(f"Columns: {df.columns.tolist()}")
    
    return df


def get_dataset_info():
    """
    Get metadata about the local dataset
    
    Returns:
        dict: Dataset metadata
    """
    import json
    
    metadata_path = 'data/dataset_metadata.json'
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please run: python download_dataset.py"
        )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def is_dataset_available():
    """
    Check if the local dataset is available
    
    Returns:
        bool: True if dataset exists locally
    """
    return os.path.exists('data/agriculture_qa.parquet')


# Quick test function
def test_dataset_loading():
    """Test that the dataset loads correctly"""
    print("🧪 Testing dataset loading...")
    print("="*80)
    
    # Check availability
    if not is_dataset_available():
        print("❌ Dataset not found locally")
        print("Run: python download_dataset.py")
        return False
    
    print("✅ Dataset files found")
    
    # Load dataset
    try:
        df = load_local_dataset('parquet')
        
        # Verify structure
        assert 'question' in df.columns, "Missing 'question' column"
        assert 'answer' in df.columns, "Missing 'answer' column"
        assert len(df) > 0, "Dataset is empty"
        
        print(f"✅ Dataset structure validated")
        print(f"✅ {len(df):,} examples ready to use")
        
        # Show sample
        print("\n📝 Sample:")
        print(f"Q: {df.iloc[0]['question']}")
        print(f"A: {df.iloc[0]['answer'][:100]}...")
        
        # Show metadata
        metadata = get_dataset_info()
        print(f"\n📊 Metadata:")
        print(f"   Source: {metadata['dataset_name']}")
        print(f"   Downloaded: {metadata['download_date']}")
        print(f"   Total examples: {metadata['total_examples']:,}")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_dataset_loading()
