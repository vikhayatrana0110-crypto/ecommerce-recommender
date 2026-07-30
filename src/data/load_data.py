import pandas as pd

def load_reviews(file_path, max_records=100000):
    """Loads reviews from gzip compressed JSONL file using fast pandas chunking."""
    chunks = []
    # Read in chunks to avoid memory issues with multi-gigabyte files
    for chunk in pd.read_json(file_path, lines=True, compression='gzip', chunksize=20000):
        # Subset to required columns only
        chunks.append(chunk[['user_id', 'parent_asin', 'rating']])
        if sum(len(c) for c in chunks) >= max_records:
            break
    df = pd.concat(chunks, ignore_index=True).head(max_records)
    return df.rename(columns={'parent_asin': 'item_id'})

def load_metadata(file_path, filter_items=None):
    """Loads metadata, filtering dynamically by active items to save memory and avoid unknown products."""
    chunks = []
    resolved_items = set()
    filter_set = set(filter_items) if filter_items is not None else None
    
    for chunk in pd.read_json(file_path, lines=True, compression='gzip', chunksize=50000):
        cols = [c for c in ['parent_asin', 'title', 'categories'] if c in chunk.columns]
        sub_chunk = chunk[cols]
        
        if filter_set is not None:
            sub_chunk = sub_chunk[sub_chunk['parent_asin'].isin(filter_set)]
            resolved_items.update(sub_chunk['parent_asin'].unique())
            
        chunks.append(sub_chunk)
        
        if filter_set is not None and len(filter_set) <= len(resolved_items):
            break
            
    df = pd.concat(chunks, ignore_index=True)
    return df.rename(columns={'parent_asin': 'item_id'})
