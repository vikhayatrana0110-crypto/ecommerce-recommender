import json
import gzip
from tqdm import tqdm


def load_reviews(file_path, max_records=1000000):
    """
    Stream large JSONL (.gz) review file safely.
    Loads only max_records to avoid RAM overload.
    """

    reviews = []

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if i >= max_records:
                break

            record = json.loads(line)

            reviews.append({
                "user_id": record.get("user_id"),
                "item_id": record.get("parent_asin"),
                "rating": record.get("rating")
            })

    import pandas as pd
    return pd.DataFrame(reviews)



def load_metadata(file_path, max_records=1000000):
    """
    Stream large metadata (.gz) file safely.
    """

    metadata = []

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f)):
            if i >= max_records:
                break

            record = json.loads(line)

            metadata.append({
                "item_id": record.get("asin"),
                "title": record.get("title"),
                "category": record.get("category")
            })

    import pandas as pd
    return pd.DataFrame(metadata)

