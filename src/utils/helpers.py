import yaml
import pickle

def load_config(config_path="src/config/config.yaml"):
    """Loads configuration parameters from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_pickle(obj, file_path):
    """Saves a python object as a pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(file_path):
    """Loads a python object from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)
