from datasets import load_dataset
import pandas as pd
import yaml

def load_config(config_path="config.yaml"):
    # reads config.yaml and returns it as a python dictionary
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(config):
    # loads the dataset from huggingface using the name in config.yaml
    dataset = load_dataset(config['data']['dataset_name'])
    
    # convert each split to a pandas dataframe
    train_df = pd.DataFrame(dataset['train'])
    val_df   = pd.DataFrame(dataset['validation'])
    test_df  = pd.DataFrame(dataset['test'])
    
    return train_df, val_df, test_df
