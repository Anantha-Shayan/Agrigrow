from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import zipfile, os

def load_kaggle_csv(dataset_slug, filename):
    """
    Download a Kaggle dataset and return a pandas DataFrame.
    The downloaded zip is deleted after extraction.
    """
    api = KaggleApi()
    api.authenticate()    
    api.dataset_download_files(dataset_slug, path=".", quiet=True)
    
    zip_name = dataset_slug.split("/")[-1] + ".zip"
    
    with zipfile.ZipFile(zip_name) as z:
        with z.open(filename) as f:
            df = pd.read_csv(f)

    os.remove(zip_name)

    return df
