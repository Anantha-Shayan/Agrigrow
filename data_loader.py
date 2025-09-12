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

    # Download dataset as zip into current folder
    api.dataset_download_files(dataset_slug, path=".", quiet=True)

    # Zip file is always named "<slug_last_part>.zip"
    zip_name = dataset_slug.split("/")[-1] + ".zip"

    # Read the CSV directly from the zip
    with zipfile.ZipFile(zip_name) as z:
        with z.open(filename) as f:
            df = pd.read_csv(f)

    # Clean up zip file
    os.remove(zip_name)

    return df
