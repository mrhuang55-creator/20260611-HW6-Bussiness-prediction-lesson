import os
import requests

def download_dataset():
    url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv"
    data_dir = "data"
    file_path = os.path.join(data_dir, "50_Startups.csv")

    # Ensure the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Successfully saved dataset to: {file_path}")
        
        # Verify file size and simple content check
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"Downloaded file size: {file_size} bytes")
            
    except Exception as e:
        print(f"Failed to download the dataset. Error: {e}")

if __name__ == "__main__":
    download_dataset()
