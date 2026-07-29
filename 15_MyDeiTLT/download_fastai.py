import os
import urllib.request
import tarfile

def download_and_extract():
    url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    data_dir = "./data"
    file_path = os.path.join(data_dir, "cifar10.tgz")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if not os.path.exists(os.path.join(data_dir, "cifar10", "train")):
        if not os.path.exists(file_path):
            print(f"Downloading CIFAR-10 from Fast.ai S3 bucket: {url}")
            urllib.request.urlretrieve(url, file_path)
            print("Download complete.")
        
        print("Extracting dataset...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    else:
        print("Dataset already exists and is extracted.")

if __name__ == "__main__":
    download_and_extract()
