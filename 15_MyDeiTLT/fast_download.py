import os
import urllib.request
import tarfile

def download_cifar10_fast():
    url = "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz"
    data_dir = "./data"
    file_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    if os.path.exists(file_path):
        print(f"{file_path} already exists. Skipping download.")
        return
        
    print(f"Downloading CIFAR-10 from fast AWS mirror: {url}")
    print("This should only take a few seconds...")
    
    # Download
    urllib.request.urlretrieve(url, file_path)
    
    print("Download complete! Torchvision will handle extraction.")

if __name__ == "__main__":
    download_cifar10_fast()
