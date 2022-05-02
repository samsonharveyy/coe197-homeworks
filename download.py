import gdown
import tarfile

def download_data():
    gdown.download("https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA", "drinks.tar.gz", quiet=False)
    response = tarfile.open("drinks.tar.gz", "r:gz")
    response.extractall()
    response.close()

def download_trained():
    #single file, no extraction
    #also in model_files folder in this repo
    gdown.download("https://drive.google.com/uc?id=18q5n_IUcjeOrMf6q2LIX_MvD2ee6bsZm", "drinks-trained-weights.pth", quiet=False)


