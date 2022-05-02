import gdown
import tarfile

def download_data():
    gdown.download("https://drive.google.com/uc?id=1AdMbVK110IKLG7wJKhga2N2fitV1bVPA", "drinks.tar.gz", quiet=False)
    response = tarfile.open("drinks.tar.gz", "r:gz")
    response.extractall()
    response.close()

def download_trained():
    #single file, no extraction
    #(for 60 epochs only)
    #gdown.download("https://drive.google.com/uc?id=18q5n_IUcjeOrMf6q2LIX_MvD2ee6bsZm", "drinks-trained-weights.pth", quiet=False)

    #updated version, 80 epochs
    gdown.download("https://drive.google.com/uc?id=1yF7MCj116-hc16xshxBjaJgU9zT_by8C", "drinks-trained-weights.pth", quiet=False)

