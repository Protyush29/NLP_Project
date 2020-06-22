import tensorflow as tf
import pandas as pd
from tensorflow import keras
import os
import re
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(ROOT_PATH, '../shared/data'))


class ImdbExtractor:
    def __init__(self):
        self.REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    def data_cleanup(self, data):
        """
        for data cleanup
            data : List of STRING DATA TO BE CLEANED UP
            returns : List(str) of cleaned data
        """
        data = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in data]
        data = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in data]
        return data

    # Load all files from a directory in a DataFrame.
    def load_directory_data(self, directory):
        data = {}
        data["sentence"] = []
        data["sentiment"] = []
        for file_path in os.listdir(directory):
            with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
                data["sentence"].append(f.read())
                data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        data["sentence"] = self.data_cleanup(data["sentence"])
        return pd.DataFrame.from_dict(data)


    # Merge positive and negative examples, add a polarity column and shuffle.
    def load_dataset(self, directory):
        pos_df = self.load_directory_data(os.path.join(directory, "pos"))
        neg_df = self.load_directory_data(os.path.join(directory, "neg"))
        pos_df["polarity"] = 1
        neg_df["polarity"] = 0
        return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


    # Download and process the dataset files.
    def download_and_load_datasets(self,force_download=False):
        print("Existing data")
        print(os.listdir(DATA))
        user_confirmation = input("Do you want to use existing data ?\n "
                                  "Answer with 'y' for yes or 'n' for no.")

        if user_confirmation == 'n':
            filelist = [f for f in os.listdir(DATA) if f.endswith(".bak")]
            for f in filelist:
                os.remove(os.path.join(DATA, f))

            dataset = keras.utils.get_file(
                fname="aclImdb.tar.gz",
                origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                extract=True)

            train_df = self.load_dataset(os.path.join(os.path.dirname(dataset),
                                                      "aclImdb", "train"))
            test_df = self.load_dataset(os.path.join(os.path.dirname(dataset),
                                                     "aclImdb", "test"))
            train_df.to_pickle(DATA + "/train.pkl")
            test_df.to_pickle(DATA + "/test.pkl")

            return train_df, test_df

        elif user_confirmation == 'y':
            print("reading data.....")
            train = pd.read_pickle(DATA + "/train.pkl")
            test = pd.read_pickle(DATA + "/test.pkl")
            print("data ready")
            return train, test

        else:
            ip = input("Wrong input, Try again? [Y/N]")
            if ip == 'Y': return self.download_and_load_datasets()
            else : return None


if __name__ == "__main__":
    obj = ImdbExtractor()
    train , test = obj.download_and_load_datasets()
    print(train.head())