import re
import os
import pandas as pd

class DataProcessing:
    def __init__(self):
        self.REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
        self.REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    def read_file(self,path):
        """
        to read a file
            path : str = valid path to the file
            returns : list of file contents
        """
        extarcted_file = []
        for line in open(path, 'r'):
            extarcted_file.append(line.strip())
        return self.data_cleanup(extarcted_file)

    def data_cleanup(self, data):
        """
        for data cleanup
            data : List = STRING DATA TO BE CLEANED UP
            returns : List(str) of cleaned data
        """
        data = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in data]
        data = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in data]
        return data

    def save_processed_data(self,data,name,label):
        data_frame = pd.DataFrame(data, columns=['text'])
        label_list = [label for i in range(0,len(data))]
        data_frame["label"] = label_list
        data_frame.to_pickle("./"+name+".pkl")

    def read_saved_data(self,name):
        return pd.read_pickle("./"+name+".pkl")


if __name__ == "__main__":
    TRAIN_NEG = "/home/protyush/Desktop/masters/NLP/imdb/train/neg"
    TRAIN_POS = "/home/protyush/Desktop/masters/NLP/imdb/train/neg"
    files = [os.path.join(TRAIN_NEG, file_) for file_ in os.listdir(TRAIN_NEG)]

    TEST = DataProcessing()
    cleaned_data = [ TEST.read_file(file_path) for file_path in files]
    TEST.save_processed_data(cleaned_data, "train_positive",label=1)
