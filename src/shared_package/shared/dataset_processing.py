import re
import os
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
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
        return extarcted_file

    def data_cleanup(self, data):
        """
        for data cleanup
            data : List = STRING DATA TO BE CLEANED UP
            returns : List(str) of cleaned data
        """
        data = [self.REPLACE_NO_SPACE.sub("", line.lower()) for line in data]
        data = [self.REPLACE_WITH_SPACE.sub(" ", line) for line in data]
        return data

    def add_label(self,data,label):
        data_frame = pd.DataFrame(data, columns=['text'])
        label_list = [label for i in range(0,len(data))]
        data_frame["label"] = label_list
        return data_frame

    def save_dataframe(self,dataframe,name):
        dataframe.to_pickle(ROOT_PATH+"/data/"+name+".pkl")
        return None

    def read_saved_data(self,name):
        return pd.read_pickle(ROOT_PATH+"/data/"+name+".pkl")

    def merge_dataframes(self,df1,df2):
        return pd.concat([df1,df2])


if __name__ == "__main__":
    TRAIN_NEG = "/home/protyush/Desktop/masters/NLP/imdb/train/neg"
    TRAIN_POS = "/home/protyush/Desktop/masters/NLP/imdb/train/neg"
    files = [os.path.join(TRAIN_NEG, file_) for file_ in os.listdir(TRAIN_NEG)]

    TEST = DataProcessing()
    data = [TEST.read_file(file_path) for file_path in files]
    cleaned_data = [ TEST.data_cleanup(element) for element in data]
    TEST.save_processed_data(cleaned_data, "train_positive",label=1)

