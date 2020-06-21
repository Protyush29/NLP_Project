import os
import pandas as pd
from shared.dataset_processing import DataProcessing
TEST = DataProcessing()
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(ROOT_PATH, '../shared/data'))

class DataExtractor:
    def extract_data(self):
        print("Existing data")
        print(os.listdir(DATA))
        user_confirmation = input("Do you want to use existing data ? answer with 'y' for yes or 'n' for no.")

        if user_confirmation == 'n':
            TRAIN_PATH = "/home/protyush/Desktop/masters/NLP/imdb/train/"
            TEST_PATH = "/home/protyush/Desktop/masters/NLP/imdb/test/"
            train = self.processing(TRAIN_PATH,"train")
            test = self.processing(TEST_PATH,"test")
            return [train,test]

        elif user_confirmation == 'y':
            print("reading data.....")
            train = TEST.read_saved_data("train")
            test = TEST.read_saved_data("test")
            print("data ready")
            return [train,test]

        else:
            ip = input("Wrong input, Try again? [Y/N]")
            if ip == 'Y': return self.extract_data()
            else : return None

    def processing(self,PATH,name):
        TRAIN_NEG = os.path.join(PATH, './neg')
        TRAIN_POS = os.path.join(PATH, './pos')

        print(f"preparing data from {PATH}")
        files = [os.path.join(TRAIN_NEG, file_) for file_ in os.listdir(TRAIN_NEG)]
        data = [TEST.read_file(file_path) for file_path in files]
        cleaned_data = [TEST.data_cleanup(element) for element in data]
        train_neg = TEST.add_label(cleaned_data, label=0)

        data = []
        cleaned_data = []

        files = [os.path.join(TRAIN_POS, file_) for file_ in os.listdir(TRAIN_POS)]
        data = [TEST.read_file(file_path) for file_path in files]
        cleaned_data = [TEST.data_cleanup(element) for element in data]
        train_pos = TEST.add_label(cleaned_data, label=1)

        train = TEST.merge_dataframes(train_pos, train_neg)
        TEST.save_dataframe(train, name)
        return train


if __name__ == "__main__":
    obj = DataExtractor()
    data = obj.extract_data()