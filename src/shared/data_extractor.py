import os
from dataset_processing import DataProcessing

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_NEG = os.path.abspath(os.path.join(ROOT_PATH, '../shared/imdb/train/neg'))
TRAIN_POS = os.path.abspath(os.path.join(ROOT_PATH, '../shared/imdb/train/pos'))
TEST_NEG = os.path.abspath(os.path.join(ROOT_PATH, '../shared/imdb/test/neg'))
TEST_POS = os.path.abspath(os.path.join(ROOT_PATH, '../shared/imdb/test/pos'))


TEST = DataProcessing()
files = [os.path.join(TRAIN_NEG, file_) for file_ in os.listdir(TRAIN_NEG)]
cleaned_data = [ TEST.read_file(file_path) for file_path in files]
TEST.save_processed_data(cleaned_data, "train_positive")