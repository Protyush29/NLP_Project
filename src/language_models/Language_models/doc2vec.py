import gensim
import random
import logging
import numpy as np
import pandas as pd
from shared.data_extractor import DataExtractor
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Doc2Vec:
    def __init__(self,size):
        self.train_data_x= None
        self.train_data_y = None
        self.test_data_x = None
        self.test_data_y = None
        self.model_dm = gensim.models.Doc2Vec(min_count=1, window=10,
                      size=size, sample=1e-3, negative=5, workers=3)
        self.model_dbow = gensim.models.Doc2Vec(min_count=1, window=10,
                        size=size,sample=1e-3, negative=5, dm=0, workers=3)

    def data_collection(self):
        """
        uses data extracted from the data extractor class
            return : None
        """
        extractor = DataExtractor()
        data = extractor.extract_data()
        self.train_data_x = data[0]["text"].tolist()
        self.train_data_y = data[0]["label"].tolist()
        self.test_data_x = data[1]["text"].tolist()
        self.test_data_y = data[1]["label"].tolist()


    def index_data(self,reviews, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
         We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
         a dummy index of the review.

         reviews : takes in list of paragraphs(reviews)
         label_type : str [eg- TRAIN or TEST]
         returns:
         labelized : list of LabeledSentence obj
        """
        LabeledSentence = gensim.models.doc2vec.LabeledSentence
        labelized = []
        for i, v in enumerate(reviews):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    def configure_model_vocab(self):
        self.model_dm.build_vocab(np.concatenate((self.train_data_x, self.test_data_x)))
        self.model_dbow.build_vocab(np.concatenate((self.train_data_x, self.test_data_x)))

    def train_model(self,train_data):
        for epoch in range(10):
            perm = np.random.permutation(train_data.shape[0])
            self.model_dm.train(train_data[perm])
            self.model_dbow.train(train_data[perm])

    def getVecs(model, corpus, size=400):
        vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    def train_vec(self, dataset, size=400):
        train_vecs_dm = self.getVecs(self.model_dm, dataset, size)
        train_vecs_dbow = self.getVecs(self.model_dbow, dataset, size)
        train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
        return train_vecs



if __name__ == "__main__":
    obj = Doc2Vec(size=400)
    obj.data_collection()
    obj.index_data(obj.train_data_x, 'TRAIN')
    obj.index_data(obj.test_data_x, 'TEST')
    obj.configure_model_vocab()
    obj.train_model(obj.train_data_x)
    train_vec = obj.train_vec(obj.train_data_x)

    obj.train_model(obj.test_data_x)
    test_vec = obj.train_vec(obj.test_data_x)


