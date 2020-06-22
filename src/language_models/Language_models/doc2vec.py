import gensim
import logging
import numpy as np
from shared.imdb_data_extraction import ImdbExtractor
from sklearn.linear_model import SGDClassifier
from gensim.test.utils import get_tmpfile
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
        extractor = ImdbExtractor()
        train , test = extractor.download_and_load_datasets()
        self.train_data_x = train["sentence"].tolist()
        self.test_data_x = test["sentence"].tolist()
        self.train_data_y = train["polarity"].tolist()
        self.test_data_y = test["polarity"].tolist()



    def label_data(self, reviews, label_type):
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

    def configure_model_vocab(self ,data):
        self.model_dm.build_vocab(data)
        self.model_dbow.build_vocab(data)

    def train_model(self,train_data):
        self.model_dm.train(train_data, total_examples=self.model_dm.corpus_count, epochs=3)
        self.model_dm.train(train_data, total_examples=self.model_dm.corpus_count, epochs=3)

    def getVecs(self, model, corpus, size=400):
        vecs = [np.array(model[z.tags[0]]).reshape((1, size)) for z in corpus]
        return np.concatenate(vecs)

    def train_vec(self, dataset, size=400):
        train_vecs_dm = self.getVecs(self.model_dm, dataset, size)
        train_vecs_dbow = self.getVecs(self.model_dbow, dataset, size)
        train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
        return train_vecs



if __name__ == "__main__":
    obj = Doc2Vec(size=400)
    #collect data
    obj.data_collection()

    #index data
    train_data_x = obj.label_data(reviews=obj.train_data_x, label_type='TRAIN')
    test_data_x = obj.label_data(reviews=obj.test_data_x, label_type='TEST')

    vocab = train_data_x+test_data_x
    #build vocab over all review
    obj.configure_model_vocab(vocab)

    #shuffle data to improve training accuracy
    obj.train_model(train_data_x)

    #get trained vectors
    train_vec = obj.train_vec(train_data_x)


    obj.train_model(test_data_x)
    #get testing vectors
    test_vec = obj.train_vec(test_data_x)

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vec, obj.train_data_y)
    print('Test Accuracy: %.2f' % lr.score(test_vec, obj.test_data_y))

