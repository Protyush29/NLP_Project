import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
from shared.imdb_data_extraction import ImdbExtractor


class ElmoEmbedding:
    def __init__(self):
        self.sess = tf.Session()
        K.set_session(sess)

    def __getData__(self):
        obj = ImdbExtractor()
        train_df, test_df = obj.download_and_load_datasets()