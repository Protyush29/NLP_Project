import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K
from keras.engine import Layer
from keras.models import Model, load_model
import keras.layers as layers
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.abspath(os.path.join(ROOT_PATH, '../Language_models/model'))
#K = None

class ElmoEmbeddingLayer(Layer):
    @staticmethod
    def set_session_backend(session):
        K = tf.compat.v1.keras.backend
        K.set_session(session)

    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        print("here")
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)
