import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
import keras.layers as layers
from Language_models.elmo import ElmoEmbeddingLayer
from shared.imdb_data_extraction import ImdbExtractor
import logging as log
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.abspath(os.path.join(ROOT_PATH, '../Language_models/model'))


class ModelElmo:
    def build_model(self):
        input_text = layers.Input(shape=(1,), dtype="string")
        #sess = tf.compat.v1.keras.backend.get_session()
        #ElmoEmbeddingLayer.set_session_backend(sess)
        embedding = ElmoEmbeddingLayer()(input_text)
        dense = layers.Dense(256, activation='relu')(embedding)
        pred = layers.Dense(1, activation='sigmoid')(dense)

        model = Model(inputs=[input_text], outputs=pred)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model


if __name__ == "__main__":
    data = ImdbExtractor()
    model_obj = ModelElmo()

    #gathering processed dataset
    train_df, test_df = data.download_and_load_datasets()

    # Creating datasets (Only take up to 150 words for memory)
    log.info("creating dataset upto 150 words for memory saving")
    train_text = train_df['sentence'].tolist()
    train_text = [' '.join(t.split()[0:150]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = train_df['polarity'].tolist()

    test_text = test_df['sentence'].tolist()
    test_text = [' '.join(t.split()[0:150]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = test_df['polarity'].tolist()
    log.info("dataset creation complete")

    # Build and fit
    log.info("Building model using keras Model")
    model = model_obj.build_model()
    log.info("Model fit in progress")
    model.fit(train_text,
              train_label,
              validation_data=(test_text, test_label),
              epochs=1,
              batch_size=32)

    log.info(f"model saved at {MODEL}/ElmoModel.h5")
    model.save(MODEL+'/ElmoModel.h5')
    pre_save_preds = model.predict(test_text[0:100])  # predictions before we clear and reload model

    # Clear and load model
    model = None
    model = model_obj.build_model()
    model.load_weights(MODEL+'/ElmoModel.h5')

    post_save_preds = model.predict(test_text[0:100])  # predictions after we clear and reload model
    all(pre_save_preds == post_save_preds)  # Are they the same?