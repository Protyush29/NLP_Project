import tensorflow as tf
import tensorflow_hub as hub
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
embeddings = elmo(
                ["the cat is on the mat", "dogs are in the fog"],
                signature="default",
                as_dict=True)["elmo"]
print(embeddings.shape)  #(3,128)