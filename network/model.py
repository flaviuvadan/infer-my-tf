""" Holds the model """

import tensorflow as tf


class Model:
    """ Model class represents the inference model """

    def __init__(self, learning_rate=0.0001, training_epochs=30):
        """ Init function """
        self.model = tf.keras.models.Sequential()  # fix this ugly import

        # input shape is 100 encoded nucleotides
        self.model.add(tf.layers.Dense(64, activation=tf.nn.sigmoid, input_shape=(100,)))
        # now performs automatic input shape inference
        self.model.add(tf.layers.Dense(64, activation=tf.nn.sigmoid))
        self.model.add(tf.layers.Dense(64, activation=tf.nn.sigmoid))

        self.optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(self.optimizer, 'categorical_crossentropy', metrics=['accuracy'])

        self.training_epochs = training_epochs

    def train(self, x, y):
        """
        Performs training with the given x, y training pairs (x, y should be zippable, otherwise Python3 will omit
        unpaired values)
        :param x: [training objects]
        :param y: [training labels of objects in x]
        """
        self.model.fit(x, y, epochs=self.training_epochs)

    def test(self, x, y):
        """
        Reports testing of the model (x, y should be zippable, otherwise Python3 will omit unpaired values)
        :param x: [testing objects]
        :param y: [testing labels of objects in x]
        :return: loss, accuracy
        """
        return self.model.evaluate(x, y)

    def predict(self, x):
        """
        Perform a single prediction of a transcription factor
        :param x: [collection of nucleotides]
        :return: predicted transcription factor binding site
        """
        return self.model.predict(x)  # this has to be further adjusted
