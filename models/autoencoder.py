import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tf.keras.layers


class Encoder(tfkl.Layer):
    """[summary]

    Args:
        tfkl ([type]): [description]
    """
    def __init__(
        self,
        input_dim=12,
        intermediate_dim=10,
        latent_dim=16,
        stddev=0.01,
        name="encoder",
        **kwargs
    ):
        super(Encoder, self).__init__(name=name)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.stddev = stddev
        # self.input_layer = tfkl.InputLayer(input_dim=[(self.input_dim, )])
        self.noise_layer = tf.keras.layers.GaussianNoise(self.stddev)
        self.linear_1 = tfkl.Dense(self.input_dim, kernel_initializer="he_normal")
        self.linear_2 = tfkl.Dense(self.intermediate_dim, kernel_initializer="he_normal")
        self.linear_3 = tfkl.Dense(self.latent_dim, kernel_initializer="he_normal")
    
    def call(self, input):
        # input = self.input_layer(input)
        x = self.noise_layer(input)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x 


class Decoder(tfkl.Layer):
    """[summary]

    Args:
        tfkl ([type]): [description]
    """
    def __init__(
        self,
        input_dim=16,
        intermediate_dim=10,
        output_dim=12,
        name="decoder",
        **kwargs
    ):
        super(Decoder, self).__init__(name=name)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        # self.input_layer = tfkl.InputLayer(input_dim=[(self.input_dim, )])
        self.linear_1 = tfkl.Dense(self.input_dim, kernel_initializer="he_normal")
        self.linear_2 = tfkl.Dense(self.intermediate_dim, kernel_initializer="he_normal")
        self.linear_3 = tfkl.Dense(self.output_dim, kernel_initializer="he_normal")
    
    def call(self, input):
        # input = self.input_layer(input)
        x = self.linear_1(input)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x 


class SDAE(tfk.Model):
    """[summary]

    Args:
        tfk ([type]): [description]
    """

    def __init__(
        self, 
        input_dim=12,
        intermediate_dim=10,
        latent_dim=16,
        output_dim=12,
        stddev=0.01,
        name="sdae",
        **kwargs
    ):
        super(SDAE, self).__init__(name=name)

        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.stddev = stddev
        self.encoder = Encoder(input_dim=self.input_dim, intermediate_dim=self.intermediate_dim, latent_dim=self.latent_dim, stddev=self.stddev)
        self.decoder = Decoder(input_dim=self.latent_dim, intermediate_dim=self.intermediate_dim, output_dim=self.output_dim)

    def call(self, input):
        h = self.encoder(input)
        x = self.decoder(h)
        return x 
