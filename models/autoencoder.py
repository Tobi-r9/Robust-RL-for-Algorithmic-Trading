import tensorflow as tf
import numpy as np
tfk = tf.keras
tfkl = tf.keras.layers


class Encoder(tfkl.Layer):
    """ The Encoder encodes the input to a hidden state. 
    To make sure that not only the identity mapping is learned, 
    gaussian noise is added to the input before fed to the Network
    """
    def __init__(
        self,
        input_dim,
        intermediate_dim,
        latent_dim,
        stddev,
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
        self.linear_1 = tfkl.Dense(self.input_dim, activation='relu', kernel_initializer="he_normal")
        self.linear_2 = tfkl.Dense(self.intermediate_dim, activation='relu', kernel_initializer="he_normal")
        self.linear_3 = tfkl.Dense(self.latent_dim, activation='relu', kernel_initializer="he_normal")
    
    def call(self, input):
        # input = self.input_layer(input)
        x = self.noise_layer(input)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x 


class Decoder(tfkl.Layer):
    """ The Decoder decodes the hidden state back to the 
        original dimension (output_dim). 
    """
    def __init__(
        self,
        input_dim,
        intermediate_dim,
        output_dim,
        name="decoder",
        **kwargs
    ):
        super(Decoder, self).__init__(name=name)
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        # self.input_layer = tfkl.InputLayer(input_dim=[(self.input_dim, )])
        self.linear_1 = tfkl.Dense(self.input_dim,activation='relu' ,kernel_initializer="he_normal")
        self.linear_2 = tfkl.Dense(self.intermediate_dim,activation='relu', kernel_initializer="he_normal")
        self.linear_3 = tfkl.Dense(self.output_dim, activation='linear', kernel_initializer="he_normal")
    
    def call(self, input):
        # input = self.input_layer(input)
        x = self.linear_1(input)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x 


class SDAE(tfk.Model):
    """ The Stochastic Denoising Autoencoder, finds a sparse hidden representation of the data,
    by adding gaussian noise to the input and reconstruct the original input with a standard Autoencoder.
    The hidden representation from the encoder should be a sparse denoised version of the input. 
    """

    def __init__(
        self, 
        input_dim=12,
        intermediate_dim=10,
        latent_dim=16,
        output_dim=12,
        stddev=1e-4,
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

    def get_hidden_state(self, input):
        """returns the input encoded by the trained Encoder

        Args:
            input (np.array or tf.tensor): shape = (n, input_dim)

        Returns:
            tf.tensor: shape = (n, hidden_dim)
        """
        return self.encoder(input)
