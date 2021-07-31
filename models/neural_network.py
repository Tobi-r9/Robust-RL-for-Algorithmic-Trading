import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkm = tf.keras.models

class NeuralNetwork(tfk.Model):

    def __init__(
        self, 
        hidden_nodes,
        dim, 
        n_actions,
        n_lstm_cells=128,
        dropout=None, 
        batch_norm=False,
        l2_reg=0,
        batch_size=32,
        last_act_fct='linear'
        ):

        super(NeuralNetwork, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.n_layers = len(hidden_nodes)
        self.batch_size = batch_size
        self.num_actions = n_actions
        self.n_lstm_cells = n_lstm_cells
        self.dim = dim
        self.dense_layers = []
        self.input_layer = tfkl.InputLayer(
            input_shape=self.dim, 
            batch_size=self.batch_size
            )
        for k in range(0, self.n_layers):
            self.dense_layers.append(
                tfkl.Dense(
                self.hidden_nodes[k], 
                activation='relu')
                )
        self.lstm_layer = tfkl.LSTM(
            self.n_lstm_cells, 
            activation='tanh', 
            recurrent_activation='sigmoid',
            return_sequences=False
            )
        self.policy_layer = tfkl.Dense(
            self.n_actions, 
            kernel_regularizer=tfk.regularizers.l2(l2_reg), 
            activation='softmax'
            )
        self.value_layer = tfkl.Dense(
            1, 
            kernel_regularizer=tfk.regularizers.l2(l2_reg), 
            activation='linear'
            )

    def call(self, input):
        x = self.input_layer(input)
        for layer in self.layers_list:
            x = layer(x)
        x = self.lstm_layer(x)
        action = self.policy_layer(x)
        value = self.value_layer(x)
        return action, value