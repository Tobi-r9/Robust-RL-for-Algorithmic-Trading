import tensorflow as tf
tfk = tf.keras
tfkl = tf.keras.layers
tfkm = tf.keras.models

class NeuralNetwork(tfk.Model):

    def __init__(
        self, 
        hidden_nodes,
        dim, 
        dropout=None, 
        batch_norm=False,
        l2_reg=0,
        last_act_fct='linear'
        ):

        super(NeuralNetwork, self).__init__()
        self.hidden_nodes = hidden_nodes
        self.n_layers = len(hidden_nodes)
        self.dim = dim
        self.layers_list = []

        for k in range(0, self.n_layers):
            if k == 0:
                #self.layers_list.append(tfkl.Flatten(input_shape=self.dim))
                self.layers_list.append(tfkl.Dense(self.hidden_nodes[k], activation='relu', input_shape=self.dim))
                if batch_norm:
                    self.layers_list.append(tfkl.BatchNormalization())
                if dropout != None:
                    self.layers_list.append(tfkl.Dropout(dropout, name='dropout_layer'))
            elif k == self.n_layers - 2:
                self.layers_list.append(tfkl.LSTM(self.hidden_nodes[k], activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
            elif k == self.n_layers - 1:
                self.layers_list.append(tfkl.Dense(self.hidden_nodes[k], kernel_regularizer=tfk.regularizers.l2(l2_reg), activation=last_act_fct))
            else:
                self.layers_list.append(tfkl.Dense(self.hidden_nodes[k], activation='relu'))
                if batch_norm:
                    self.layers_list.append(tfkl.BatchNormalization())
                if dropout != None:
                    self.layers_list.append(tfkl.Dropout(dropout, name='dropout_layer'))

        self.model = tfkm.Sequential(self.layers_list)

    def call(self, x, training=False):
        for layer in self.layers_list:
            x = layer(x)
            if 'dropout_layer' in layer.name:
                x = layer(x, training=training)
        return x