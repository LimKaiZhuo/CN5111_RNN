from keras.models import Model, load_model
from keras.layers import Dense, TimeDistributed, Input, LSTM, Reshape
import keras
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict


def load_model_ensemble(model_directory) -> List:
    """
    Load list of trained keras models from a .h5 saved file that can be used for prediction later
    :param model_directory: model directory where the h5 models are saved in. NOTE: All the models in the directory will
     be loaded. Hence, make sure all the models in there are part of the ensemble and no unwanted models are in the
     directory
    :return: [List: keras models]
    """

    # Loading model names into a list
    model_name_store = []
    directory = model_directory
    for idx, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".h5"):
            model_name_store.append(directory + '/' + filename)
    print('Loading the following models from {}. Total models = {}'.format(directory, len(model_name_store)))

    # Loading model class object into a list
    model_store = []
    for name in model_name_store:
        model_store.append(load_model(name))
        print('Model {} has been loaded'.format(name))

    return model_store


def model_ensemble_prediction(model_store, features_c_norm):
    """
    Run prediction given one set of feactures_c_norm input, using all the models in model store.
    :param model_store: List of keras models returned by the def load_model_ensemble
    :param features_c_norm: ndarray of shape (1, -1). The columns represents the different features.
    :return: List of metrics.
    """
    predictions_store = []
    for model in model_store:
        predictions = model.predict(features_c_norm)
        predictions_store.append(predictions)
    predictions_store = np.array(predictions_store)
    predictions_mean = np.mean(predictions_store, axis=0)

    return predictions_mean


def create_hparams(lstm_units=None, hidden_layers=None,
                   learning_rate=0.001, optimizer='Adam', epochs=100, batch_size=64,
                   activation='relu',
                   reg_l1=0, reg_l2=0,
                   verbose=1):
    """
    Creates hparam dict for input into create_DNN_model or other similar functions. Contain Hyperparameter info
    :return: hparam dict
    """
    assert type(
        hidden_layers) == list, 'hidden_layers must be input as a list. eg:' \
                                ' [10, 20] means 2 hidden layers with 10 followed by 20 nodes each'
    names = ['LSTM_units', 'hidden_layers',
             'learning_rate', 'optimizer', 'epochs', 'batch_size',
             'activation',
             'reg_l1', 'reg_l2',
             'verbose']
    values = [lstm_units, hidden_layers,
              learning_rate, optimizer, epochs, batch_size,
              activation,
              reg_l1, reg_l2,
              verbose]
    hparams = dict(zip(names, values))
    return hparams


class LSTMmodel:
    def __init__(self, fl, mode, hparams):
        """
        Initialises new DNN model based on input features_dim, labels_dim, hparams
        :param features_dim: Number of input feature nodes. Integer
        :param labels_dim: Number of output label nodes. Integer
        :param hparams: Dict containing hyperparameter information. Dict can be created using create_hparams() function.
        hparams includes:
        hidden_layers: List containing number of nodes in each hidden layer. [10, 20] means 10 then 20 nodes.
        """
        self.features_dim = fl.features_c_norm.shape[1:]
        self.window_size = fl.window_size
        self.hparams = hparams
        self.labels_dim = fl.labels.shape[-1]

        features_in = Input(shape=self.features_dim, name='input')  # Skip 1st dim

        # Selection of model
        if mode == 'LSTM_td':
            f_in = Input(shape=self.features_dim, name='LSTM_input')
            x = LSTM(units=hparams['LSTM_units'], activation='relu', return_sequences=True, name='LSTM_unit')(f_in)
            # Hidden time distributed units
            for idx, nodes in enumerate(hparams['hidden_layers']):
                x = TimeDistributed(Dense(nodes, activation='relu', name='Hidden_{}'.format(idx+1)))(x)

            # Final time distributed output
            x = TimeDistributed(Dense(self.labels_dim, activation='linear', name='Output'))(x)

            model = Model(inputs=f_in, outputs=x, name='LSTM')
            model.summary()
        elif mode == 'LSTM':
            f_in = Input(shape=self.features_dim, name='LSTM_input')
            x = LSTM(units=hparams['LSTM_units'], activation='relu', return_sequences=False, name='LSTM_unit')(f_in)
            # Hidden time distributed units
            for idx, nodes in enumerate(hparams['hidden_layers']):
                x = Dense(nodes, activation='relu', name='Hidden_{}'.format(idx+1))(x)

            # Final time distributed output
            x = Dense(self.window_size, activation='linear', name='Output')(x)
            x = Reshape(target_shape=(self.window_size, self.labels_dim))(x)

            model = Model(inputs=f_in, outputs=x, name='LSTM')
            model.summary()
        else:
            raise KeyError('Mode is not within the list of accepted modes')

        x = model(features_in)

        self.model = Model(inputs=features_in, outputs=x)
        optimizer = keras.optimizers.adam()
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        # self.model.summary()

    def train_model(self, fl,
                    save_name='mt.h5', save_dir='./save/models/',
                    save_mode=False, plot_name=None):
        # Training model
        training_features = fl.features_c_norm
        training_labels = fl.labels

        history = self.model.fit(training_features, training_labels,
                                 epochs=self.hparams['epochs'],
                                 batch_size=self.hparams['batch_size'],
                                 verbose=self.hparams['verbose'])
        # Debugging check to see features and prediction
        # pprint.pprint(training_features)
        # pprint.pprint(self.model.predict(training_features))
        # pprint.pprint(training_labels)
        # Saving Model
        if save_mode:
            self.model.save(save_dir + save_name)
        # Plotting
        if plot_name:
            # summarize history for accuracy
            plt.plot(history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
        return self.model

    def eval(self, eval_fl):
        features = eval_fl.features_c_norm
        labels = np.squeeze(eval_fl.labels[:,:,0])
        predictions = np.squeeze(self.model.predict(features)[:,:,0])
        mse = mean_squared_error(labels, predictions)
        return predictions, mse


