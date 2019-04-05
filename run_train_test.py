from own_package.models import create_hparams
from own_package.train_test import run_train_test

hparams = create_hparams(lstm_units=20, hidden_layers=[20], epochs=120, batch_size=128, learning_rate=0.001)

run_train_test(model_mode='LSTM', hparams=hparams, window_size=48, loader_file='./excel/results.xlsx',
               save_model=True)

