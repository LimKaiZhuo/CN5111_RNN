import matplotlib.pyplot as plt
import numpy as np
from own_package.models import load_model_ensemble, model_ensemble_prediction


def test(control_action):
    model_store = load_model_ensemble('./results/LSTM17/models')

    if len(control_action.shape) != 3:
        # Control action should be for 1 day only
        control_action = control_action.reshape(-1,48,1)

    predictions = np.squeeze(model_ensemble_prediction(model_store, control_action))
    plt.plot(predictions[0,:], label='1')
    plt.plot(predictions[1, :], label='2')
    plt.legend()
    plt.show()

