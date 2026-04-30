from pathlib import Path
import torch as th
import numpy as np

def get_last_index(path: str, file_name: str, extension: str) -> int:
    last_index = -1

    extension = extension.lstrip(".")

    for p in Path(path).glob(f"{file_name}*.{extension}"):
        suffix = p.stem[len(file_name):]
        if suffix.isdigit():
            last_index = max(last_index, int(suffix))

    return last_index

class LSTMWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.lstm_state = None


        self.device = next(self.model.parameters()).device

        self.reset()
    
    def reset(self):
        #reset_hidden
        if hasattr(self.model, 'features_extractor') and hasattr(self.model.features_extractor, 'lstm'):
            lstm_module = self.model.features_extractor.lstm
            num_layers = lstm_module.num_layers
            hidden_size = lstm_module.hidden_size

            reset_hidden = self.model.features_extractor.reset_hidden

            num_dirs = 2 if lstm_module.bidirectional else 1
            
            self.lstm_state = (
                th.zeros(num_layers * num_dirs, 1, hidden_size).to(self.device),
                th.zeros(num_layers * num_dirs, 1, hidden_size).to(self.device),
            )

            reset_hidden()
        else:
            self.lstm_state = None
    
    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        self.model.eval()

        with th.inference_mode():
            obs = np.array(obs)

            if obs.ndim == 4 and obs.shape[0] == 1:
                obs = obs[0]

            if episode_start is not None and np.any(episode_start):
                self.reset()

            if self.lstm_state is not None:
                lstm_state = self.lstm_state
            else:
                lstm_state = None

            action, self.lstm_state = self.model.predict(
                obs,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )

            action = np.array(action)

            if action.ndim == 1:
                action = action.reshape(1, -1)

            return action, self.lstm_state

    def __call__(self, obs, deterministic=False):
        return self.predict(obs, deterministic)

def reset_lstm_state(model):
    global lstm_state
    
    if model is not None and hasattr(model, 'lstm'):
        num_layers = model.lstm.num_layers
        hidden_size = model.lstm.hidden_size
        batch_size = 1  # Single environment
        
        h = th.zeros(num_layers, batch_size, hidden_size)
        c = th.zeros(num_layers, batch_size, hidden_size)
        
        lstm_state = (h, c)
    else:
        lstm_state = None