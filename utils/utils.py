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
        # if hasattr(self.model, 'lstm') and hasattr(self.model, 'features_extractor') and hasattr(self.model.features_extractor, 'reset_hidden'):
        #     num_layers = self.model.lstm.num_layers
        #     hidden_size = self.model.lstm.hidden_size
        #     batch_size = 1
            
        #     self.lstm_state = (
        #         th.zeros(num_layers, batch_size, hidden_size).to(self.device),
        #         th.zeros(num_layers, batch_size, hidden_size).to(self.device),
        #     )
      
        # elif hasattr(self.model, 'features_extractor') and hasattr(self.model.features_extractor, 'reset_hidden'):
        #     self.model.features_extractor.reset_hidden()
        #     self.lstm_state = None
        if hasattr(self.model, 'features_extractor') and hasattr(self.model.features_extractor, 'lstm'):
            lstm_module = self.model.features_extractor.lstm
            num_layers = lstm_module.num_layers
            hidden_size = lstm_module.hidden_size
            
            num_dirs = 2 if lstm_module.bidirectional else 1
            
            self.lstm_state = (
                th.zeros(num_layers * num_dirs, 1, hidden_size).to(self.device),
                th.zeros(num_layers * num_dirs, 1, hidden_size).to(self.device),
            )
        else:
            self.lstm_state = None
    
    def predict(self, obs, deterministic=False):
        """
        Preditct the LSTM action probabilty
        """
        if not hasattr(self.model, 'lstm'):
            return self.model.predict(obs, deterministic=deterministic)
        
        if not isinstance(obs, th.Tensor):
            obs_tensor = th.as_tensor(obs, device=self.device).float()
        else:
            obs_tensor = obs
        
        if obs_tensor.dim() == 3:  # (4, 128, 128)
            obs_tensor = obs_tensor.unsqueeze(0)  # (1, 4, 128, 128)
        
        with th.inference_mode():
            x = obs_tensor.to(self.device).half()
            if x.max() > 1.0:
                x = x / 255.0
            x = (x - 0.5) / 0.5
            
            batch_size, n_frames, h, w = x.shape
            x = x.unsqueeze(2)  # (1, 4, 1, 128, 128)
            cnn_input = x.view(-1, 1, h, w)
            cnn_features = self.model.features_extractor.cnn(cnn_input)
            sequence = cnn_features.view(batch_size, n_frames, -1)
            
            lstm_out, self.lstm_state = self.model.lstm(sequence, self.lstm_state)
            
            last_hidden = lstm_out[:, -1, :]
            
            features = self.model.features_extractor.linear(last_hidden)
            
            action_logits = self.model.action_net(features)
            
            # if deterministic:
            #     action = th.argmax(action_logits, dim=1)
            # else:
            #     probs = th.softmax(action_logits, dim=1)
            #     action = th.multinomial(probs, num_samples=1).squeeze()
            
            # action_idx = action.cpu().item()
            
            # action_binary = np.zeros(18, dtype=np.int8)
            # action_binary[action_idx] = 1
            probs = th.sigmoid(action_logits) # Sigmoid em vez de Softmax
            action_binary = (probs > 0.5).cpu().numpy().astype(np.int8).squeeze()
        
        return action_binary, None
    
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