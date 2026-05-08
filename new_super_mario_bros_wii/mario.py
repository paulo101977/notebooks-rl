
import sys
import os

sys.path.insert(0, os.path.abspath("../utils"))
sys.path.insert(0, os.path.abspath("../sdlarch-rl"))

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from typing import Tuple
from collections import deque
from sdlarch_rl import make
import numpy as np
from stable_baselines3.common.atari_wrappers import WarpFrame
from sdlarch_rl.utils.utils import TimeLimit, FrameSkip, AugmentObservation, RandomStateWrapper
from stable_baselines3.common.monitor import Monitor

myEnv = None

def make_env(env_id=0,human=False, width=None, height=None):
    def _init():
        render_mode="rgb_array"

        if human:
            render_mode = "human"

        env = make(
            "NewSuperMarioBros-Wii",
            render_mode=render_mode,
            width=width,
            height=height,
            env_id=env_id,
        )
    
        env = NewSuperMarioBrosActionWrapper(env)
        # env = AugmentObservation(env)
        env = WarpFrame(env, width=96, height=96)
        env = FrameSkip(env, skip=4)
        env = TimeLimit(env, max_steps=4000)
        env = Monitor(env)
        return env

    return _init



class NewSuperMarioBrosActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # self.action_space = gym.spaces.MultiBinary(4)
        
        self.buttons = np.zeros(16, dtype=np.int8)

        self.action_list = [
            [0, 0, 0, 0], # 0: NOOP (does nothing)

            # right
            [0, 1, 0, 0], # 1: right
            [0, 1, 0, 1], # 2: right + jump
            [0, 1, 1, 0], # 3: right + run
            [0, 1, 1, 1], # 4: right + jump + run

            # #left
            # [1, 0, 0, 0], # 3: left
            # [1, 0, 0, 1], # 4: left + jump
        ]

        self.action_space = gym.spaces.Discrete(len(self.action_list))

    def action(self, act):
        self.buttons = np.zeros(16, dtype=np.int8)

        if act.ndim > 1:
            act = act[0]
        

        act = self.action_list[act]

        self.buttons[4] = act[0]
        self.buttons[5] = act[1]

        self.buttons[9] = act[2] #run
        self.buttons[1] = act[3] #jump
            
        return self.buttons



class TemporalAttentionLSTM(BaseFeaturesExtractor):
    """
    LSTM with temporal attention
    """
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 features_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 debug: bool = False):
        super().__init__(observation_space, features_dim)
        
        self.n_frames = observation_space.shape[0]  # 1 or 4 (frames)
        self.frame_height = observation_space.shape[1]
        self.frame_width = observation_space.shape[2]
        self.debug = debug
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional= True, # False, # True,
            dropout=0.2
        )
        
        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size * 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size * 2, 1)
        )
        

        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

        self.hidden_state = None
        self.hidden_reset = True

        self.window_size = 10
        self.feature_buffer = deque(maxlen=self.window_size)
    

    def repackage_hidden(self, h):
        if isinstance(h, th.Tensor):
            return h.detach()
        else:
            # For LSTM, the state  is a tuple (h, c)
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        device = observations.device
        
        x = observations.float()

        if x.shape[-1] == 1 and x.ndim == 4:
            x = x.permute(0, 3, 1, 2)

        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        
        cnn_features = self.cnn(x)  # (batch, 512)

        if len(self.feature_buffer) > 0:
            if self.feature_buffer[0].shape[0] != batch_size:
                self.feature_buffer.clear()

        if batch_size == 1:
            self.feature_buffer.append(cnn_features.detach())
        else:
            self.feature_buffer.append(cnn_features)

            for i in range(len(self.feature_buffer) - 1):
                self.feature_buffer[i] = self.feature_buffer[i].detach()

        if len(self.feature_buffer) == 1:
            while len(self.feature_buffer) < self.window_size:
                self.feature_buffer.append(self.feature_buffer[-1])

        sequence = th.stack(list(self.feature_buffer), dim=1)
        
        sequence.requires_grad_(True)

        should_reset = (
            self.hidden_reset or 
            self.hidden_state is None or 
            self.hidden_state[0].shape[1] != batch_size
        )

        if should_reset:
            num_directions = 2
            hidden_size_total = self.lstm_num_layers * num_directions
            h0 = th.zeros(hidden_size_total, batch_size, self.lstm_hidden_size, device=device)
            c0 = th.zeros(hidden_size_total, batch_size, self.lstm_hidden_size, device=device)
            current_hidden = (h0, c0)
            self.hidden_reset = False
        else:
            current_hidden = self.repackage_hidden(self.hidden_state)

        lstm_out, last_hidden = self.lstm(sequence, current_hidden)

        self.hidden_state = last_hidden

        if self.debug:
            if batch_size > 1:
                print(f"GRAD CHECK | CNN: {self.cnn[0].weight.grad is not None} | LSTM: {self.lstm.weight_hh_l0.grad is not None}")

            if observations.shape[0] > 1:
                first_frame_mean = observations[0].mean().item()
                last_frame_mean = observations[-1].mean().item()
                print(f"DEBUG SEQUENCE | Batch initial: {first_frame_mean:.2f} | Batch final: {last_frame_mean:.2f}")
            # --- Temporary debug
            if batch_size == 1:
                h_data = self.hidden_state[0].detach()
                print(f"DEBUG LSTM | Mean: {h_data.mean().item():.4f} | Max: {h_data.max().item():.4f} | Var: {h_data.var().item():.4f}")
            else:
                h_sample = last_hidden[0].detach() 
                print(f"TRAIN LSTM | Batch Mean: {h_sample.mean().item():.4f} | Grad: {'Yes' if self.lstm.weight_hh_l0.grad is not None else 'No'}")

            if observations.shape[0] > 1:
                print(f"DEBUG Training | Window size: {sequence.shape[1]} frame(s) | Batch: {sequence.shape[0]}")
            else:
                print(f"DEBUG GAME   | Window size: {sequence.shape[1]} frame(s)")
            # -----------------------------------------

        attention_weights = th.softmax(self.attention(lstm_out), dim=1)
        context = th.sum(attention_weights * lstm_out, dim=1)
        features = self.linear(context)
        
        return features
    
    def reset_hidden(self, dones=None):
        self.hidden_state = None
        self.hidden_reset = True
        self.feature_buffer.clear()

