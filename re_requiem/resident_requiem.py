
import gymnasium as gym
import subprocess
import win32gui
import numpy as np
import win32process
import time
import psutil
import vgamepad as vg
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from typing import Tuple
import dxcam
from collections import deque

TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS


class RERequiemEnv(gym.Env):

    def __init__(
        self, 
    ) -> None:

        self.hwnd = self.find_window_by_process_name("re9")
        print(self.hwnd)
        process = None
        self.pid = None

        if not self.hwnd:
            process = subprocess.Popen([
                r"G:\SteamLibrary\steamapps\common\RESIDENT EVIL requiem BIOHAZARD requiem\re9.exe"
            ])

            self.pid = process.pid

        self.hide_window = False

        # search process ID
        self.wait_start()

        self.gamepad = vg.VDS4Gamepad()

        self.prev_keys = set()

        self.action_space = gym.spaces.MultiBinary(18)


        self.height = 480
        self.width = 854

        self.camera = dxcam.create(output_color="RGB", max_buffer_len=1)
        self.region = self._get_window_region()
        self.camera.start(region=self.region, target_fps=60)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        self.img = None
    
    def _get_window_region(self):
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        return (left + 20, top + 100, right, bot)
    
    def render(self):
        return self.img

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self.prev_keys = set()

        observation = self._get_observation()

        # print("reset obs shape", observation.shape)
        
        return observation, {}

    def step(self, actions: np.ndarray):
        frame_start = time.perf_counter()
        current = set()

        x_value = 0
        y_value = 0
        right_x = 0
        right_y = 0

        if not len(actions) > 1:
           actions = actions[0]

        # dpad
        if actions[0] > 0:
            current.add(-2)
            y_value = -1.0

        if actions[1]:
            current.add(-2)
            y_value = 1.0
  
        if actions[2]:
            current.add(-2)
            x_value = -1.0

        if  actions[3]:
            current.add(-2)
            x_value = 1.0
  

        # buttons
        if actions[4]:
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_CROSS)

        if actions[5]:
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)

        if actions[6]:
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_SQUARE)


        if abs(actions[7]) > 0:
            current.add(-4)
        
        if abs(actions[8]) > 0:
            current.add(-5)

        if actions[9]:
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT)

        # convert analog to discrete action
        if np.any(actions[10:14] == 1):
            current.add(-3)
            if actions[10]:
                right_x = 0.5
            if actions[11]:
                right_x = 1
            if actions[12]:
                right_x = -.5
            if actions[13]:
                right_x = -1

        # convert analog to discrete action
        if np.any(actions[14:18] == 1):
            current.add(-3)
            # right_y = -actions[8]
            if actions[14]:
                right_y = -0.5
            if actions[15]:
                right_y = -1
            if actions[16]:
                right_y = 0.5
            if actions[17]:
                right_y = 1

        for b in current - self.prev_keys:
            if b > 0:
                self.gamepad.press_button(b)

        if -2 in current:
            self.gamepad.left_joystick_float(x_value_float=x_value, y_value_float=y_value)

        if -3 in current:
            self.gamepad.right_joystick_float(x_value_float=right_x, y_value_float=right_y)

        if -4 in current:
            self.gamepad.left_trigger(value=255)

        if -5 in current:
            self.gamepad.right_trigger(value=255)


        for b in self.prev_keys - current:
            if b > 0:
                self.gamepad.release_button(b)
            elif b == -2:
                self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
            elif b == -3:
                self.gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)
            elif b == -4:
                self.gamepad.left_trigger(value=0)
            elif b == -5:
                self.gamepad.right_trigger(value=0)

        self.gamepad.update()
        self.prev_keys = current.copy()


        observation = self._get_observation()

        # elapsed = time.perf_counter() - frame_start
        # sleep_time = FRAME_TIME - elapsed
        # if sleep_time > 0:
        #     time.sleep(sleep_time)

        return observation, 0.0, False, False, {}

    # fastest _get_observation
    def _get_observation(self):
        # frame = self.camera.grab(region=self.region)
        frame = self.camera.get_latest_frame()
        
        if frame is None:
            return self.img if self.img is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.img = frame
        
        return cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

    def find_window_by_process_name(self, process_name):
        def callback(hwnd, result):
            if win32gui.IsWindowVisible(hwnd):
                tid, win_pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(win_pid)
                    if process_name.lower() in process.name().lower():
                        result.append(hwnd)
                except:
                    pass
            return True

        result = []
        win32gui.EnumWindows(callback, result)
        return result[0] if result else None

    def wait_start(self):
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            self.hwnd = self.find_window_by_process_name("re9")
            if self.hwnd:
                print(f"Window founded HWND: {self.hwnd}.")
                
                # <<< Hide the emulator window >>>
                if self.hide_window:
                    print("Hide the window...")
                    # SW_HIDE = 0. Hide the window.
                    # win32gui.ShowWindow(hwnd, win32con.SW_HIDE)

                    rect = win32gui.GetWindowRect(self.hwnd)

                    x, y, w, h = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
                    win32gui.MoveWindow(self.hwnd, -w, -h, w, h, True)
                break
            print("next search...")
            time.sleep(1)


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = th.sigmoid(self.conv(x))
        return x * attention


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
        
        self.n_frames = observation_space.shape[0]  # 4
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
    
    # def forward(self, observations: th.Tensor) -> th.Tensor:
    #     batch_size = observations.shape[0]
        
    #     x = observations.float()
    #     if x.max() > 1.0:
    #         x = x / 255.0
    #     x = (x - 0.5) / 0.5
        
    #     x = x.unsqueeze(2)
    #     cnn_input = x.view(-1, 1, self.frame_height, self.frame_width)
    #     cnn_features = self.cnn(cnn_input)
    #     sequence = cnn_features.view(batch_size, self.n_frames, -1)
        
    #     # LSTM
    #     lstm_out, _ = self.lstm(sequence)  # (batch, 4, hidden_size*2)
        
    #     attention_weights = th.softmax(self.attention(lstm_out), dim=1)
    #     context = th.sum(attention_weights * lstm_out, dim=1)
        
    #     features = self.linear(context)
        
    #     return features

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
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        
        cnn_features = self.cnn(x)  # (batch, 512)

        if batch_size == 1:
            self.feature_buffer.append(cnn_features.detach())
        else:
            self.feature_buffer.append(cnn_features)

            for i in range(len(self.feature_buffer) - 1):
                self.feature_buffer[i] = self.feature_buffer[i].detach()
        # self.feature_buffer.append(cnn_features.detach())

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
        # if batch_size == 1:
        #     self.hidden_state = (last_hidden[0].detach(), last_hidden[1].detach())
        #     self.hidden_reset = False
        # else:
        #     self.hidden_state = None

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

