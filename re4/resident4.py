
import gymnasium as gym
import subprocess
import win32gui
import win32ui
from ctypes import windll
import numpy as np
import win32process
import time
import psutil
import vgamepad as vg
import cv2
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS


class RE4Env(gym.Env):

    def __init__(
        self, 
    ) -> None:

        self.hwnd = self.find_window_by_process_name("re4")
        print(self.hwnd)
        process = None
        self.pid = None

        if not self.hwnd:
            process = subprocess.Popen([
                r"F:\SteamLibrary\steamapps\common\RESIDENT EVIL 4  BIOHAZARD RE4\re4.exe"
            ])

            self.pid = process.pid

        self.hide_window = False

        # search process ID
        self.wait_start()


        self.gamepad = vg.VDS4Gamepad()

        self.prev_keys = set()

        self.action_space = gym.spaces.MultiBinary(18)


        self.height = 128
        self.width = 228

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8,
        )

        self.img = None

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
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_CROSS) # light kick

        if actions[5]:
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_CIRCLE)

        if actions[6]: # strong kick
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_SQUARE) # heavy attack


        if abs(actions[7]) > 0:
            current.add(-4)
        
        if abs(actions[8]) > 0:
            current.add(-5)

        if actions[9]:
            current.add(vg.DS4_BUTTONS.DS4_BUTTON_THUMB_LEFT)

        # TODO: convert to discrete action
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

        elapsed = time.perf_counter() - frame_start
        sleep_time = FRAME_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        return observation, 0.0, False, False, {}
    
    def _get_observation(self):
        if self.hwnd == 0:
            raise ValueError("HWND of window not founded")

        left, top, right, bot = win32gui.GetClientRect(self.hwnd)
        width =  right - left
        height = bot - top + 70

        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)


        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)
        
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8').reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
        
        img_rgb = img[100:, 20:, [2, 1, 0]]

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        self.img = img_rgb

        img_rgb = cv2.resize(img_rgb, (self.width, self.height))

        return img_rgb

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
            self.hwnd = self.find_window_by_process_name("re4")
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


# class Re4CNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
#         super().__init__(observation_space, features_dim)
#         n_input_channels = observation_space.shape[0]  # stacked frames (4)
        

#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         with th.no_grad():
#             sample = th.zeros(1, n_input_channels, 128, 128)
#             n_flatten = self.cnn(sample).shape[1]
        
#         self.linear = nn.Sequential(
#             nn.Linear(n_flatten, features_dim),
#             nn.ReLU(),
#             nn.Linear(features_dim, features_dim),
#             nn.ReLU(),
#         )
    
#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         observations = observations.float()
        
#         if observations.max() > 1.0:
#             observations = observations / 255.0
        
#         return self.linear(self.cnn(observations))

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = th.sigmoid(self.conv(x))
        return x * attention

class Re4CNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # stacked frames (4)
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
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
            SpatialAttention(512),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, n_input_channels, 128, 128)
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, features_dim),
            nn.Tanh() if features_dim <= 10 else nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.float()
        
        if observations.max() > 1.0:
            observations = observations / 255.0
        
        observations = (observations - 0.5) / 0.5
        
        return self.linear(self.cnn(observations))

class TemporalAttentionLSTM(BaseFeaturesExtractor):
    """
    LSTM with temporal attention
    """
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 features_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2):
        super().__init__(observation_space, features_dim)
        
        self.n_frames = observation_space.shape[0]  # 4
        self.frame_height = observation_space.shape[1]
        self.frame_width = observation_space.shape[2]
        
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
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
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
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        
        x = x.unsqueeze(2)
        cnn_input = x.view(-1, 1, self.frame_height, self.frame_width)
        cnn_features = self.cnn(cnn_input)
        sequence = cnn_features.view(batch_size, self.n_frames, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(sequence)  # (batch, 4, hidden_size*2)
        
        attention_weights = th.softmax(self.attention(lstm_out), dim=1)
        context = th.sum(attention_weights * lstm_out, dim=1)
        
        features = self.linear(context)
        
        return features


class TransferLearningLSTM(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 features_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 freeze_backbone_init: bool = False):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 4 frames stacked
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.n_frames = n_input_channels
        self.freeze_backbone_init = freeze_backbone_init
        
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False  # 1 channel
        )
        
        with th.no_grad():
            self.backbone.conv1.weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        if freeze_backbone_init:
            self.freeze_layers(until_layer=6)
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        self.cnn_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.2 if lstm_num_layers > 1 else 0
        )
        
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
        self.lstm_state = None
        self.batch_counter = 0
    
    def freeze_layers(self, until_layer=6):
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers[:until_layer]):
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def init_lstm_state(self, batch_size=1, device='cpu'):
        self.lstm_state = (
            th.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device),
            th.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        
        # x: (batch, 4, 128, 128)
        frames = []
        for t in range(self.n_frames):
            frame = x[:, t:t+1, :, :]  # (batch, 1, 128, 128)
            frames.append(frame)
        
        cnn_features = []
        for frame in frames:
            feat = self.backbone(frame)  # (batch, 512, 1, 1)
            feat = feat.view(-1, 512)   # (batch, 512)
            feat = self.cnn_projection(feat)  # (batch, 512)
            cnn_features.append(feat)
        
        sequence = th.stack(cnn_features, dim=1)
        
        lstm_out, self.lstm_state = self.lstm(sequence, self.lstm_state)

        self.batch_counter += 1
        if self.batch_counter % 1 == 0:
            if self.lstm_state is not None:
                self.lstm_state = (
                    self.lstm_state[0].detach(),
                    self.lstm_state[1].detach()
                )
        
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        features = self.head(last_hidden)
        
        return features


class TransferLearningEfficientNetLSTM(BaseFeaturesExtractor):
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 features_dim: int = 512,
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 freeze_backbone_init: bool = False):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0]  # 4 frames stacked
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.n_frames = n_input_channels
        
        # ========== EfficientNet-B3 ==========
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, 40, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        with th.no_grad():
            self.backbone.features[0][0].weight = nn.Parameter(
                original_conv.weight.mean(dim=1, keepdim=True)
            )
        
        # Remove classifier
        self.backbone.classifier = nn.Identity()
        
        # EfficientNet-B0 has 1280 features
        self.cnn_projection = nn.Sequential(
            nn.Linear(1280, 1024),  # 1280 → 1024
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),    # 1024 → 512
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # ========== LSTM ==========
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.2 if lstm_num_layers > 1 else 0
        )
        
        # ========== final head ==========
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
        self._features_dim = features_dim
        self.lstm_state = None
        self.batch_counter = 0
    
    def init_lstm_state(self, batch_size=1, device='cpu'):
        self.lstm_state = (
            th.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device),
            th.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        x = observations.float()
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5
        
        frames = []
        for t in range(self.n_frames):
            frame = x[:, t:t+1, :, :]  # (batch, 1, 128, 128)
            frames.append(frame)
        
        cnn_features = []
        for frame in frames:
            feat = self.backbone(frame)  # (batch, 1280)
            feat = self.cnn_projection(feat)  # (batch, 512)
            cnn_features.append(feat)
        
        sequence = th.stack(cnn_features, dim=1)  # (batch, 4, 512)
        
        # LSTM
        if self.lstm_state is None:
            self.init_lstm_state(batch_size, observations.device)
        
        lstm_out, self.lstm_state = self.lstm(sequence, self.lstm_state)
        
        self.batch_counter += 1
        if self.batch_counter % 1 == 0:
            if self.lstm_state is not None:
                self.lstm_state = (
                    self.lstm_state[0].detach(),
                    self.lstm_state[1].detach()
                )
        
        # last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        
        # final head
        features = self.head(last_hidden)
        
        return features