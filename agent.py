import torch
import torch.nn as nn
import torch.optim as optim
import os
import config
import numpy as np

class AgentModel(nn.Module):
    def __init__(self):
        super(AgentModel, self).__init__()
        # CNN body
        self.conv_layers = nn.Sequential(
            nn.Conv2d(config.INPUT_CHANNELS, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Action Choice Head
        self.action_head = nn.Sequential(
            nn.Linear(self._get_conv_output_size(), 512),
            nn.ReLU(),
            nn.Linear(512, len(config.ACTION_SPACE)),
            nn.LogSoftmax(dim=1)
        )

        # Action Parameters Head
        self.params_head = nn.Sequential(
            nn.Linear(self._get_conv_output_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # For drag: start_x, start_y, end_x, end_y
        )

    def _get_conv_output_size(self):
        o = self.conv_layers(torch.zeros(1, config.INPUT_CHANNELS, *config.TARGET_RESOLUTION))
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.action_head(x), self.params_head(x)

class Agent:
    def __init__(self, device):
        self.device = device
        self.model = AgentModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.memory = None # To be assigned later

    def get_action(self, state, epsilon=0.1):
        if torch.rand(1) < epsilon:
            # Explore: choose a random action
            action_type = torch.tensor([[random.choice(config.ACTION_SPACE)]], device=self.device, dtype=torch.long)
            # For random action, parameters can be random within the frame
            params = torch.rand(1, 4, device=self.device)
        else:
            # Exploit: choose the best action from the model
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs, params = self.model(state_tensor)
                action_type = action_probs.max(1)[1].view(1, 1)

        return action_type, params

    def train(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        transitions = self.memory.sample(config.BATCH_SIZE)
        batch = list(zip(*transitions))

        states = torch.FloatTensor(batch[0]).to(self.device)

        action_types = torch.cat([a[0] for a in batch[1]]).to(self.device)
        action_params = torch.cat([a[1] for a in batch[1]]).to(self.device)

        rewards = torch.FloatTensor(batch[2]).to(self.device)

        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch[3])), device=self.device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch[3] if s is not None]).to(self.device)

        self.optimizer.zero_grad()

        # Get current Q values
        current_action_probs, current_action_params = self.model(states)

        # Calculate loss for action type
        action_loss = nn.NLLLoss()(current_action_probs, action_types.squeeze(1))

        # Calculate loss for action parameters
        param_loss = nn.MSELoss()(current_action_params, action_params)

        # Total loss
        total_loss = action_loss + param_loss

        # Optimize the model
        total_loss.backward()
        self.optimizer.step()

    def save_model(self, path=config.MODEL_PATH):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path=config.MODEL_PATH):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, weights_only=True))
            self.model.eval()
