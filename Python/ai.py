import sys
import socket
import json
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters and configuration
INPUT_CHANNELS = 1
NUM_ACTIONS = 64
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 100000
TARGET_UPDATE = 1000
SAVE_INTERVAL = 10000
PRINT_INTERVAL = 1000
EPSILON_START = 0.9
EPSILON_END = 0.01
EPSILON_DECAY = 0.99999
NEGATIVE_REWARD_TRAINING = 0.8
SERVER_ADDRESS = ("localhost", 4242)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Directions to check for neighboring cells
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class ConvDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ConvDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def get_valid_actions(observation):
    valid_actions = []
    for idx, val in enumerate(observation):
        if val == -1:  # Covered cell
            x, y = idx // 8, idx % 8
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    neighbor_idx = nx * 8 + ny
                    if observation[neighbor_idx] != -1:  # Uncovered neighbor
                        valid_actions.append(idx)
                        break
    return valid_actions


def model_output(observation, epsilon):
    state = torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)
    with torch.no_grad():
        action_values = policy_net(state)  # Shape: (1, 64)

    # Penalize actions for revealed cells
    action_values = action_values.view(-1)
    for idx, val in enumerate(observation):
        if val != -1:
            action_values[idx] -= 1e6  # Penalize the specific cell

    # Epsilon-greedy action selection
    if random.random() < epsilon:
        valid_actions = get_valid_actions(observation)
        if valid_actions:
            action_index = random.choice(valid_actions)
        else:
            action_index = random.randint(0, 63)
    else:
        action_index = action_values.argmax().item()

    # Convert index to x, y coordinates (1-based indexing)
    action1 = (action_index // 8) + 1
    action2 = (action_index % 8) + 1
    return [action1, action2]


def initialize_model(input_channels, num_actions):
    policy_net = ConvDQN(input_channels, num_actions).to(device)
    target_net = ConvDQN(input_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    return policy_net, target_net, optimizer, criterion


def save_model(model, filename):
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), filename)
    model.to(device)


def process_observation(observation):
    if len(observation) != 64:
        return None
    return torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)


def train_model(
    policy_net, target_net, optimizer, criterion, memory, batch_size, gamma=GAMMA
):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states)  # Shape: [batch_size, 64]
    q_value = q_values.gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q_values = next_q_values.max(dim=1)[0]

    expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))
    loss = criterion(q_value, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_on_data(data, policy_net, target_net, optimizer, criterion, device):
    prev_observation = data["prev_observation"]
    new_observation = data["new_observation"]
    action = data["action"]
    reward = data["reward"]

    if reward > 0 or (reward <= 0 and random.random() < NEGATIVE_REWARD_TRAINING):
        prev_state = (
            torch.tensor(prev_observation, dtype=torch.float32).unsqueeze(0).to(device)
        )
        new_state = (
            torch.tensor(new_observation, dtype=torch.float32).unsqueeze(0).to(device)
        )
        action_index = action["x"] * 8 + action["y"]
        action_tensor = torch.tensor([[action_index]], dtype=torch.long).to(device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

        policy_net.train()
        state_action_values = policy_net(prev_state).gather(1, action_tensor)

        with torch.no_grad():
            target_net.eval()
            next_state_values = target_net(new_state).max(1)[0].unsqueeze(1)
            expected_state_action_values = (next_state_values * GAMMA) + reward_tensor

        loss = criterion(state_action_values, expected_state_action_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def generate_and_send_action(sock, server, observation, epsilon):
    action = model_output(observation, epsilon)
    action_message = json.dumps({"action": action})
    sock.sendto(action_message.encode(), server)
    return {"x": action[0], "y": action[1]}


def main():
    global policy_net
    policy_net, target_net, optimizer, criterion = initialize_model(
        INPUT_CHANNELS, NUM_ACTIONS
    )
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    total_steps = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        initial_message = "Hello from Ai"
        print(f"Sending: {initial_message}")
        sock.sendto(initial_message.encode(), SERVER_ADDRESS)

        while True:
            try:
                data, server = sock.recvfrom(65536)
                message = data.decode()

                if message == "close":
                    print("Received close message. Terminating connection.")
                    break

                data_received = json.loads(message)
                prev_observation = data_received["prev_observation"]
                new_observation = data_received["new_observation"]
                reward = data_received["reward"]
                action = data_received["action"]
                done = data_received.get("done", False)

                prev_state = process_observation(prev_observation)
                current_state = process_observation(new_observation)

                if prev_state is None or current_state is None:
                    continue

                x = action["x"] - 1
                y = action["y"] - 1
                action_index = x * 8 + y
                if 0 <= action_index < 64:
                    memory.append(
                        (prev_state, action_index, reward, current_state, done)
                    )

                train_model(
                    policy_net, target_net, optimizer, criterion, memory, BATCH_SIZE
                )

                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

                next_action = model_output(new_observation, epsilon)
                action_index = next_action[0] * 8 + next_action[1] - 9
                x = action_index // 8 + 1
                y = action_index % 8 + 1
                action_to_send = {"x": x, "y": y}
                action_message = json.dumps({"action": action_to_send})
                sock.sendto(action_message.encode(), server)

                total_steps += 1

                if total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if total_steps % PRINT_INTERVAL == 0:
                    print(f"Step: {total_steps}, Epsilon: {epsilon:.4f}")

                if total_steps % SAVE_INTERVAL == 0:
                    save_model(policy_net, f"policy_net_{total_steps}.pth")
                    print(f"Model saved at step {total_steps}")

            except json.JSONDecodeError as e:
                print("Error decoding JSON from Godot:", e)

    finally:
        print("Closing socket")
        save_model(policy_net, "policy_net_final.pth")
        print("Final model saved")
        sock.close()


if __name__ == "__main__":
    main()
