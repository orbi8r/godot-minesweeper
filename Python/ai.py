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
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
TARGET_UPDATE = 1000
SAVE_INTERVAL = 10000
PRINT_INTERVAL = 1000
EPSILON_START = 0.9
EPSILON_END = 0.001
EPSILON_DECAY = 0.99999
NEGATIVE_REWARD_TRAINING = 1
SERVER_ADDRESS = ("localhost", 4242)

DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


class ConvDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(ConvDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def initialize_model(input_channels, num_actions):
    policy_net = ConvDQN(input_channels, num_actions).to(device)
    target_net = ConvDQN(input_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    return policy_net, target_net, optimizer, criterion, scheduler


def process_observation(observation):
    if len(observation) != 64:
        return None
    return torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)


def save_model(model, filename):
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), filename)
    model.to(device)


def train_model(policy_net, target_net, memory, optimizer, criterion, batch_size):
    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states)
    actions = torch.tensor(actions).view(-1, 1).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.cat(next_states)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states)
    if q_values is None:
        return None

    next_q_values = target_net(next_states)
    if next_q_values is None:
        return None

    # Check if actions are within the valid range
    if actions.max() >= q_values.size(1) or actions.min() < 0:
        return None

    q_value = q_values.gather(1, actions).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + (GAMMA * next_q_value * (1 - dones))

    loss = criterion(q_value, expected_q_value.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def model_output(observation, epsilon):
    state = torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)
    with torch.no_grad():
        action_values = policy_net(state)  # Shape: (1, 64)

    # Penalize already covered cells
    # for idx, val in enumerate(observation):
    #   if val != -1:
    #        action_values[0, idx] -= 1e6

    valid_actions = get_valid_actions(observation)
    # Epsilon-greedy action selection
    if random.random() < epsilon:
        if valid_actions:
            return random.choice(valid_actions)
        else:
            return random.randint(0, 63)
    else:
        return action_values.argmax().item()


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


def main():
    global policy_net
    policy_net, target_net, optimizer, criterion, scheduler = initialize_model(
        INPUT_CHANNELS, NUM_ACTIONS
    )
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    total_steps = 0
    last_data = None  # Variable to store the last data
    total_reward = 0  # Variable to track total reward
    latest_train_model_loss = 0  # Variable to store the latest loss from train_model
    action_message = None  # Variable to store the action message

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

                if data_received == last_data:
                    sock.sendto(action_message.encode(), server)
                    continue
                last_data = data_received

                prev_observation = list(map(int, data_received["prev_observation"]))
                new_observation = list(map(int, data_received["new_observation"]))
                reward = int(data_received["reward"])
                action = {k: int(v) for k, v in data_received["action"].items()}
                done = bool(data_received.get("done", False))

                if len(prev_observation) != 64 or len(new_observation) != 64:
                    continue

                prev_state = process_observation(prev_observation)
                current_state = process_observation(new_observation)

                if prev_state is None or current_state is None:
                    continue

                x = action["x"] - 1
                y = action["y"] - 1
                action_index = x * 8 + y

                if reward > 0 or (
                    reward <= 0 and random.random() < NEGATIVE_REWARD_TRAINING
                ):
                    memory.append(
                        (prev_state, action_index, reward, current_state, done)
                    )

                if total_steps % BATCH_SIZE == 0:
                    train_model_loss = train_model(
                        policy_net, target_net, memory, optimizer, criterion, BATCH_SIZE
                    )
                    if train_model_loss is not None:
                        latest_train_model_loss = train_model_loss

                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

                next_action = model_output(new_observation, epsilon)
                x = next_action // 8 + 1
                y = next_action % 8 + 1
                action_to_send = {"x": x, "y": y}
                action_message = json.dumps(
                    {"action": action_to_send, "observation": new_observation}
                )
                sock.sendto(action_message.encode(), server)

                total_steps += 1
                total_reward += reward

                if total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if total_steps % PRINT_INTERVAL == 0:
                    avg_reward = total_reward / PRINT_INTERVAL
                    print(
                        f"Step: {total_steps}, Epsilon: {epsilon:.4f}, Avg Reward: {avg_reward:.4f}, Train model loss: {latest_train_model_loss:.8f}"
                    )
                    total_reward = 0

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
