import sys
import socket
import json
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("is cuda available", torch.cuda.is_available())
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),  # Output for 64 cells
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # Output shape: (batch_size, 64)


def modeloutput(observation, epsilon):
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
        available_actions = [idx for idx, val in enumerate(observation) if val == -1]
        if available_actions:
            action_index = random.choice(available_actions)
        else:
            action_index = random.randint(0, 63)
    else:
        action_index = action_values.argmax().item()

    # Convert index to x, y coordinates (1-based indexing)
    action1 = (action_index % 8) + 1
    action2 = (action_index // 8) + 1
    return [action1, action2]


def initialize_model(input_channels, num_actions):
    policy_net = DQN(input_channels, num_actions).to(device)
    target_net = DQN(input_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    return policy_net, target_net, optimizer, criterion


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def process_observation(observation):
    if len(observation) != 64:
        return None
    return torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)


def train_model(
    policy_net, target_net, optimizer, criterion, memory, batch_size, gamma=0.99
):
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # Combine tensors properly and move to GPU
    states = torch.cat(states).to(device)  # Shape: [batch_size, 1, 8, 8]
    actions = (
        torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    )  # Shape: [batch_size, 1]
    rewards = torch.tensor(rewards, dtype=torch.float32).to(
        device
    )  # Shape: [batch_size]
    next_states = torch.cat(next_states).to(device)  # Shape: [batch_size, 1, 8, 8]
    dones = torch.tensor(dones, dtype=torch.float32).to(device)  # Shape: [batch_size]

    # Compute Q-values for current states
    q_values = policy_net(states)  # Shape: [batch_size, 64]

    # Gather Q-values for the actions taken
    q_value = q_values.gather(1, actions).squeeze(1)  # Shape: [batch_size]

    # Compute Q-values for next states
    with torch.no_grad():
        next_q_values = target_net(next_states)  # Shape: [batch_size, 64]
        max_next_q_values = next_q_values.max(dim=1)[0]  # Shape: [batch_size]

    # Compute the expected Q values
    expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = criterion(q_value, expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def generate_and_send_action(sock, server, observation, epsilon):
    action = modeloutput(observation, epsilon)
    action_message = json.dumps({"action": action})
    sock.sendto(action_message.encode(), server)
    return {"x": action[0], "y": action[1]}


def main():
    server_address = ("localhost", 4242)

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    input_channels = 1
    num_actions = 64  # Corrected to match the action space size

    global policy_net
    policy_net, target_net, optimizer, criterion = initialize_model(
        input_channels, num_actions
    )

    memory = deque(maxlen=10000)
    batch_size = 64
    target_update = 10

    prev_state = None
    prev_action = None
    total_steps = 0
    save_interval = 10000  # Save model every 1000 steps
    print_interval = 1000  # Print progress every 100 steps

    # Epsilon parameters
    epsilon_start = 0.9  # Starting epsilon value
    epsilon_end = 0.1  # Minimum epsilon value
    epsilon_decay = 100000  # Decay rate

    epsilon = epsilon_start

    try:
        initial_message = "Hello from Ai"
        print(f"Sending: {initial_message}")
        sock.sendto(initial_message.encode(), server_address)

        while True:
            data, server = sock.recvfrom(65536)
            message = data.decode()

            if message == "close":
                print("Received close message. Terminating connection.")
                break

            try:
                data_received = json.loads(message)
                observation = data_received["observation"]
                reward = data_received["reward"]
                done = data_received.get("done", False)

                current_state = process_observation(observation)
                if current_state is None:
                    continue

                if prev_state is not None and prev_action is not None:
                    action_index = (prev_action["y"] - 1) * 8 + (prev_action["x"] - 1)
                    memory.append(
                        (prev_state, action_index, reward, current_state, done)
                    )
                    train_model(
                        policy_net, target_net, optimizer, criterion, memory, batch_size
                    )

                # Update epsilon
                epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                    -1.0 * total_steps / epsilon_decay
                )

                # Generate and send action
                prev_action = generate_and_send_action(
                    sock, server, observation, epsilon
                )
                prev_state = current_state

                total_steps += 1
                if total_steps % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Periodically print progress details
                if total_steps % print_interval == 0:
                    print(f"Step: {total_steps}, Epsilon: {epsilon:.4f}")

                # Periodically save the model
                if total_steps % save_interval == 0:
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
