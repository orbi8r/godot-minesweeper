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
            nn.Linear(512, num_actions * 8),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(-1, 2, 8)


def modeloutput(observation, epsilon):
    state = torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)
    with torch.no_grad():
        action_values = policy_net(state)

    # Penalize actions corresponding to non `-1` values in the observation
    for i in range(8):
        for j in range(8):
            index = i * 8 + j
            if observation[index] != -1:
                action_values[0, 0, j] -= 1e6  # Penalize x-coordinate
                action_values[0, 1, i] -= 1e6  # Penalize y-coordinate

    # Epsilon-greedy strategy with decaying epsilon
    if random.random() < epsilon:
        # Select random actions from available positions
        available_coords = [
            (j + 1, i + 1)
            for i in range(8)
            for j in range(8)
            if observation[i * 8 + j] == -1
        ]
        if available_coords:
            action1, action2 = random.choice(available_coords)
            print(f"Random action selected: ({action1}, {action2})")
        else:
            action1, action2 = random.randint(1, 8), random.randint(1, 8)
            print(
                f"No available moves, selecting random action: ({action1}, {action2})"
            )
    else:
        # Select the action with the highest value
        action1 = action_values[0, 0].argmax().item() + 1
        action2 = action_values[0, 1].argmax().item() + 1
        print(f"Greedy action selected: ({action1}, {action2})")
    return [action1, action2]


def initialize_model(input_channels, num_actions):
    policy_net = DQN(input_channels, num_actions).to(device)
    target_net = DQN(input_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    return policy_net, target_net, optimizer, criterion


def process_observation(observation):
    if len(observation) != 64:
        print(f"Invalid observation length: {len(observation)}")
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
    actions = torch.cat(actions, dim=0).to(device)  # Shape: [batch_size, 2]
    rewards = (
        torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    )  # Shape: [batch_size, 1]
    next_states = torch.cat(next_states).to(device)  # Shape: [batch_size, 1, 8, 8]
    dones = (
        torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    )  # Shape: [batch_size, 1]

    # Compute Q-values
    q_values = policy_net(states)  # Shape: [batch_size, 2, 8]

    # Gather Q-values for the actions taken
    action_indices = actions.unsqueeze(-1)  # Shape: [batch_size, 2, 1]
    q_value = q_values.gather(2, action_indices).squeeze(-1)  # Shape: [batch_size, 2]

    # Compute next Q-values using Double DQN
    next_q_values = policy_net(next_states)  # Shape: [batch_size, 2, 8]
    next_actions = next_q_values.argmax(
        dim=2, keepdim=True
    )  # Shape: [batch_size, 2, 1]
    next_q_state_values = target_net(next_states)  # Shape: [batch_size, 2, 8]
    next_q_value = next_q_state_values.gather(2, next_actions).squeeze(
        -1
    )  # Shape: [batch_size, 2]

    # Compute expected Q-values
    expected_q_value = rewards + gamma * next_q_value.max(1)[0].unsqueeze(1) * (
        1 - dones
    )

    # Compute loss
    q_value_max = q_value.max(1)[0].unsqueeze(1)  # Shape: [batch_size, 1]
    loss = criterion(q_value_max, expected_q_value.detach())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def generate_and_send_action(sock, server, observation, epsilon):
    output = modeloutput(observation, epsilon)
    output_message = json.dumps({"output": output})
    print(f"Sending output: {output}")
    sock.sendto(output_message.encode(), server)
    return output


def main():
    server_address = ("localhost", 4242)

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    input_channels = 1
    num_actions = 2

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

    # Epsilon parameters
    epsilon_start = 0.9  # Starting epsilon value
    epsilon_end = 0.1  # Minimum epsilon value
    epsilon_decay = 100000  # Decay rate

    try:
        initial_message = "Hello from Ai"
        print(f"Sending: {initial_message}")
        sock.sendto(initial_message.encode(), server_address)

        while True:
            print("Waiting for data...")
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
                print(f"Received observation: {observation}, reward: {reward}")

                current_state = process_observation(observation)
                if current_state is None:
                    continue

                if prev_state is not None and prev_action is not None:
                    action_indices = [prev_action[0] - 1, prev_action[1] - 1]
                    action_tensor = torch.tensor([action_indices], dtype=torch.long).to(
                        device
                    )  # Move to GPU
                    memory.append(
                        (prev_state, action_tensor, reward, current_state, done)
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

            except json.JSONDecodeError as e:
                print("Error decoding JSON from Godot:", e)
    finally:
        print("Closing socket")
        sock.close()


if __name__ == "__main__":
    main()
