import sys
import socket
import json
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Hyperparameters and configuration
INPUT_CHANNELS = 1
NUM_ACTIONS = 64
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.01  # Increased initial learning rate
MIN_LEARNING_RATE = 1e-5
MEMORY_SIZE = 50000
TARGET_UPDATE = 1000
SAVE_INTERVAL = 10000
PRINT_INTERVAL = 1000
EPSILON_START = 0.9
EPSILON_END = 0.001
EPSILON_DECAY = 0.99999
NEGATIVE_REWARD_TRAINING = 1
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
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
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

    # Penalize already covered cells
    for idx, val in enumerate(observation):
        if val != -1:
            action_values[0, idx] -= 1e6

    valid_actions = get_valid_actions(observation)
    # Epsilon-greedy action selection
    if random.random() < epsilon:
        if valid_actions != []:
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
    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=5
    )  # Adjusted scheduler parameters
    criterion = nn.MSELoss()
    return policy_net, target_net, optimizer, scheduler, criterion


def save_model(model, filename):
    model_cpu = model.to("cpu")
    torch.save(model_cpu.state_dict(), filename)
    model.to(device)


def process_observation(observation):
    if len(observation) != 64:
        return None
    return torch.tensor(observation, dtype=torch.float32).view(1, 1, 8, 8).to(device)


def train_on_data(
    data,
    policy_net,
    target_net,
    optimizer,
    scheduler,
    criterion,
    device,
    batch_size,
    gamma=GAMMA,
):
    if len(data) < batch_size:
        return None

    batch = random.sample(data, batch_size)
    filtered_batch = []
    processed_experiences = set()  # Set to track processed experiences

    for experience in batch:
        prev_observation, action, reward, new_observation, done = experience

        # Convert tensors to numpy arrays and then to bytes
        experience_hash = (
            prev_observation.cpu().numpy().tobytes(),
            action,
            reward,
            new_observation.cpu().numpy().tobytes(),
            done,
        )

        # Check if the experience has already been processed
        if experience_hash in processed_experiences:
            continue

        # Process negative rewards only sometimes
        if reward <= 0 and random.random() >= NEGATIVE_REWARD_TRAINING:
            continue

        filtered_batch.append(experience)
        processed_experiences.add(
            experience_hash
        )  # Add to the set of processed experiences

    if len(filtered_batch) < batch_size:
        return None

    states, actions, rewards, next_states, dones = zip(*filtered_batch)

    # Concatenate the tensors instead of creating new ones
    states = torch.cat(states, dim=0).to(device)  # Shape: [batch_size, 1, 8, 8]
    next_states = torch.cat(next_states, dim=0).to(device)

    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
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
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(loss)

    # Return the loss value
    return loss.item()


def main():
    global policy_net
    policy_net, target_net, optimizer, scheduler, criterion = initialize_model(
        INPUT_CHANNELS, NUM_ACTIONS
    )
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START
    total_steps = 0
    last_data = None  # Variable to store the last data
    total_reward = 0  # Variable to track total reward
    last_print_total_steps = 0  # Variable to store the last print step
    last_save_step = 0  # Variable to store the last save step
    last_target_update_step = 0  # Variable to store the last target update step
    last_batch_total_steps = 0  # Variable to store the last batch total steps
    latest_train_on_data_loss = (
        None  # Variable to store the latest loss from train_on_data
    )
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

                # Only proceed if prev_observation is not equal to new_observation
                if (
                    prev_observation != new_observation
                    and prev_observation != [-1] * 64
                ):
                    x = action["x"] - 1
                    y = action["y"] - 1
                    action_index = x * 8 + y
                    if 0 <= action_index < 64:
                        if last_data != data_received:
                            memory.append(
                                (prev_state, action_index, reward, current_state, done)
                            )
                            # Update the last data
                            last_data = data_received

                            # Update total_steps before the training condition
                            total_steps += 1
                            total_reward += reward

                # Training condition: train every BATCH_SIZE steps
                if (
                    total_steps % BATCH_SIZE == 0
                    and len(memory) >= BATCH_SIZE
                    and total_steps != last_batch_total_steps
                ):
                    train_on_data_loss = train_on_data(
                        memory,
                        policy_net,
                        target_net,
                        optimizer,
                        scheduler,
                        criterion,
                        device,
                        BATCH_SIZE,
                    )
                    if train_on_data_loss is not None:
                        latest_train_on_data_loss = train_on_data_loss
                    last_batch_total_steps = (
                        total_steps  # Update the last batch total steps
                    )

                # Update epsilon
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

                # Generate the next action
                next_action = model_output(new_observation, epsilon)
                action_index = next_action[0] * 8 + next_action[1] - 9
                x = action_index // 8 + 1
                y = action_index % 8 + 1
                action_to_send = {"x": x, "y": y}
                action_message = json.dumps(
                    {"action": action_to_send, "observation": new_observation}
                )
                sock.sendto(action_message.encode(), server)

                if (
                    total_steps % TARGET_UPDATE == 0
                    and total_steps != last_target_update_step
                ):
                    target_net.load_state_dict(policy_net.state_dict())
                    last_target_update_step = (
                        total_steps  # Update the last target update step
                    )

                if (
                    total_steps % PRINT_INTERVAL == 0
                    and total_steps != last_print_total_steps
                ):
                    avg_reward = total_reward / PRINT_INTERVAL
                    for param_group in optimizer.param_groups:
                        current_lr = param_group["lr"]
                    print(
                        f"Step: {total_steps}, Epsilon: {epsilon:.4f}, Avg Reward: {avg_reward:.4f}, Train loss: {latest_train_on_data_loss}, Learning rate: {current_lr}"
                    )
                    total_reward = 0  # Reset total reward
                    last_print_total_steps = total_steps  # Update the last print step

                if total_steps % SAVE_INTERVAL == 0 and total_steps != last_save_step:
                    save_model(policy_net, f"policy_net_{total_steps}.pth")
                    print(f"Model saved at step {total_steps}")
                    last_save_step = total_steps  # Update the last save step

                # Update learning rate
                optimizer.param_groups[0]["lr"] = max(
                    optimizer.param_groups[0]["lr"], MIN_LEARNING_RATE
                )

            except json.JSONDecodeError as e:
                print("Error decoding JSON from Godot:", e)

    finally:
        print("Closing socket")
        sock.close()
        print("Final model saved")
        save_model(policy_net, "final_policy_net.pth")


if __name__ == "__main__":
    main()
