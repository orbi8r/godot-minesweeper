import sys
import socket
import json
import random


def algorithm_output(observation):
    grid = [[0] * 8 for i in range(8)]
    for i in range(len(observation)):
        x = i // 8
        y = i % 8
        grid[x][y] = observation[i]

    direction = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    action = [[0] * 8 for i in range(8)]

    for i in range(8):
        for j in range(8):
            if grid[i][j] != -1 and grid[i][j] != 0:
                number_of_covered_neighbours = 0
                for d in direction:
                    x = i + d[0]
                    y = j + d[1]
                    if x >= 0 and x < 8 and y >= 0 and y < 8:
                        if grid[x][y] == -1:
                            number_of_covered_neighbours += 1
                if number_of_covered_neighbours != 0:
                    for d in direction:
                        x = i + d[0]
                        y = j + d[1]
                        if 0 <= x < 8 and 0 <= y < 8:
                            if grid[x][y] == -1:
                                probability = grid[i][j] / number_of_covered_neighbours
                                action[x][y] = max(action[x][y], probability)

    for i in range(8):
        for j in range(8):
            if action[i][j] == 1:
                grid[i][j] = -2
                for d in direction:
                    x = i + d[0]
                    y = j + d[1]
                    if x >= 0 and x < 8 and y >= 0 and y < 8:
                        if grid[x][y] > 0:
                            grid[x][y] -= 1

    action = [[0] * 8 for i in range(8)]

    for i in range(8):
        for j in range(8):
            if grid[i][j] == 0:
                for d in direction:
                    x = i + d[0]
                    y = j + d[1]
                    if x >= 0 and x < 8 and y >= 0 and y < 8:
                        if grid[x][y] == -1:
                            return (x + 1, y + 1)
            elif grid[i][j] != -1 and grid[i][j] != -2:
                number_of_covered_neighbours = 0
                for d in direction:
                    x = i + d[0]
                    y = j + d[1]
                    if x >= 0 and x < 8 and y >= 0 and y < 8:
                        if grid[x][y] == -1:
                            number_of_covered_neighbours += 1
                if number_of_covered_neighbours != 0:
                    for d in direction:
                        x = i + d[0]
                        y = j + d[1]
                        if 0 <= x < 8 and 0 <= y < 8:
                            if grid[x][y] == -1:
                                probability = grid[i][j] / number_of_covered_neighbours
                                action[x][y] = max(action[x][y], probability)

    min_value = 2
    min_coords = []
    for i in range(8):
        for j in range(8):
            if action[i][j] < min_value and grid[i][j] == -1:
                for d in direction:
                    x = i + d[0]
                    y = j + d[1]
                    if x >= 0 and x < 8 and y >= 0 and y < 8:
                        if grid[x][y] != -1:
                            min_value = action[i][j]
                            break
    for i in range(8):
        for j in range(8):
            if action[i][j] == min_value and grid[i][j] == -1:
                for d in direction:
                    x = i + d[0]
                    y = j + d[1]
                    if x >= 0 and x < 8 and y >= 0 and y < 8:
                        if grid[x][y] != -1:
                            min_coords.append((i + 1, j + 1))
                            break
    if min_coords.__len__() == 0:
        for i in range(8):
            for j in range(8):
                if grid[i][j] == -1:
                    min_coords.append((i + 1, j + 1))
    if min_coords.__len__() == 0:
        return (0, 0)
    else:
        return random.choice(min_coords)


def main():
    server_address = ("localhost", 4242)

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Send a message to establish the connection
        initial_message = "Hello from Algorithm"
        print(f"Sending: {initial_message}")
        sock.sendto(initial_message.encode(), server_address)

        while True:
            # Wait for the data from Godot
            data, server = sock.recvfrom(65536)
            message = data.decode()

            if message == "close":
                print("Received close message. Terminating connection.")
                break

            try:
                data_received = json.loads(message)
                prev_observation = data_received["prev_observation"]
                new_observation = data_received["new_observation"]
                action = data_received["action"]

                # Process the observation to generate output
                output = algorithm_output(new_observation)
                print(f"Output: {output}")
                output_message = json.dumps(
                    {
                        "action": {"x": output[0], "y": output[1]},
                        "observation": new_observation,
                    }
                )
                sock.sendto(output_message.encode(), server)
            except json.JSONDecodeError as e:
                print("Error decoding JSON from Godot:", e)
    finally:
        print("Closing socket")
        sock.close()


if __name__ == "__main__":
    main()
