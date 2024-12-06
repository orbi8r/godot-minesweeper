import sys
import socket
import json
import random


def modeloutput():
    return [random.randint(1, 7), random.randint(1, 7)]


def main():
    server_address = ("localhost", 4242)

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Send a message to establish the connection
        initial_message = "Hello from test.py"
        print(f"Sending: {initial_message}")
        sock.sendto(initial_message.encode(), server_address)

        while True:
            # Wait for the data from Godot
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
                print(f"Received observation: {observation}, reward: {reward}")

                # Process the observation and reward to generate output
                output = modeloutput()  # Replace with your model's output
                output_message = json.dumps({"output": output})
                print(f"Sending output: {output}")
                sock.sendto(output_message.encode(), server)
            except json.JSONDecodeError as e:
                print("Error decoding JSON from Godot:", e)
    finally:
        print("Closing socket")
        sock.close()


if __name__ == "__main__":
    main()
