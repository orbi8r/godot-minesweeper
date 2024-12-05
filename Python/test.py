import sys
import socket
import ast


def sum_array(array):
    return sum(array)


def main():
    server_address = ("localhost", 4242)

    # Create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Send a message to establish the connection
        initial_message = "Hello from test.py"
        print(f"Sending: {initial_message}")
        sock.sendto(initial_message.encode(), server_address)

        # Wait for the array from Godot
        print("Waiting for array...")
        data, server = sock.recvfrom(4096)
        message = data.decode()
        array = ast.literal_eval(message)
        print(f"Received array: {array}")

        # Process the array (e.g., sum the elements)
        result = sum_array(array)
        print(f"Sum of array: {result}")

        # Send the result back to Godot
        print(f"Sending result: {result}")
        sock.sendto(str(result).encode(), server)

    finally:
        print("Closing socket")
        sock.close()


if __name__ == "__main__":
    main()
