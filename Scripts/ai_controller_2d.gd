extends Node2D

var udp_socket := PacketPeerUDP.new()
var port := 4242
var array := [1, 2, 3, 4, 5]
var initial_message_received := false
var python_ip = ""
var python_port = 0

func _ready() -> void:
	# Bind the UDP socket to the port
	var error = udp_socket.bind(port)
	if error != OK:
		print("Failed to bind UDP socket to port ", port)
	else:
		print("UDP socket listening on port ", port)

func _process(_delta):
	# Check if there's any incoming data
	while udp_socket.get_available_packet_count() > 0:
		var packet = udp_socket.get_packet()
		var message = packet.get_string_from_utf8()
		print("Received: ", message)
		
		if not initial_message_received:
			# Check for the initial message from the Python script
			if message == "Hello from test.py":
				initial_message_received = true
				python_ip = udp_socket.get_packet_ip()
				python_port = udp_socket.get_packet_port()
				print("Initial message received from ", python_ip, ":", python_port)
				print("Sending array...")
				send_array()
		else:
			# Process the result from the Python script
			print("Result from Python script: ", message)

func send_array():
	var message = str(array)
	udp_socket.set_dest_address(python_ip, python_port)
	udp_socket.put_packet(message.to_utf8_buffer())
	print("Sent array: ", message)
