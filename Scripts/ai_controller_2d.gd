extends Node

var udp_socket := PacketPeerUDP.new()
var port := 4242
var python_ip = ""
var python_port = 0
var initial_message_received := false

@onready var ai_game: Node2D = $"../.."
@onready var ai_controller: Node = $"."

@export var array = []
func set_array(value):
	array = value
	send_array()


func _ready() -> void:
	# Bind the UDP socket to the port
	var error = udp_socket.bind(port)
	if error != OK:
		print("Failed to bind UDP socket to port ", port)
	else:
		print("UDP socket listening on port ", port)
	# Set process modes
	ai_game.process_mode = Node.ProcessMode.PROCESS_MODE_INHERIT  # ai_game will inherit process mode
	self.process_mode = Node.ProcessMode.PROCESS_MODE_ALWAYS    # ai_controller will always process
	# Pause the scene tree
	get_tree().paused = true

func _process(_delta):
	# This will continue running even when the scene tree is paused
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
				# Unpause the scene tree once the connection is established
				get_tree().paused = false
		else:
			# Process the result from the Python script
			print("Result from Python script: ", message)
			# Handle the message as needed

func send_array():
	var message = str(array)
	udp_socket.set_dest_address(python_ip, python_port)
	udp_socket.put_packet(message.to_utf8_buffer())
	print("Sent array: ", message)

func _exit_tree():
	send_close_message()

func send_close_message():
	var message = "close"
	udp_socket.set_dest_address(python_ip, python_port)
	udp_socket.put_packet(message.to_utf8_buffer())
	print("Sent close message")
