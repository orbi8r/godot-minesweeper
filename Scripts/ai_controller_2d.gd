extends Node

var udp_socket: PacketPeerUDP = PacketPeerUDP.new()
var port := 4242
var python_ip: String = ""
var python_port: int = 0
var initial_message_received: bool = false

@onready var Minesweeper: Node = %MinesweeperTileset
@onready var foreground_tiles: TileMapLayer = %ForegroundTiles

@onready var ai_game: Node2D = $"../.."
@onready var ai_controller: Node = $"."

var action = Vector2i(0,0)
@export var observation_array = []
@export var reward = 0.0:
	set = _set_reward

var current_action = Vector2i.ZERO  # Initialize current_action
var previous_observation = []
var new_observation = []

func _set_reward(value):
	reward = value
	send_observation_and_reward()


func _ready() -> void:
	# Bind the UDP socket to the port
	var error = udp_socket.bind(port)
	if error != OK:
		print("Failed to bind UDP socket to port ", port)
	else:
		print("UDP socket listening on port ", port)
	# Set process modes
	ai_game.process_mode = Node.PROCESS_MODE_INHERIT  # ai_game will inherit process mode
	self.process_mode = Node.PROCESS_MODE_ALWAYS  # ai_controller will always process
	# Pause the scene tree
	get_tree().paused = true


func _process(_delta):
	# This will continue running even when the scene tree is paused
	# Check if there's any incoming data
	while udp_socket.get_available_packet_count() > 0:
		var result = udp_socket.get_packet()
		var message = result.get_string_from_utf8()
		
		if not initial_message_received:
			# Check for the initial message from the Python script
			if message == "Hello from Ai":
				initial_message_received = true
				python_ip = udp_socket.get_packet_ip()
				python_port = udp_socket.get_packet_port()
				print("Initial message received from ", python_ip, ":", python_port)
				# Unpause the scene tree once the connection is established
				get_tree().paused = false
		else:
			# Process the result from the Python script
			var data = JSON.parse_string(message)
			var output = data["action"]
			current_action = Vector2i(output["x"], output["y"])  # Update current_action
			# Update the board with the received action
			update_board_with_action(current_action)


func send_observation_and_reward():
	if initial_message_received and observation_array.size() == 64:
		var data = {
			"observation": observation_array,
			"reward": reward,
		}
		var message = JSON.stringify(data)
		udp_socket.set_dest_address(python_ip, python_port)
		udp_socket.put_packet(message.to_utf8_buffer())
		print(message,action)


func send_data():
	if initial_message_received and previous_observation.size() == 64 and new_observation.size() == 64:
		var data = {
			"prev_observation": previous_observation,
			"new_observation": new_observation,
			"reward": reward,
			"action": {"x": current_action.x, "y": current_action.y}
		}
		var message = JSON.stringify(data)
		udp_socket.set_dest_address(python_ip, python_port)
		udp_socket.put_packet(message.to_utf8_buffer())


func _exit_tree():
	send_close_message()


func send_close_message():
	var message = "close"
	udp_socket.set_dest_address(python_ip, python_port)
	udp_socket.put_packet(message.to_utf8_buffer())
	print("Sent close message")


func update_board_with_action(action_pos: Vector2i):
	if action_pos in Minesweeper.covered_cells and action_pos not in Minesweeper.flagged:
		foreground_tiles.erase_cell(action_pos)
		Minesweeper.covered_cells.erase(action_pos)
		
		if action_pos in Minesweeper.mines:
			foreground_tiles.reveal_all_mines()
		elif Minesweeper.numbers.get(action_pos, -1) == 0:
			foreground_tiles.flood_fill(action_pos)
		else:
			foreground_tiles.set_cell(action_pos, Minesweeper.SOURCE_ID, Minesweeper.tile(str(Minesweeper.numbers[action_pos])))
	
	if Minesweeper.covered_cells.size() == Minesweeper.mines.size():
		if Minesweeper.covered_cells == Minesweeper.mines:
			Minesweeper.gamestatus = 1
			Minesweeper.wins += 1
