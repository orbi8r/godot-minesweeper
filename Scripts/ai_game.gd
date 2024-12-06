extends Node2D

@onready var hover_tiles: TileMapLayer = %HoverTiles
@onready var flag_tiles: TileMapLayer = %FlagTiles
@onready var foreground_tiles: TileMapLayer = %ForegroundTiles
@onready var mine_number_tiles: TileMapLayer = %MineNumberTiles
@onready var background_tiles: TileMapLayer = %BackgroundTiles
@onready var Minesweeper: Node = %MinesweeperTileset
@onready var time_spent: TextEdit = $UI/TimeSpent
@onready var mines_left: TextEdit = $UI/MinesLeft
@onready var wins: TextEdit = $UI/Wins
@onready var generation: TextEdit = $UI/Generation
@onready var ai_controller: Node = %ai_controller

var previous_time_spent = -1.0
var previous_mines_left = -1
var previous_wins = -1
var previous_generation = -1
var rewards_collection = 0

var previous_cell = Vector2i.ZERO
var directions = [
		Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1),
		Vector2i(1, 1), Vector2i(-1, -1), Vector2i(1, -1), Vector2i(-1, 1)
	]

func _process(delta: float) -> void:
	
	if Minesweeper.gamestatus == 1:
		Minesweeper.gamestatus = 0
		reset()
	
	ai_controller.observation_array = ai_observation()
	ai_controller.reward = ai_input_and_reward()
	
	Minesweeper.timespent += delta
	var current_time_spent = floor(Minesweeper.timespent * 10) / 10
	if current_time_spent != previous_time_spent:
		time_spent.text = "Time: " + str(current_time_spent)
		previous_time_spent = current_time_spent
	
	#rewards_collection += ai_controller_2d.reward
	#mines_left.text = "Reward: " + str(floor(rewards_collection)) + " (" + str(ai_controller_2d.reward) + ")"

	if Minesweeper.wins != previous_wins:
		wins.text = "Wins: " + str(Minesweeper.wins)
		previous_wins = Minesweeper.wins

	if Minesweeper.generation != previous_generation:
		generation.text = "Generation: " + str(Minesweeper.generation)
		previous_generation = Minesweeper.generation


func reset():
	Minesweeper.covered_cells.clear()
	Minesweeper.flagged.clear()
	Minesweeper.mines.clear()
	Minesweeper.numbers.clear()
	Minesweeper.timespent = 0
	Minesweeper.generation += 1
	Minesweeper.mode = "Ai"
	rewards_collection = 0
	
	flag_tiles.erase_all_flags()
	background_tiles.set_background()
	foreground_tiles.set_foreground()
	mine_number_tiles.set_mines()
	mine_number_tiles.set_numbers()


func ai_observation():
	var obs = []
	for cell in Minesweeper.cells:
		if cell in Minesweeper.covered_cells:
			obs.append(-1)
		else:
			if cell in Minesweeper.mines:
				obs.append(-2)
			else:
				obs.append(Minesweeper.numbers[cell])
	return obs


func ai_input_and_reward():
	foreground_tiles.hovered_cell = ai_controller.action
	
	var adjacent_cells = []
	for dir in directions:
		adjacent_cells.append(previous_cell + dir)
	
	var reward = 0

	if ai_controller.action not in Minesweeper.covered_cells:
		reward -= 0.3
	elif ai_controller.action in adjacent_cells:
		reward += 0.9
	elif ai_controller.action in Minesweeper.mines:
		print("Loss",Minesweeper.generation)
		reward -= 1
	elif Minesweeper.wins != previous_wins:
		print("win",Minesweeper.generation)
		reward += 2
	else:
		reward -= 0.3
	
	if Minesweeper.timespent > 100:
		reward -= 1
		reset()
		
	previous_cell = ai_controller.action
	return reward


### AI SIDE
#@onready var minesweeper_tileset: Node = %MinesweeperTileset
#@onready var ai_game: Node2D = $".."
#
#var cell : Vector2i = Vector2i.ZERO
#
#
#func get_obs() -> Dictionary:
	#var obs = ai_game.ai_observation()
	#return {"obs": []}
#
#
#func get_reward() -> float:
	#return reward
#
#
#func get_action_space() -> Dictionary:
	#return {
		#"ChooseX": {"size": 8, "action_type": "discrete"},
		#"ChooseY": {"size": 8, "action_type": "discrete"},
	#}
#
#func set_action(action) -> void:
	#cell.x = action["ChooseX"]+1
	#cell.y = action["ChooseY"]+1
