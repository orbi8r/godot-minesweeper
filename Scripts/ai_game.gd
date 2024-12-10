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
@onready var ai_controller: Node2D = $AIController2D

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

var previous_action = null
var previous_reward = 0

var previous_observation = null

func _process(delta: float) -> void:
	if Minesweeper.gamestatus == 1:
		Minesweeper.gamestatus = 0
		reset()

	Minesweeper.timespent += delta
	var current_time_spent = floor(Minesweeper.timespent * 10) / 10
	if current_time_spent != previous_time_spent:
		time_spent.text = "Time: " + str(current_time_spent)
		previous_time_spent = current_time_spent
	
	rewards_collection += ai_controller.reward
	mines_left.text = "Reward: " + str(ai_controller.reward) + " (" + str(floor(rewards_collection)) + ")"

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


func calculate_reward(current_action):
	var reward = 0

	# Additional reward logic
	if current_action in Minesweeper.mines:
		reward -= 1
	elif current_action not in Minesweeper.covered_cells and current_action in Minesweeper.cells:
		reward -= 0.9
	else:
		var adjacent_to_uncovered = false
		for dir in directions:
			var adjacent_cell = current_action + dir
			if adjacent_cell in Minesweeper.cells and adjacent_cell not in Minesweeper.covered_cells:
				adjacent_to_uncovered = true
				break
		
		if adjacent_to_uncovered or Minesweeper.covered_cells == Minesweeper.cells:
			reward += 0.5  # Reward for selecting a cell near uncovered cells
		else:
			reward -= 0.7  # Penalize for selecting a cell not near uncovered cells

	# Additional reward for winning
	if Minesweeper.wins != previous_wins:
		print("win", Minesweeper.generation)
		reward += 2

	# Store the current action, and reward as previous
	previous_action = current_action
	previous_reward = reward

	return reward


func get_current_observation():
	var observation = []
	for cell in Minesweeper.cells:
		if cell in Minesweeper.covered_cells:
			observation.append(-1)
		elif cell in Minesweeper.mines:
			observation.append(-2)
		else:
			observation.append(Minesweeper.numbers.get(cell, 0))
	return observation
	
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
