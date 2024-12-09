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
@onready var algorithm_controller: Node = %Algorithm_controller

var previous_time_spent = -1.0
var previous_mines_left = -1
var previous_wins = -1
var previous_generation = -1

var previous_cell = Vector2i.ZERO
var directions = [
	Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1),
	Vector2i(1, 1), Vector2i(-1, -1), Vector2i(1, -1), Vector2i(-1, 1)
]

var previous_action = null

var previous_observation = null
var new_observation = []

func _process(delta: float) -> void:
	if Minesweeper.gamestatus == 1:
		Minesweeper.gamestatus = 0
		reset()
	
	if previous_observation == null:
		previous_observation = get_current_observation()
	else:
		new_observation = get_current_observation()
		algorithm_controller.previous_observation = previous_observation
		algorithm_controller.new_observation = new_observation
		algorithm_controller.send_data()
		previous_observation = new_observation

	Minesweeper.timespent += delta
	var current_time_spent = floor(Minesweeper.timespent * 10) / 10
	if current_time_spent != previous_time_spent:
		time_spent.text = "Time: " + str(current_time_spent)
		previous_time_spent = current_time_spent

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
	Minesweeper.mode = "Algorithm"
	
	flag_tiles.erase_all_flags()
	background_tiles.set_background()
	foreground_tiles.set_foreground()
	mine_number_tiles.set_mines()
	mine_number_tiles.set_numbers()


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
