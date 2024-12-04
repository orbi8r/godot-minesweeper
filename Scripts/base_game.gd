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

var previous_time_spent = -1.0
var previous_mines_left = -1
var previous_wins = -1
var previous_generation = -1


func _process(delta: float) -> void:
	if Minesweeper.gamestatus == 1:
		Minesweeper.gamestatus = 0
		reset()
	
	Minesweeper.timespent += delta
	var current_time_spent = floor(Minesweeper.timespent * 10) / 10
	if current_time_spent != previous_time_spent:
		time_spent.text = "Time: " + str(current_time_spent)
		previous_time_spent = current_time_spent

	if Minesweeper.minesleft != previous_mines_left:
		mines_left.text = "Flags Left: " + str(Minesweeper.minesleft)
		previous_mines_left = Minesweeper.minesleft

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
	
	flag_tiles.erase_all_flags()
	background_tiles.set_background()
	foreground_tiles.set_foreground()
	mine_number_tiles.set_mines()
	mine_number_tiles.set_numbers()
