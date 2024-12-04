extends Node


@export var BOARDSIZE : int = 8
@export var CELLSIZE : int = 10 #in Pixels
@export var SOURCE_ID = 3
@export var MINE_COUNT : int = 8
var border = []
var cells = []
var mines = []
var flagged = []
var covered_cells = []
var numbers = {}
var timespent = 0
var minesleft = 0
var generation = 0
var wins = 0
var gamestatus = 0


#Set Tile names
func tile(chosenTile):
	var tile_map = {
		"Background1": Vector2i(0, 0),
		"Background2": Vector2i(1, 0),
		"Foreground1": Vector2i(2, 0),
		"Foreground2": Vector2i(3, 0),
		"Mine": Vector2i(4, 0),
		"Flag": Vector2i(5, 0),
		"Hover": Vector2i(6, 0),
		"Border": Vector2i(7, 0),
		"1": Vector2i(0, 1),
		"2": Vector2i(1, 1),
		"3": Vector2i(2, 1),
		"4": Vector2i(3, 1),
		"5": Vector2i(4, 1),
		"6": Vector2i(5, 1),
		"7": Vector2i(6, 1),
		"8": Vector2i(7, 1)
	}
	return tile_map.get(chosenTile, Vector2i(-1, -1))


func remove_duplicates(arr: Array) -> Array:
	var unique_arr = []
	var seen = {}
	for item in arr:
		if not seen.has(item):
			unique_arr.append(item)
			seen[item] = true
	unique_arr.sort()
	return unique_arr


func _process(_delta: float) -> void:
	cells = remove_duplicates(cells)
	covered_cells = remove_duplicates(covered_cells)
	mines = remove_duplicates(mines)
	flagged = remove_duplicates(flagged)
	border = remove_duplicates(border)
