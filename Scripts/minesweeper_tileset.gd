extends Node


@export var BOARDSIZE : int = 8
@export var SOURCE_ID = 3
@export var MINE_COUNT : int = 10
var border = []
var cells = []
var mines = []
var numbers = {}


#Set Tile names
func tile(chosenTile):
	match chosenTile:
		"Background1":
			return Vector2i(0,0)
		"Background2":
			return Vector2i(1,0)
		"Foreground1":
			return Vector2i(2,0)
		"Foreground2":
			return Vector2i(3,0)
		"Mine":
			return Vector2i(4,0)
		"Flag":
			return Vector2i(5,0)
		"Hover":
			return Vector2i(6,0)
		"Border":
			return Vector2i(7,0)
		"1":
			return Vector2i(0,1)
		"2":
			return Vector2i(1,1)
		"3":
			return Vector2i(2,1)
		"4":
			return Vector2i(3,1)
		"5":
			return Vector2i(4,1)
		"6":
			return Vector2i(5,1)
		"7":
			return Vector2i(6,1)
		"8":
			return Vector2i(7,1)
		_:
			return Vector2i(-1,-1)
