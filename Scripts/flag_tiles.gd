extends TileMapLayer

var hovered_cell : Vector2i
@onready var Minesweeper: Node = %MinesweeperTileset


func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		hovered_cell = floor(event.position / (Minesweeper.CELLSIZE * (Minesweeper.BOARDSIZE+1)))


func _process(_delta: float) -> void:
	Minesweeper.minesleft = Minesweeper.MINE_COUNT - Minesweeper.flagged.size()
	if Input.is_action_just_pressed("RightClick"):
		if hovered_cell not in Minesweeper.flagged:
			if hovered_cell in Minesweeper.cells and Minesweeper.flagged.size() < Minesweeper.MINE_COUNT:
				set_cell(hovered_cell, Minesweeper.SOURCE_ID, Minesweeper.tile("Flag"))
				Minesweeper.flagged.append(hovered_cell)
		else:
			erase_cell(hovered_cell)
			Minesweeper.flagged.erase(hovered_cell)


func erase_all_flags():
	for cell in Minesweeper.cells:
		erase_cell(cell)
