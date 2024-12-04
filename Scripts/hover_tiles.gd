extends TileMapLayer

@onready var Minesweeper: Node = %MinesweeperTileset


func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var hovered_cell : Vector2i = floor(event.position / (Minesweeper.CELLSIZE * (Minesweeper.BOARDSIZE+1)))
		for cell in Minesweeper.cells:
			if cell == hovered_cell and cell in Minesweeper.covered_cells:
				set_cell(hovered_cell, Minesweeper.SOURCE_ID, Minesweeper.tile("Hover"))
			else:
				erase_cell(cell)
