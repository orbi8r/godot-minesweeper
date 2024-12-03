extends TileMapLayer


func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var select_cell : Vector2i = floor(event.position / (Minesweeper.CELLSIZE * (Minesweeper.BOARDSIZE+1)))
		var hovered_cell = select_cell
		for cell in Minesweeper.cells:
			if cell == hovered_cell:
				set_cell(hovered_cell, Minesweeper.SOURCE_ID, Minesweeper.tile("Hover"))
			elif cell != hovered_cell:
				erase_cell(cell)
