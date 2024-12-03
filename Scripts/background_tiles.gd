extends TileMapLayer


func set_background():
	
	for x in range(0,Minesweeper.BOARDSIZE+2):
		for y in range(0,Minesweeper.BOARDSIZE+2):
			if (x == 0 or x == Minesweeper.BOARDSIZE+1) or (y == 0 or y == Minesweeper.BOARDSIZE+1):
				Minesweeper.border.append(Vector2i(x,y))
			else: 
				Minesweeper.cells.append(Vector2i(x,y))
	
	for cell in Minesweeper.border:
		set_cell(cell,Minesweeper.SOURCE_ID,Minesweeper.tile("Border"))
		
	for cell in Minesweeper.cells:
		if (cell.x + cell.y) % 2 == 0:
			set_cell(cell,Minesweeper.SOURCE_ID,Minesweeper.tile("Background1"))
		else:
			set_cell(cell,Minesweeper.SOURCE_ID,Minesweeper.tile("Background2"))
