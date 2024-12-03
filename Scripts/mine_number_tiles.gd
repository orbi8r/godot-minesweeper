extends TileMapLayer


func set_mines():
	var number_of_minesleft = Minesweeper.MINE_COUNT
	
	while number_of_minesleft != 0:
		var chosen_cell = Minesweeper.cells.pick_random()
		
		if chosen_cell not in Minesweeper.mines:
			set_cell(chosen_cell,Minesweeper.SOURCE_ID,Minesweeper.tile("Mine"))
			Minesweeper.mines.append(chosen_cell)
			number_of_minesleft -= 1


func set_numbers():
	for cell in Minesweeper.cells:
		Minesweeper.numbers[cell] = 0
	
	for cell in Minesweeper.cells:
		if cell in Minesweeper.mines:
			if Vector2i(cell.x + 1,cell.y + 1) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x + 1,cell.y + 1)] += 1
			if Vector2i(cell.x + 1,cell.y - 1) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x + 1,cell.y - 1)] += 1
			if Vector2i(cell.x - 1,cell.y + 1) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x - 1,cell.y + 1)] += 1
			if Vector2i(cell.x - 1,cell.y - 1) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x - 1,cell.y - 1)] += 1
			if Vector2i(cell.x + 1,cell.y) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x + 1,cell.y)] += 1
			if Vector2i(cell.x - 1,cell.y) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x - 1,cell.y)] += 1
			if Vector2i(cell.x,cell.y + 1) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x,cell.y + 1)] += 1
			if Vector2i(cell.x,cell.y - 1) in Minesweeper.cells:
				Minesweeper.numbers[Vector2i(cell.x,cell.y - 1)] += 1
			
			
	for cell in Minesweeper.cells:
		if cell not in Minesweeper.mines:
			set_cell(cell,Minesweeper.SOURCE_ID,Minesweeper.tile(str(Minesweeper.numbers[cell])))
