extends TileMapLayer
@onready var Minesweeper: Node = %MinesweeperTileset

var hovered_cell : Vector2i = Vector2i(-1, -1)


func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Minesweeper.mode == "Ai":
		hovered_cell = floor(event.position / (Minesweeper.CELLSIZE * (Minesweeper.BOARDSIZE+1)))


func _process(_delta: float) -> void:
	if (Input.is_action_just_pressed("LeftClick") and hovered_cell != Vector2i(-1, -1)) or Minesweeper.mode != "Human":
		if hovered_cell in Minesweeper.covered_cells and hovered_cell not in Minesweeper.flagged:
			erase_cell(hovered_cell)
			Minesweeper.covered_cells.erase(hovered_cell)
			
			if hovered_cell in Minesweeper.mines:
				reveal_all_mines()
			elif Minesweeper.numbers.get(hovered_cell, -1) == 0:
				flood_fill(hovered_cell)
	
	if Minesweeper.covered_cells.size() == Minesweeper.mines.size():
		if Minesweeper.covered_cells == Minesweeper.mines:
			Minesweeper.gamestatus = 1
			Minesweeper.wins += 1


func flood_fill(cell: Vector2i) -> void:
	var stack = [cell]
	var visited = {}
	var directions = [
		Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1),
		Vector2i(1, 1), Vector2i(-1, -1), Vector2i(1, -1), Vector2i(-1, 1)
	]

	while stack.size() > 0:
		var current = stack.pop_back()
		if current in visited:
			continue
		visited[current] = true
		for direction in directions:
			var neighbor = current + direction
			if neighbor in Minesweeper.covered_cells and neighbor not in Minesweeper.flagged:
				erase_cell(neighbor)
				Minesweeper.covered_cells.erase(neighbor)
				if Minesweeper.numbers.get(neighbor, -1) == 0:
					stack.append(neighbor)


func reveal_all_mines() -> void:
	for mine in Minesweeper.mines:
		if mine in Minesweeper.covered_cells:
			if Minesweeper.mode == "Human":
				await get_tree().create_timer(0.1).timeout
			erase_cell(mine)
			Minesweeper.covered_cells.erase(mine)
	Minesweeper.gamestatus = 1


func set_foreground():
	Minesweeper.covered_cells = Minesweeper.cells.duplicate()
	
	for cell in Minesweeper.covered_cells:
		if (cell.x + cell.y) % 2 == 0:
			set_cell(cell, Minesweeper.SOURCE_ID, Minesweeper.tile("Foreground1"))
		else:
			set_cell(cell, Minesweeper.SOURCE_ID, Minesweeper.tile("Foreground2"))
