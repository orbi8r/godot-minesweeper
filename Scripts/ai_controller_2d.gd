extends AIController2D 

@onready var Minesweeper: Node = %MinesweeperTileset
@onready var ai_game: Node2D = $".."

var cell : Vector2i = Vector2i.ZERO


func get_obs() -> Dictionary:
	var obs = ai_game.get_current_observation()
	return {"obs": []}


func get_reward() -> float:
	return reward


func get_action_space() -> Dictionary:
	return {
		"ChooseX": {"size": 8, "action_type": "discrete"},
		"ChooseY": {"size": 8, "action_type": "discrete"},
	}

func set_action(action) -> void:
	cell.x = action["ChooseX"]+1
	cell.y = action["ChooseY"]+1
	reward += ai_game.calculate_reward(cell)
	ai_game.update_board_with_action(cell)
