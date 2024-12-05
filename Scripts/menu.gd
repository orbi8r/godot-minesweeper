extends Node2D


func _on_human_pressed() -> void:
	get_tree().change_scene_to_file("res://Scenes/base_game.tscn")


func _on_ai_pressed() -> void:
	get_tree().change_scene_to_file("res://Scenes/ai_game.tscn")
