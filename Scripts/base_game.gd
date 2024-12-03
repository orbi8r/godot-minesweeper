extends Node2D

@onready var hover_tiles: TileMapLayer = %HoverTiles
@onready var flag_tiles: TileMapLayer = %FlagTiles
@onready var foreground_tiles: TileMapLayer = %ForegroundTiles
@onready var mine_number_tiles: TileMapLayer = %MineNumberTiles
@onready var background_tiles: TileMapLayer = %BackgroundTiles


func _ready() -> void:
	reset()


func reset():
	background_tiles.set_background()
	mine_number_tiles.set_mines()
	mine_number_tiles.set_numbers()
