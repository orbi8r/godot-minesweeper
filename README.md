# Godot Minesweeper

Minesweeper in Godot, but it's Human vs AI vs Algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

This project is a Minesweeper game developed using the Godot Engine v4.3. The purpose of this project is to see who is faster in beating Minesweeper: a Human, a Deep RL AI, or an Algorithm. 

It features three modes: Human vs AI vs Algorithm, and uses the Stable Baselines 3 model for Deep Reinforcement Learning AI.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/godot-minesweeper.git
    ```
2. Open the project in Godot Engine:
    - Launch Godot Engine.
    - Click on `Import` and navigate to the cloned repository.
    - Select `project.godot` and open it.

## Usage

1. Run the project from the Godot editor by clicking the play button.
2. In the main menu, select the mode you want to play:
    - `Human`: Play as a human player.
    - `AI`: Watch the AI play.
    - `Algorithm`: Observe the algorithm solving the board.

## Project Structure

- `.godot/`: Contains Godot-specific configuration and cache files.
- `Addons/`: Contains any plugins or addons used in the project.
- `Assets/`: Contains game assets like images and icons.
- `Scenes/`: Contains the game scenes.
- `Scripts/`: Contains the game scripts.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
