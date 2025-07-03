# 🔺 Single Player Tetress Solver using A* Search

This project implements an intelligent solver for a grid-based game involving colored tiles and Tetrimino-style placements. The goal is to eliminate a **target BLUE token** by placing **RED tetrimino pieces** optimally.

> ⚠️ Note: This repository contains **only the `search.py` file**. You will need to implement the rest of the environment (`core.py`, I/O parsing, etc.) to run the solver.

---

## 💡 Problem Overview

Single-player **Tetress** is a simplified version of the full two-player game, where only Red plays, and the aim is to eliminate a single **target Blue token** by clearing the row or column it's in — using the fewest possible **PLACE** actions.

### Game Rules Summary

* Grid is 11×11, with wrapping behavior (top connects to bottom, left to right).
* Red plays only `PLACE` actions (4-tile Tetrimino shapes).
* Completing a row or column removes all tokens in it.
* The goal is to remove the **target Blue token** in the shortest number of moves.
* If no solution is possible, return `None`.

---

## 🚀 Features

* ✅ A\* Search with a priority queue (`heapq`) and custom `Node` class
* ✅ Heuristic combines:

  * Number of empty cells near the target
  * Manhattan distance to nearby RED tiles
  * Count of isolated (non-adjacent) RED tiles
* ✅ 19 legal Tetrimino variants used to generate successors
* ✅ Accurate tile placement with line clearing and board wrapping
* ✅ Efficient state space pruning and hashing for performance

---

## 📐 Heuristic Design

The `calculate_h` function estimates the cost to clear the target cell by balancing:

* `num_empty`: Empty cells near the target — the fewer, the better.
* `num_alone`: RED tokens that are isolated — these are penalized.
* `manhattan_distance`: Distance to nearest RED token — closer is better.

**Combined Heuristic:**

```text
h(n) = num_empty + 10 * num_alone + (manhattan_distance / 5)
```

---

## 🔧 Key Components

| Function                | Description                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `search()`              | Main A\* search loop                                           |
| `generate_successors()` | Generates all valid `PlaceAction` states                       |
| `calculate_h()`         | Heuristic cost estimate for A\*                                |
| `apply_piece()`         | Places a tetrimino and simulates line clearing                 |
| `check_clear()`         | Removes full rows or columns                                   |
| `generate_pieces()`     | Returns all 19 rotated/translated tetromino shapes from a cell |

---


## 📁 Repository Structure

```
.
├── search/
│   └── program.py       # Entry point for the A* Tetress solver
├── test_cases/          # Contains 18 different example test cases
└── README.md            # Project documentation
```


---

## 🛠️ Installation & Usage

### Requirements

* Python 3.12
* No external libraries required beyond the standard library.

### Setup

```bash
git clone https://github.com/NemoDeFish/a-star-search-for-single-player-tetress
cd a-star-search
```

### Running the Agent

You must implement or stub the following files to use `search.py`:

1. `core.py`

    Provides required data types.


2. Input Handler (e.g. `__main__.py`)

    You need a script that:

    * Parses `.csv` board files into a `{Coord: PlayerColor}` dictionary
    * Extracts the target `Coord`
    * Calls the `search(initial_state, target)` function
    * Prints actions or `None`

    Example call:

    ```bash
    python -m search < test-vis1.csv
    ```

    ---

    ### 3. CSV Board Format

    * 11×11 grid of comma-separated values
    * `r` = RED token
    * `b` = BLUE token
    * `B` = TARGET BLUE token
    * empty = empty cell

    Example:

    ```csv
    ,,,,,,,,,,,
    ,,,,,,r,,,,
    ,,,,,,r,,,,
    ,,,,rr,,,,,
    ,,,,,,r,,,,
    ,,,,,,r,,,,
    ,,,,,,r,,,,
    ,,,,,,r,,,,
    ,,,,,,r,,,,
    ,,,,,,r,,,B
    ,,,,,,,,,,,
    ```

---

## 🧪 Test Cases

18 test cases are provided in the `test_cases/` directory:

These `.csv` files describe the board layout using:

* `r`: Red token
* `b`: Blue token
* `B`: Target Blue token
* empty cell: blank
