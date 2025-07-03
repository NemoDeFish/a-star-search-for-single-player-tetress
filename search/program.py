import heapq
from typing import Generator
from .core import PlayerColor, Coord, PlaceAction
from .utils import render_board

# Define the size of the grid
ROW = 11
COL = 11
RADIUS = 4


class Node:
    """
    A class representing the nodes in the A* search algorithm.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: array of Coords that help to trace the place actions
            leading to the current node.
        `depth`: depth of the node
    """

    def __init__(
        self,
        board: dict[Coord, PlayerColor] = None,
        action_sequence: list[Coord] = None,
        depth: int = None,
    ):
        # Current state of the board
        self.board = board
        # Actions taken to reach this state
        self.action_sequence = action_sequence
        # Depth of this node in the search tree
        self.depth = depth

        # f, g, and h values for use in A* search algorithm
        self.f = 0
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.board == other.board


def search(board: dict[Coord, PlayerColor], target: Coord) -> list[PlaceAction] | None:
    """
    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: the target BLUE coordinate to remove from the board.

    Returns:
        A list of "place actions" as PlaceAction instances, or `None` if no
        solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, target, ansi=True))

    # Do some impressive AI stuff here to find the solution...

    # Initialise the closed list containing visited nodes
    closed_list = []

    # Initialise the start node details containing red tile
    start_node = Node(board=board, action_sequence=[], depth=0)
    start_node.f = start_node.g = start_node.h = 0

    # Initialise the open list containing nodes to be visited with the start node
    open_list = []
    heapq.heappush(open_list, start_node)

    # Main loop of A* search algorithm
    while len(open_list) > 0:
        # Pop the node with the smallest f value from the open list
        current_node = heapq.heappop(open_list)

        # Mark the cell as visited
        closed_list.append(current_node)

        # For each direction, check the successors
        for new_board, action in generate_successors(current_node.board):
            # If the successor is not visited, we continue, else we exit the loop
            for closed_child in closed_list:
                if current_node == closed_child:
                    continue

            # Calculate the new f, g, and h values
            new_g = current_node.g + 1
            new_h = calculate_h(new_board, target)
            new_f = new_g + new_h

            # Update the successor node details
            successor_node = Node(
                board=new_board,
                action_sequence=current_node.action_sequence + [action],
                depth=current_node.depth + 1,
            )
            successor_node.g = new_g
            successor_node.h = new_h
            successor_node.f = new_f

            # Print the current board state
            print(render_board(successor_node.board, target, ansi=True))

            # Solution found
            if successor_node.board.get(Coord(target.r, target.c)) is None:
                solution = []
                for piece in successor_node.action_sequence:
                    pieces = []
                    # Format the Coord values into array of PlaceActions format
                    for coord in piece:
                        pieces.append(Coord(coord[0], coord[1]))
                    solution.append(PlaceAction(*pieces))
                return solution

            # If the cell is not in the open list
            if successor_node not in open_list:
                # Add the cell to the open list
                heapq.heappush(open_list, successor_node)

    # No solution found
    return None


def num_empty(board: dict[Coord, PlayerColor], target: Coord) -> int:
    """
    Function to check the number of empty coordinates in a single row OR
    column, depending on which one is smaller.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: the target BLUE coordinate to remove from the board.

    Returns:
        An integer number of the number of empty coordinates in a single row
        OR column, depending on which is smaller.
    """

    num_col = 0
    # Count empty coordinates in target column
    for i in range(COL):
        if board.get(Coord(i, target.c)) is None:
            num_col += 1

    num_row = 0
    # Count empty coordinates in target row
    for i in range(ROW):
        if board.get(Coord(target.r, i)) is None:
            num_row += 1

    # Return the minimum number between the target row or column
    return min(num_col, num_row)


def manhattan_distance(board: dict[Coord, PlayerColor], target: Coord) -> int:
    """
    Function to calculate the Manhattan Distance of the closest RED coordinate
    to the target BLUE coordinate

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: the target BLUE coordinate to remove from the board.

    Returns:
        An integer number of the Manhattan distance
    """

    # Find all RED coordinate and its respective Manhattan distance
    value = {
        i: abs(i.r - target.r) + abs(i.c - target.c)
        for i in board
        if board[i] == PlayerColor.RED
    }

    # Return the Manhattan distance value of the closest RED coordinate
    return value[min(value, key=value.get)]


def count_adjacent(board: dict[Coord, PlayerColor], target) -> int:
    """
    Function to calculate the number of adjacent BLUE or RED coordinates in a single row OR column

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: the target BLUE coordinate to remove from the board.

    Returns:
        An integer number of the Manhattan distance
    """

    count_row = 0
    for row in range(ROW):
        if (
            row + 1 < ROW
            and board.get(Coord(r=row, c=target.c)) is not None
            and board.get(Coord(r=row + 1, c=target.c)) is not None
        ):
            count_row += 1

    count_col = 0
    for col in range(COL):
        if (
            col + 1 < COL
            and board.get(Coord(r=target.r, c=col)) is not None
            and board.get(Coord(r=target.r, c=col + 1)) is not None
        ):
            count_col += 1

    return max(count_row + 1, count_col + 1)


def calculate_h(board: dict[Coord, PlayerColor], target: Coord) -> int:
    """
    Function to calculate the heuristic h(n) of a node, which is the
    estimated cost to goal from n. In this problem, we took into account
    three heuristics: the number of empty blocks in the target row or column,
    the number of blocks that are alone and not adjacent to each other, and
    the Manhattan distance from the closest RED coordinate to the target BLUE
    coordinate. Because ideally, we want the number of blocks to be adjacent to
    each other so that the target can be eliminated as efficiently as
    possible.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `target`: the target BLUE coordinate to remove from the board.

    Returns:
        An integer number of the heuristic value
    """

    # Calculate the number of individual blocks not adjacent to other blocks in
    # the target row or column
    num_alone = ROW - num_empty(board, target) - count_adjacent(board, target)

    # Here, we prioritise the number of individual blocks more and thus multiply
    # it with a higher weight of 10
    return (
        num_empty(board, target)
        + num_alone * 10
        + manhattan_distance(board, target) / 5
    )


def bound(coord: tuple) -> tuple:
    """
    Function to bound the coordinates that go off the grid.

    Parameters:
        `coord`: a tuple containing the x and y values of the coordinate to bound.

    Returns:
        A tuple of the bounded x and y coordinates
    """

    return (coord[0] % ROW, coord[1] % COL)


def generate_around(board: dict[Coord, PlayerColor], closest: Coord) -> list[Coord]:
    """
    Function to generate the possible empty coordinates around a single RED coordinate
    where it could be possible to insert a tetrimino. Because the maximum length of a
    tetrimino is 4 blocks, we search up to a radius of 4 from the specified RED
    coordinate.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `closest`: the specified RED coordinate to expand and search for empty
            coordinates.

    Returns:
        A list of possible empty Coord coordinates around the specified RED coordinate
    """

    # Initialize the empty list
    empty = []

    # Search from the left to right of the specified RED coordinate
    for x in range(closest.r - RADIUS, closest.r + RADIUS + 1):
        # Search from the top to bottom of the specified RED coordinate
        for y in range(closest.c - RADIUS, closest.c + RADIUS + 1):
            # Ensure that the generated coordinates to not exceed the maximum radius
            distance = abs(x - closest.r) + abs(y - closest.c)

            # Bound the coordinate and convert it to a Coord data type if it exceeds
            # the grid
            coord = Coord(x % 11, y % 11)

            # Check that the distance is within radius, coordinate is not the current
            # RED coordinate, and that the coordinate is empy
            if (
                distance <= RADIUS
                and coord != closest
                and board.get(Coord(r=coord.r, c=coord.c)) is None
            ):
                # Add the coordinate to the empty list
                empty.append(coord)

    # Return the list of empty coordinates
    return empty


def find_empty_spots(board: dict[Coord, PlayerColor]) -> list[Coord]:
    """
    Function to find all empty coordinate spots close to all the RED coordinates
    on the grid that a possible tetrimino can be placed.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.

    Returns:
        A list of possible empty Coord coordinates around the specified RED coordinate
    """

    # Initialise the list of possible empty spots
    empty_spots = []

    for coord in board:
        # Find all RED coordinates on the board
        if board[coord] == PlayerColor.RED:
            # For each coordinated generated
            for empty in generate_around(board, Coord(coord.r, coord.c)):
                # Check if generated coordinate is inside empty_spots
                if empty not in empty_spots:
                    empty_spots.append(empty)

    return empty_spots


def check_valid(board: dict[Coord, PlayerColor], coord: tuple) -> bool:
    """
    Function to check whether a place tetrimino is valid in the sense that it
    should have a neighbouring red note adjacent to any one of the four pieces
    of the tetrimino.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `coord`: a tuple containing the x and y values of the coordinate to bound.

    Returns:
        A boolean to determine if the placed tetrimino is valid.
    """

    # Bound the adjacent coordinates in case they go off the grid
    pieces = list(
        map(
            bound,
            [
                (coord[0] + 1, coord[1]),
                (coord[0] - 1, coord[1]),
                (coord[0], coord[1] + 1),
                (coord[0], coord[1] - 1),
            ],
        )
    )

    # Check if each adjacent piece has a corresponding RED coordinate
    for piece in pieces:
        if board.get(Coord(r=piece[0], c=piece[1])) == PlayerColor.RED:
            return True

    return False


def try_place(board: dict[Coord, PlayerColor], pieces: list[tuple[Coord]]) -> bool:
    """
    Function to check whether a place tetrimino is valid in the sense that it
    should have a neighbouring red note adjacent to any one of the four pieces
    of the tetrimino.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `pieces`: a list of pieces where each piece contains 4 Coord coordinates
            representing a possible tetrimino

    Returns:
        A boolean to determine if the placed tetrimino is valid.
    """

    # Initialise the number of adjacent pieces to a tetrimino
    adjacent = 0

    for coord in pieces:
        # Check if coordinate is already occupied
        if board.get(Coord(r=coord[0], c=coord[1])):
            return False
        # Check if it any of the tetrimino piece has adjacent red tile
        if check_valid(board, coord):
            adjacent += 1

    return adjacent > 0


def check_clear(board: dict[Coord, PlayerColor]):
    """
    Function to evaluate whether a row or a column in the grid can be cleared when
    an entire row or column is occupied. If it is occupied, it can be cleared
    by setting it to None.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
    """

    # Check rows
    for row in range(ROW):
        # If an entire row is occupied
        if all(board.get(Coord(r=row, c=col)) for col in range(COL)):
            # Set each coordinate of the corresponding row to None
            for i in range(ROW):
                board[Coord(r=row, c=i)] = None

    # Check rows
    for col in range(ROW):
        # If an entire column is occupied
        if all(board.get(Coord(r=row, c=col)) for row in range(COL)):
            # Set each coordinate of the corresponding row to None
            for i in range(COL):
                board[Coord(r=i, c=col)] = None


def apply_piece(board: dict[Coord, PlayerColor], piece: list[tuple[Coord]]):
    """
    Function to place a tetrimino onto the board.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.
        `piece`: a list of coordinates where each piece contains four Coord
            coordinates representing a possible tetrimino.

    Returns:
        A dictionary of the board state after placing the tetrimino.
    """

    # First copy the existing board to a new board
    new_board = board.copy()

    # Place each piece of the terimino onto the new board
    for coord in piece:
        new_board[Coord(coord[0], coord[1])] = PlayerColor.RED

    # Check whether the newly placed tetrimino can clear the board
    check_clear(new_board)

    # Return the state of the new board
    return new_board


def generate_successors(
    board: dict[Coord, PlayerColor]
) -> Generator[dict[Coord, PlayerColor], list[Coord], None]:
    """
    Function to generate the successor node states.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to "player colours". The keys are `Coord` instances,
            and the values are `PlayerColor` instances.

    Returns:
        A dictionary of the board state after generating the successor nodes.
    """

    # Find empty spots
    empty_spots = find_empty_spots(board)
    for spot in empty_spots:
        # Generate pieces by passing spot as a list of Coord instances
        pieces = generate_pieces([spot])
        for piece in pieces:
            # Check for valid placements
            if try_place(board, piece):
                new_board = apply_piece(board, piece)
                yield new_board, piece


def generate_pieces(coord: list[Coord]) -> list[list[Coord]]:
    """
    Function to generate each of the 19 possible "fixed" tetrimino variations.

    Parameters:
        `coord`: a list of Coord coordinates that are empty and serve as the
            head of the tetrimino.

    Returns:
        A list of the possible tetrimino variations as four Coord coordinates.
    """

    pieces = []
    for head in coord:
        # Ensure head is a Coord instance
        assert isinstance(head, Coord)
        # I1
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c + 1),
                        (head.r, head.c + 2),
                        (head.r, head.c + 3),
                    ],
                )
            )
        )
        # I2
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r - 1, head.c),
                        (head.r - 2, head.c),
                        (head.r - 3, head.c),
                    ],
                )
            )
        )
        # O
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r + 1, head.c),
                        (head.r, head.c + 1),
                        (head.r + 1, head.c + 1),
                    ],
                )
            )
        )
        # T1
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c - 1),
                        (head.r + 1, head.c - 1),
                        (head.r, head.c - 2),
                    ],
                )
            )
        )
        # T2
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r - 1, head.c),
                        (head.r - 2, head.c),
                        (head.r - 1, head.c - 1),
                    ],
                )
            )
        )
        # T3
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c + 1),
                        (head.r - 1, head.c + 1),
                        (head.r, head.c + 2),
                    ],
                )
            )
        )
        # T4
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r + 1, head.c),
                        (head.r + 2, head.c),
                        (head.r + 1, head.c + 1),
                    ],
                )
            )
        )
        # J1
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r + 1, head.c),
                        (head.r + 2, head.c),
                        (head.r + 1, head.c - 1),
                    ],
                )
            )
        )
        # J2
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c - 1),
                        (head.r, head.c - 2),
                        (head.r - 1, head.c - 2),
                    ],
                )
            )
        )
        # J3
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r - 1, head.c),
                        (head.r - 2, head.c),
                        (head.r - 2, head.c + 1),
                    ],
                )
            )
        )
        # J4
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c + 1),
                        (head.r, head.c + 2),
                        (head.r + 1, head.c + 2),
                    ],
                )
            )
        )
        # L1
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r + 1, head.c),
                        (head.r + 2, head.c),
                        (head.r + 2, head.c + 1),
                    ],
                )
            )
        )
        # L2
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c - 1),
                        (head.r, head.c - 2),
                        (head.r + 1, head.c - 2),
                    ],
                )
            )
        )
        # L3
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r - 1, head.c),
                        (head.r - 2, head.c),
                        (head.r - 2, head.c - 1),
                    ],
                )
            )
        )
        # L4
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c + 1),
                        (head.r, head.c + 2),
                        (head.r - 1, head.c + 2),
                    ],
                )
            )
        )
        # Z1
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c + 1),
                        (head.r + 1, head.c + 1),
                        (head.r + 1, head.c + 2),
                    ],
                )
            )
        )
        # Z2
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r - 1, head.c),
                        (head.r - 1, head.c + 1),
                        (head.r - 2, head.c + 1),
                    ],
                )
            )
        )
        # S1
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r, head.c + 1),
                        (head.r - 1, head.c + 1),
                        (head.r - 1, head.c + 2),
                    ],
                )
            )
        )
        # S2
        pieces.append(
            list(
                map(
                    bound,
                    [
                        (head.r, head.c),
                        (head.r - 1, head.c),
                        (head.r - 1, head.c - 1),
                        (head.r - 2, head.c - 1),
                    ],
                )
            )
        )

    # Return a list of all the possible tetrimino places
    return pieces