import chardet

def path_finder(input) -> str:
    """
        This function reads a file containing information about a pipe system, constructs
        a 2D list which acts as a representation of the pipe system, and performs a 
        depth-first search to find all sinks connected to the source ('*').

        Args:
            input (str): The path to the input file containing pipe network information.

        Returns:
            str: A string containing all connected sinks in alphabetical order.

        Raises:
            ValueError: If the input file contains invalid coordinates or doesn't meet
                        the specified format.

    """
    # To detect encoding. 'rb' flag used to read this as a binary file.
    with open(input, 'rb') as file:
        # Don't need to read the whole input text to identify the encoding.
        buffer_size = 10*3
        text_chunk = file.read(buffer_size)
    encoding_format=chardet.detect(text_chunk).get("encoding")

    line_contents = []
    max_x, max_y = 0, 0

    with open(input, encoding=encoding_format) as file:
        for line in file:
            # Splitting on whitespaces.
            character, str_x, str_y = line.split(" ")
            try:
                x = int(str_x)
                y = int(str_y)
            except ValueError as e:
                raise ValueError("Could not parse coordinates - Input file did not meet provided specifications.")
            if x < 0 or y < 0:
                raise ValueError("Coordinates below 0 were identified - please check if the input file met the provided specifications.")
            line_contents.append((character, x, y))
            max_x = max(x, max_x)
            max_y = max(y, max_y)

    # Initializing grid.
    # The inner elements are x-coordinates, outer elements are y-coordinates
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Now populate the grid:
    # Unpacking tuple
    for cell_value, x_cood, y_cood in line_contents:
        try:
            grid[y_cood][x_cood] = str(cell_value)
        except IndexError:
            print(f"Out of index error for coordinates: {x_cood, y_cood}")
            continue
    
    
    # Display the grid
    for col in range(max_y + 1):
        print(f"{grid[col]}")

    # Hashmap to map directions to pipes. Key - pipe, value - list of lists containing directions that we can move in.
    directions_map = {  
            '═': [[0, 1], [0, -1]],
            '║': [[1, 0], [-1, 0]],
            '╔': [[-1, -1], [-1, 1]],
            '╗': [[1, 1], [-1, -1]],
            '╚': [[1, 1], [-1, -1]],
            '╝': [[1, -1], [-1, 1]],
            '╠': [[1, 1], [-1, 1], [1, -1], [-1, -1], [1, 0], [-1, 0]],
            '╣': [[1, -1], [-1, -1], [-1, 1], [1, 1], [1, 0], [-1, 0]],
            '╦': [[1, 1], [1, -1], [-1, -1], [-1, 1], [0, 1], [0, -1]],
            '╩': [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 1], [0, -1]]
        }

    # Global variables for the DFS function:
    connected_sinks = set()
    visited = set()

    def dfs(no_col, no_row):
        """
            Performs a depth-first search on the grid.

            This function is called recursively to explore the pipe network and find
            connected sinks. It keeps track of visited cells to avoid visiting them again,
            and adds all the connected sinks to a set.

            Args:
                no_col (int): The current column (y-coordinate) in the grid.
                no_row (int): The current row (x-coordinate) in the grid.

            Global variables used:
                grid (List[List[str]]): The 2D grid representation of the pipe network.
                visited (Set[Tuple[int, int]]): A set to keep track of visited cells.
                connected_sinks (Set[str]): A set to store found sinks.
                directions_map (Dict[str, List[List[int]]]): A map of pipe types to possible directions.
            
            Returns: 
                None.
        """
        # Invalidating conditions
        if (no_col < 0 or no_col >= max_y + 1 or
            no_row < 0 or no_row >= max_x + 1 or
            (no_col, no_row) in visited or
            grid[no_col][no_row] == ' '):
            return
        
        # Mark the current cell as visited
        visited.add((no_col, no_row))
        val = grid[no_col][no_row]
        
        # Check if English character. 
        if val.isascii() and val.isalpha():
            connected_sinks.add(val)
            # Assuming sinks can be connected to other sinks only in 4 directions:
            sink_directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for directions in sink_directions:
                dy = directions[0]
                dx = directions[1]
                curr_col, curr_row = no_col + dy, no_row + dx
                dfs(curr_col, curr_row)
        
        # Check if the current cell is a pipe.
        elif val in directions_map.keys():
            all_directions = directions_map[val]
            # Iterate through all moves allowed for this pipe.
            for directions in all_directions:
                dy = directions[0]
                dx = directions[1]
                curr_col, curr_row = no_col + dy, no_row + dx
                dfs(curr_col, curr_row)
        return
    
    # Finally, run DFS starting at the source.
    for col in range(max_y + 1):
        for row in range(max_x + 1):
            if grid[col][row] == '*':
                dfs(col + 1, row)
                dfs(col - 1, row)
                dfs(col, row + 1)
                dfs(col, row - 1)

    # Need it in alphabetical order, so sort the set before extracting elements.
    return_str = ''.join(elem for elem in sorted(connected_sinks))
    return return_str

# Example usage.
input_filepath = 'input.txt'
return_val = path_finder(input_filepath)
print(f"Connected sinks: {return_val}")
