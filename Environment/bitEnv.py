import random

def generate_grid():
    grid = 0
    for _ in range(2):
        grid = spawn_new_tile(grid)
    return grid

def spawn_new_tile(grid = int()):
    free_cells = get_free_cells(grid)
    if not free_cells:
        return grid
    target_location = random.choice(free_cells)
    bit_location = 64 - target_location * 4
    if random.randint(0, 9) < 9:
        grid |= (1 << bit_location - 4)
    else:
        grid |= (1 << bit_location - 3)
    return grid


def get_free_cells(grid = int()):
    #binary is 64 bit
    binary_grid = bin(grid)[2:].zfill(64)
    free_cells = []
    for i in range(0, 64, 4):        
        if eval("0b" + binary_grid[i:i+4]) == 0:
            free_cells.append(int(i // 4))
    return free_cells

def grid2board(grid = int()):
    board = []    
    binary_grid = bin(grid)[2:].zfill(64)
    for i in range(0, 64, 4):
        power = eval("0b" + binary_grid[i:i+4])
        tile = 2 ** power if power > 0 else 0
        board.append(tile)    
    board = [board[i:i+4] for i in range(0, 16, 4)]
    return board

