import random
import os
import time

'''
The entire board                   -> 64-bit integer
A sing line or colunm in the board -> 16-bit integer
A single tile                      -> 4-bit integer
'''


class _2048:    
    def step(self, move):        
        grid, changed = moveGrid(self.grid, move)                
        done = not getLegalMoves(grid)          
        reward = 1
        self.score += reward
        self.grid = grid
        return grid, reward, done, None
    
    def reset(self):
        self.grid = generateGrid()
        self.score = 0
        self.time_log = time.time()
        return self.grid

    def render(self):
        os.system("cls")
        board = grid2Board(self.grid)
        print("+-----+-----+-----+-----+")
        for row in range(4):
            for tile in board[row]:
                print(f"|{tile:<5}", end = "")
            print("|")
            print("+-----+-----+-----+-----+")
        spending_time = time.time() - self.time_log
        self.time_log = time.time()
        print(spending_time)

    def close(self):
        pass

# 60 - 4x - 16y
shift_tile_mask = [[60, 56, 52, 48], [44, 40, 36, 32], [28, 24, 20, 16], [12, 8, 4, 0]]

shift_row_mask = [48, 32, 16, 0]



def getTile(grid, row, col):
    return (grid >> shift_tile_mask[row][col]) & 15


def setTile(grid, row, col, value):
    shift_length = shift_tile_mask[row][col]    
    grid &= ~(15 << shift_length) # remove
    grid |= value << shift_length # input
    return grid

def getRow(grid, row):
    return (grid >> shift_row_mask[row]) & 65535

def setRow(grid, row, row_value):
    shift_length = shift_row_mask[row]
    grid &= ~(65535 << shift_length) # remove
    grid |= row_value << shift_length # input
    return grid

def getCol(grid, col):
    col_value = getTile(grid, 0, col) << 12 |\
        getTile(grid, 1, col) << 8 |\
        getTile(grid, 2, col) << 4 |\
        getTile(grid, 3, col)
    return col_value

def setCol(grid, col, col_value):
    grid = setTile(grid, 0, col, col_value >> 12 & 15)
    grid = setTile(grid, 1, col, col_value >> 8 & 15)
    grid = setTile(grid, 2, col, col_value >> 4 & 15)
    grid = setTile(grid, 3, col, col_value & 15)
    return grid
            

def generateGrid():
    grid = 0
    for _ in range(2):
        grid = spawnTile(grid)
    return grid


def spawnTile(grid=int()):
    free_tiles = getFreeTile(grid)
    if not free_tiles:
        return grid

    row, col = random.choice(free_tiles)
    if random.randint(0, 9) < 9:
        grid = setTile(grid, row, col, 1)
    else:
        grid = setTile(grid, row, col, 2)
    return grid


def getFreeTile(grid=int()):
    # binary is 64 bit
    free_cells = []
    for row in range(4):
        for col in range(4):
            tile_num = getTile(grid, row, col)
            if tile_num == 0:
                free_cells.append([row, col])
    return free_cells


def grid2Board(grid=int()):
    board = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    for row in range(4):
        for col in range(4):
            tile_num = getTile(grid, row, col)
            tile = 2 ** tile_num if tile_num > 0 else 0
            board[row][col] = tile
    return board


def trans(n, m):
    # merge
    if n == m and n != 0 and n < 15:
        return 0, n + 1, True
    # move
    if m == 0:
        return 0, n, False
    # noting
    return n, m, False


def transRightDown(a, b, c, d):
    c, d, m1 = trans(c, d)
    b, c, m2 = trans(b, c)
    if not (m1 or m2):
        c, d, m1 = trans(c, d)

    a, b, m3 = trans(a, b)

    if not (m2 or m3):
        b, c, _ = trans(b, c)

    if not (m1 or m2):
        c, d, _ = trans(c, d)
    return a, b, c, d

def transLeftUp(a, b, c, d):
    b, a, m1 = trans(b, a)
    c, b, m2 = trans(c, b)
    if not (m1 or m2):
        b, a, m1 = trans(b, a)

    d, c, m3 = trans(d, c)

    if not (m2 or m3):
        c, b, _ = trans(c, b)

    if not (m1 or m2):
        b, a, _ = trans(b, a)    
    return a, b, c, d

# grid = generate_grid()
# grid2board(grid)

def encodeUint16(a, b, c, d):
    return a << 12 | b << 8 |  c << 4 | d

def decodeUint16(val):
    a = val >> 12 & 15
    b = val >> 8 & 15
    c = val >> 4 & 15
    d = val & 15    
    return a, b, c, d


def prepareTrans(file_name = "move_transition.txt"):
    try:
        if os.stat(file_name).st_size == 876026:
            with open(file_name, "r") as file:
                data = file.read().split("@")
                moveRightDown = eval(data[0])
                moveLeftUp = eval(data[1])                
            return moveRightDown, moveLeftUp
    except:
        pass

    def addTrans(trans, t):
        trans.append(encodeUint16(t[0], t[1], t[2], t[3]))

    nibbleMax = 16

    moveRightDown = []
    moveLeftUp = []

    for a in range(nibbleMax):
        for b in range(nibbleMax):
            for c in range(nibbleMax):
                for d in range(nibbleMax):
                    addTrans(moveRightDown, transRightDown(a, b, c, d))
                    addTrans(moveLeftUp, transLeftUp(a, b, c, d))

    # Generate the file transition.go with the 2 slices.
    with open(file_name, "w") as file:
        file.write(str(moveRightDown))
        file.write("@")
        file.write(str(moveLeftUp))
    return moveRightDown, moveLeftUp

moveRightDown, moveLeftUp = prepareTrans()


def moveGrid(grid, move):
    '''
    0 -> Left
    1 -> Up
    2 -> Right
    3 -> Down
    '''
    origin = grid
    if move == 0:
        for row in range(4):
            line = getRow(grid, row)
            changed_line = moveLeftUp[line]
            grid = setRow(grid, row, changed_line)
        
    elif move == 1:
        for col in range(4):
            line = getCol(grid, col)
            changed_line = moveLeftUp[line]
            grid = setCol(grid, col, changed_line)
    
    elif move == 2:
        for row in range(4):
            line = getRow(grid, row)
            changed_line = moveRightDown[line]
            grid = setRow(grid, row, changed_line)

    elif move == 3:
        for col in range(4):
            line = getCol(grid, col)
            changed_line = moveRightDown[line]
            grid = setCol(grid, col, changed_line)
          
    moved = origin != grid    
    grid = spawnTile(grid)
    return grid, moved

def moveForHeuristic(grid, move):
    """
    The logic of game is same, but this function gives tile difference.
    """    
    difference = 0
    if move == 0:
        for row in range(4):
            line = getRow(grid, row)
            changed_line = moveLeftUp[line]
            for old_v, new_v in zip(decodeUint16(changed_line),decodeUint16(line)):
                difference += abs(new_v - old_v)
            grid = setRow(grid, row, changed_line)
        
    elif move == 1:
        for col in range(4):
            line = getCol(grid, col)
            changed_line = moveLeftUp[line]
            for old_v, new_v in zip(decodeUint16(changed_line),decodeUint16(line)):
                difference += abs(new_v - old_v)
            grid = setCol(grid, col, changed_line)
    
    elif move == 2:
        for row in range(4):
            line = getRow(grid, row)
            changed_line = moveRightDown[line]
            for old_v, new_v in zip(decodeUint16(changed_line),decodeUint16(line)):
                difference += abs(new_v - old_v)
            grid = setRow(grid, row, changed_line)

    elif move == 3:
        for col in range(4):
            line = getCol(grid, col)
            changed_line = moveRightDown[line]
            for old_v, new_v in zip(decodeUint16(changed_line),decodeUint16(line)):
                difference += abs(new_v - old_v)
            grid = setCol(grid, col, changed_line)
              
    if difference:
        grid = spawnTile(grid)
    return grid, difference

def getLegalMoves(grid):
    moves = []
    for move in range(4):
        _, changed = moveGrid(grid, move)
        if changed:
            moves.append(move)
    return moves

def isEnd(grid):    
    moves = getLegalMoves(grid)
    return not moves

def view(grid):
    for v in grid2Board(grid):
        print(v)
    print("")




if __name__ == "__main__":
    env = _2048()
    retry = True
    while retry:        
        done = False
        score = 0
        grid = env.reset()
        while not done:
            env.render()
            
            try:
                action = input("\n2,4,8,6 입력하세요 (하, 좌, 상, 우)\n")
                action = {"4":0, "8":1, "6":2, "2":3, "5":3}[action]
            except KeyError:
                action = input("\n2,4,8,6 입력하세요 (하, 좌, 상, 우)\n")
                action = {"4":0, "8":1, "6":2, "2":3, "5":3}[action]
            grid, reward, done, info = env.step(action)            
            score += reward        
        response = input(f"게임 종료! [{score}]점\n계속하시겠습니까? (y/n)\n")
        if response == "n":
            retry = False
        