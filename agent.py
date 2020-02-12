import math
import time
import random

from gameobjects import GameObject
from move import Move, Direction

from enum import Enum
from copy import deepcopy
import heapq


class Compass(Enum):
    N = 0
    NE = 1
    E = 2
    SE = 3
    S = 4
    SW = 5
    W = 6
    NW = 7


class Agent:
    redraw = True

    testing = False  # on when benchmarking, when A* does not find a path, die
    done = False
    x = 0

    scores = []
    deaths = 0

    times = []
    time_counter = 0

    sudoku = False

    board_width = 0
    board_height = 0

    food_target = None
    path = None

    # RL parameters
    RL = True  # set to true to turn on RL
    grow = False

    q_table = None
    initial_reward = 0
    explore = 0.8
    alpha = 0.6     # 0 < alpha <= 1 (learning rate)
    gamma = 0.6     # 0 < gamma = 1 (discount factor)
    idle_reward = -0.04
    die_reward = -1
    food_reward = 1
    counter = 0

    def __init__(self):
        """" Constructor of the Agent, can be used to set up variables """
        import main
        self.board_width = main.board_width
        self.board_height = main.board_height

    def get_move(self, board, score, turns_alive, turns_to_starve, direction, head_position, body_parts):
        """This function behaves as the 'brain' of the snake. You only need to change the code in this function for
        the project. Every turn the agent needs to return a move. This move will be executed by the snake. If this
        functions fails to return a valid return (see return), the snake will die (as this confuses its tiny brain
        that much that it will explode). The starting direction of the snake will be North.

        :param board: A two dimensional array representing the current state of the board. The upper left most
        coordinate is equal to (0,0) and each coordinate (x,y) can be accessed by executing board[x][y]. At each
        coordinate a GameObject is present. This can be either GameObject.EMPTY (meaning there is nothing at the
        given coordinate), GameObject.FOOD (meaning there is food at the given coordinate), GameObject.WALL (meaning
        there is a wall at the given coordinate. TIP: do not run into them), GameObject.SNAKE_HEAD (meaning the head
        of the snake is located there) and GameObject.SNAKE_BODY (meaning there is a body part of the snake there.
        TIP: also, do not run into these). The snake will also die when it tries to escape the board (moving out of
        the boundaries of the array)

        :param score: The current score as an integer. Whenever the snake eats, the score will be increased by one.
        When the snake tragically dies (i.e. by running its head into a wall) the score will be reset. In ohter
        words, the score describes the score of the current (alive) worm.

        :param turns_alive: The number of turns (as integer) the current snake is alive.

        :param turns_to_starve: The number of turns left alive (as integer) if the snake does not eat. If this number
        reaches 1 and there is not eaten the next turn, the snake dies. If the value is equal to -1, then the option
        is not enabled and the snake can not starve.

        :param direction: The direction the snake is currently facing. This can be either Direction.NORTH,
        Direction.SOUTH, Direction.WEST, Direction.EAST. For instance, when the snake is facing east and a move
        straight is returned, the snake wil move one cell to the right.

        :param head_position: (x,y) of the head of the snake. The following should always hold: board[head_position[
        0]][head_position[1]] == GameObject.SNAKE_HEAD.

        :param body_parts: the array of the locations of the body parts of the snake. The last element of this array
        represents the tail and the first element represents the body part directly following the head of the snake.

        :return: The move of the snake. This can be either Move.LEFT (meaning going left), Move.STRAIGHT (meaning
        going straight ahead) and Move.RIGHT (meaning going right). The moves are made from the viewpoint of the
        snake. This means the snake keeps track of the direction it is facing (North, South, West and East).
        Move.LEFT and Move.RIGHT changes the direction of the snake. In example, if the snake is facing north and the
        move left is made, the snake will go one block to the left and change its direction to west.
        """

        if self.done:
            return Move.STRAIGHT

        if self.RL:
            return self.q_learning(board, score, turns_alive, turns_to_starve, direction, head_position, body_parts)

        find_tail = False



        # A* search
        if self.path is None:
            self.path = self.a_star_search(board, head_position, body_parts)
        if self.path is None and len(body_parts) > 1:
            self.path = self.a_star_search_algorithm(board, head_position, body_parts, body_parts[-1], True)
            find_tail = True
        if self.path is None:
            # self.done = True
            #print("Could not find a path")
            # insert path algorithm to stay alive as long as possible until path is found
            move = self.get_undecided_safe_move(board, direction, head_position)
            if move is None:
                #print("WTF: Could not find a safe move anymore")
                move = Move.STRAIGHT
                # x = input()
            return move

        # Follow the path given by A*
        move = None
        n = self.path.pop(-1)

        if find_tail:
            self.path = None

        if direction == Direction.NORTH:
            if head_position[0] != n[0]:
                if n[0] - head_position[0] == -1:
                    move = Move.LEFT
                elif n[0] - head_position[0] == 1:
                    move = Move.RIGHT
            elif head_position[1] != n[1]:
                if n[1] - head_position[1] == -1:
                    move = Move.STRAIGHT

        if direction == Direction.EAST:
            if head_position[0] != n[0]:
                if n[0] - head_position[0] == 1:
                    move = Move.STRAIGHT
            elif head_position[1] != n[1]:
                if n[1] - head_position[1] == -1:
                    move = Move.LEFT
                elif n[1] - head_position[1] == 1:
                    move = Move.RIGHT

        if direction == Direction.SOUTH:
            if head_position[0] != n[0]:
                if n[0] - head_position[0] == -1:
                    move = Move.RIGHT
                elif n[0] - head_position[0] == 1:
                    move = Move.LEFT
            elif head_position[1] != n[1]:
                if n[1] - head_position[1] == 1:
                    move = Move.STRAIGHT

        if direction == Direction.WEST:
            if head_position[0] != n[0]:
                if n[0] - head_position[0] == -1:
                    move = Move.STRAIGHT
            elif head_position[1] != n[1]:
                if n[1] - head_position[1] == -1:
                    move = Move.RIGHT
                elif n[1] - head_position[1] == 1:
                    move = Move.LEFT

        if move is None:
            # When the algorithm wants the snake to go down (because it does not have tailpieces yet)
            # Go left or right, depending on which is closer to the food node
            move = self.get_safe_move_closest_to_food(board, direction, head_position)

            if move is None:
                move = Move.STRAIGHT
                #print("Died because of walls?")
                #x = input()
            # Did not follow path, reset path
            self.path = None
        return move

    def q_learning(self, board, score, turns_alive, turns_to_starve, direction, head_position, body_parts):

        if self.q_table is None:
            # initialize q table
            # TODO
            # self.q_table = [[[self.initial_reward for x in range(self.board_height*self.board_width)] for y in range(4)] for z in range(self.board_height*self.board_width)]
            self.q_table = [
                [[self.initial_reward for x in range(self.board_height * self.board_width)] for y in range(4)] for z in
                range(8)]

        if self.food_target is None:
            for x in range(self.board_width):
                for y in range(self.board_height):
                    if board[x][y] == GameObject.FOOD:
                        self.food_target = 5*x + y

        # initial list of possible directions
        directions = None
        if direction == Direction.NORTH:
            directions = [(Move.STRAIGHT, (0, -1)), (Move.RIGHT, (1, 0)), (Move.LEFT, (-1, 0))]
        elif direction == Direction.EAST:
            directions = [(Move.STRAIGHT, (1, 0)), (Move.RIGHT, (0, 1)), (Move.LEFT, (0, -1))]
        elif direction == Direction.SOUTH:
            directions = [(Move.STRAIGHT, (0, 1)), (Move.RIGHT, (-1, 0)), (Move.LEFT, (1, 0))]
        elif direction == Direction.WEST:
            directions = [(Move.STRAIGHT, (-1, 0)), (Move.RIGHT, (0, -1)), (Move.LEFT, (0, 1))]

        x = int(math.floor(self.food_target/self.board_width))
        y = self.food_target % self.board_width

        compass = None
        if x == head_position[0] and y >= head_position[1]:
            compass = Compass.S
        elif x == head_position[0] and y <= head_position[1]:
            compass = Compass.N
        elif y == head_position[1] and x >= head_position[0]:
            compass = Compass.E
        elif y == head_position[1] and x <= head_position[0]:
            compass = Compass.W
        elif x >= head_position[0] and y >= head_position[1]:
            compass = Compass.SE
        elif x >= head_position[0] and y <= head_position[1]:
            compass = Compass.NE
        elif x <= head_position[0] and y >= head_position[1]:
            compass = Compass.SW
        elif x <= head_position[0] and y <= head_position[1]:
            compass = Compass.NW
        # print(x)
        # print(y)
        # print(head_position)
        # print(compass)

        # Choose random move weighted towards the best move
        moves = []
        for m in directions:
            # if ((head_position[0] + m[1][0]) < 0) or ((head_position[1] + m[1][1]) < 0):
            #     # default value for dying out of bounds
            #     moves.append((m, self.die_reward))
            # elif ((head_position[0] + m[1][0]) >= self.board_width) or ((head_position[1] + m[1][1]) >= self.board_width):
            #     moves.append((m, self.die_reward))
            #else:
            xa = (direction.value + m[0].value) % 4
            ya = self.board_height * (head_position[0]) + head_position[1]
            # TODO
            #moves.append((m, self.q_table[self.food_target][xa][ya]))
            moves.append((m, self.q_table[compass.value][xa][ya]))

        random_number = random.random()
        best_move = None
        for m in moves:
            if best_move is None:
                best_move = m
            else:
                if m[1] > best_move[1]:
                    best_move = m
        moves.remove(best_move)
        if random_number <= self.explore:
            move = best_move
            q = "Best Move"
        elif random_number <= self.explore + float((1-self.explore))/2:
            move = moves.pop(-1)
            q = "Explore Move"
        else:
            move = moves.pop(0)
            q = "Explore Move"
        # update q_table
        if (head_position[0] + move[0][1][0] < 0) or (head_position[0] + move[0][1][0] >= self.board_width):
            value = self.die_reward
        elif (head_position[1] + move[0][1][1] < 0) or (head_position[1] + move[0][1][1] >= self.board_width):
            value = self.die_reward
        elif board[head_position[0] + move[0][1][0]][head_position[1] + move[0][1][1]] == GameObject.WALL:
            value = self.die_reward
        elif board[head_position[0] + move[0][1][0]][head_position[1] + move[0][1][1]] == GameObject.FOOD:
            value = self.food_reward
        else:
            value = self.idle_reward

        self.bellman(5 * head_position[0] + head_position[1], (direction.value + move[0][0].value) % 4, value, 5*move[0][1][0] + move[0][1][1], compass.value)
        # print("Direction: " + str(direction) + "\t" + str(move[0][0]))
        # print(q)
        # print(value)
        # for x in self.q_table:
        #     print(x)
        return move[0][0]

    def bellman(self, pos, direction, value, next, compass):
        # if self.explore < 1:
        #     self.explore += 0.001
        # else:
        #     print("Pick only best moves")
            #print(self.counter)
        if self.counter >= 1000: #self.board_height*self.board_width * (self.board_width*self.board_height*4)**2 * (1-self.explore):
            print("Only pick best moves now")
            self.explore = 1
        max_q = []
        if pos + next >= 0 and pos+next < 25:
            for i in range(4):
                # TODO
                # max_q.append(self.q_table[self.food_target][i][pos+next])
                max_q.append(self.q_table[compass][i][pos+next])
        else:
            max_q.append(self.die_reward)
        # TODO
        # self.q_table[self.food_target][direction][pos] = self.q_table[self.food_target][direction][pos] + self.alpha * (value + self.gamma * max(max_q) - self.q_table[self.food_target][direction][pos])
        self.q_table[compass][direction][pos] = self.q_table[compass][direction][pos] + self.alpha * (
                    value + self.gamma * max(max_q) - self.q_table[compass][direction][pos])
        if self.counter < 1000:
            self.counter += 1
            self.redraw = True
        else:
            self.redraw = True
        return True

    def a_star_search(self, board, head_position, body_parts):
        food_array = []
        for x in range(self.board_width):
            for y in range(self.board_height):
                if board[x][y] == GameObject.FOOD:
                    food_array.append((x, y))
        return self.get_food_target(food_array, board, head_position, body_parts)

    def a_star_search_algorithm(self, board, head_position, body_parts, food, look_for_tail=False):
        # self.x += 1
        # start = time.time()

        # A* algorithm
        frontier = []
        heapq.heappush(frontier, (0, head_position))
        came_from = {head_position: None}
        cost_so_far = {head_position: 0}
        found = False

        while len(frontier) != 0:

            board_copy = deepcopy(board)
            body_parts_copy = deepcopy(body_parts)

            current = heapq.heappop(frontier)[1]
            # if self.RL:
            #     current *= -1

            if self.grow:
                negligible_body_parts = []

                for i in range(0, self.get_moves_amount(came_from, current, head_position)):
                    if len(body_parts) != 0:
                        if len(body_parts_copy) != 0:
                            negligible_body_parts.append(body_parts_copy[-1])
                            del body_parts_copy[-1]

                for parts in negligible_body_parts:
                    board_copy[parts[0]][parts[1]] = GameObject.EMPTY

            # If the snake would die after it would eat the food
            # Make undecided moves until it will not
            # Snake will commit sudoku if game is has no other options left
            # With multiple food nodes, snake will sudoku like a dumbass
            if not look_for_tail:
                neighbors = self.get_neighbors(board_copy, food, True)
                found_head = False
                for n in neighbors:
                    if n == head_position and len(neighbors) > 1:
                        found_head = True
                if len(neighbors) <= 1 and not found_head:
                    if not self.sudoku:
                        #print("dayum")
                        return None

            if current == food:
                found = True
                break

            neighbors = self.get_neighbors(board_copy, current)

            for n in neighbors:
                # Cost to travel between nodes is always 1
                # Heuristic is the, possibly, diagonal distance (Pythagoras)
                # if self.RL:
                #     new_cost = cost_so_far[current] + self.q_table[current[0]][current[1]]
                # else:
                new_cost = cost_so_far[current] + 1

                # if self.RL:
                #     if n not in cost_so_far or new_cost > cost_so_far[n]:
                #         cost_so_far[n] = new_cost
                #         # Pythagoras distance
                #         # priority = cost_so_far[n] + math.sqrt(
                #         #     abs(n[0] - food[0]) ** 2 + abs(n[1] - food[1]) ** 2)
                #         # Manhattan distance
                #         priority = cost_so_far[n] + abs(n[0] - food[0]) + abs(n[1] - food[1])
                #         heapq.heappush(frontier, (-1 * priority, n))
                #         came_from[n] = current
                #else:
                if n not in cost_so_far or new_cost < cost_so_far[n]:
                    cost_so_far[n] = new_cost
                    # Pythagoras distance
                    # priority = cost_so_far[n] + math.sqrt(
                    #     abs(n[0] - food[0]) ** 2 + abs(n[1] - food[1]) ** 2)
                    # Manhattan distance
                    priority = cost_so_far[n] + abs(n[0]-food[0]) + abs(n[1]-food[1])
                    heapq.heappush(frontier, (priority, n))
                    came_from[n] = current

        path = None
        if found:
            current = food
            path = []
            while current != head_position:
                path.append(current)
                current = came_from[current]

        # end = time.time()
        # self.times.append(end - start)
        # self.time_counter += 1
        # if self.x >= 30:
        #     self.done = True

        return path

    @staticmethod
    def get_moves_amount(came_from, current, head_position):
        i = 0
        while current is not head_position:
            i += 1
            current = came_from[current]
        return i

    def get_food_target(self, foods, board, head_position, body_parts):
        distance = math.inf
        best_path = None
        best_food = None
        for f in foods:
            path = self.a_star_search_algorithm(board, head_position, body_parts, f)
            if path is not None:
                if len(path) < distance:
                    distance = len(path)
                    best_path = path
                    best_food = f
        self.food_target = best_food
        return best_path

    def get_neighbors(self, board, current, sudoku=False):
        walls = 0
        neighbors = []
        left = True
        right = True
        up = True
        down = True
        if current[0] == 0:
            left = False
        if current[0] == self.board_width - 1:
            right = False
        if current[1] == 0:
            up = False
        if current[1] == self.board_height - 1:
            down = False

        if up:
            if board[current[0]][current[1] - 1] == GameObject.EMPTY or \
                    board[current[0]][current[1] - 1] == GameObject.FOOD:
                neighbors.append((current[0], current[1] - 1))
            if board[current[0]][current[1] - 1] == GameObject.WALL:
                walls += 1
        if left:
            if board[current[0] - 1][current[1]] == GameObject.EMPTY or \
                    board[current[0] - 1][current[1]] == GameObject.FOOD:
                neighbors.append((current[0] - 1, current[1]))
            if board[current[0] - 1][current[1]] == GameObject.WALL:
                walls += 1
        if down:
            if board[current[0]][current[1] + 1] == GameObject.EMPTY or \
                    board[current[0]][current[1] + 1] == GameObject.FOOD:
                neighbors.append((current[0], current[1] + 1))
            if board[current[0]][current[1] + 1] == GameObject.WALL:
                walls += 1
        if right:
            if board[current[0] + 1][current[1]] == GameObject.EMPTY or \
                    board[current[0] + 1][current[1]] == GameObject.FOOD:
                neighbors.append((current[0] + 1, current[1]))
            if board[current[0] + 1][current[1]] == GameObject.WALL:
                walls += 1

        # if snake would die after eating food, commit sudoku
        if sudoku:
            directions = 0
            if up:
                directions += 1
            if down:
                directions += 1
            if left:
                directions += 1
            if right:
                directions += 1
            if directions - walls <= 1:
                #print("COMMIT SUDOKU")
                self.sudoku = True
        return neighbors

    def get_safe_move_closest_to_food(self, board, direction, head_position):
        move = None

        if direction == Direction.NORTH:
            distance = math.inf
            if head_position[0] != self.board_width - 1:
                if board[head_position[0] + 1][head_position[1]] == GameObject.EMPTY:
                    move = Move.RIGHT
                    distance = abs(head_position[0] + 1 - self.food_target[0]) + abs(
                        head_position[1] - self.food_target[1])
            if head_position[0] != 0:
                if board[head_position[0] - 1][head_position[1]] == GameObject.EMPTY:
                    if abs(head_position[0] - 1 - self.food_target[0]) + abs(
                            head_position[1] - self.food_target[1]) < distance:
                        move = Move.LEFT

        if direction == Direction.EAST:
            distance = math.inf
            if head_position[1] != self.board_height - 1:
                if board[head_position[0]][head_position[1] + 1] == GameObject.EMPTY:
                    move = Move.RIGHT
                    distance = abs(head_position[0] - self.food_target[0]) + abs(
                        head_position[1] + 1 - self.food_target[1])
            if head_position[1] != 0:
                if board[head_position[0]][head_position[1] - 1] == GameObject.EMPTY:
                    if abs(head_position[0] - self.food_target[0]) + abs(head_position[1]-1 - self.food_target[1]) < distance:
                        move = Move.LEFT

        if direction == Direction.SOUTH:
            distance = math.inf
            if head_position[0] != self.board_width - 1:
                if board[head_position[0] + 1][head_position[1]] == GameObject.EMPTY:
                    move = Move.LEFT
                    distance = abs(head_position[0] + 1 - self.food_target[0]) + abs(
                        head_position[1] - self.food_target[1])
            if head_position[0] != 0:
                if board[head_position[0] - 1][head_position[1]] == GameObject.EMPTY:
                    if abs(head_position[0]-1 - self.food_target[0]) + abs(head_position[1] - self.food_target[1]) < distance:
                        move = Move.RIGHT

        if direction == Direction.WEST:
            distance = math.inf
            if head_position[1] != self.board_height - 1:
                if board[head_position[0]][head_position[1] + 1] == GameObject.EMPTY:
                    move = Move.LEFT
                    distance = abs(head_position[0] - self.food_target[0]) + abs(
                        head_position[1] + 1 - self.food_target[1])
            if head_position[1] != 0:
                if board[head_position[0]][head_position[1] - 1] == GameObject.EMPTY:
                    if abs(head_position[0] - self.food_target[0]) + abs(head_position[1]-1 - self.food_target[1]) < distance:
                        move = Move.RIGHT

        return move

    def get_undecided_safe_move(self, board, direction, head_position):
        #x = input()
        move = None

        if direction == Direction.NORTH:
            best_score = math.inf
            if head_position[1] != 0:
                if board[head_position[0]][head_position[1] - 1] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0], head_position[1] - 1)))
                    if score != 0:
                        best_score = score
                        move = Move.STRAIGHT
            if head_position[0] != self.board_width - 1:
                if board[head_position[0] + 1][head_position[1]] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0]+1, head_position[1])))
                    if score != 0:
                        if score <= best_score:
                            best_score = score
                            move = Move.RIGHT
            if head_position[0] != 0:
                if board[head_position[0] - 1][head_position[1]] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0]-1, head_position[1])))
                    if score != 0:
                        if score <= best_score:
                            move = Move.LEFT

            return move

        if direction == Direction.EAST:
            best_score = math.inf
            if head_position[0] != self.board_width - 1:
                if board[head_position[0] + 1][head_position[1]] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0] + 1, head_position[1])))
                    if score != 0:
                        best_score = score
                        move = Move.STRAIGHT
            if head_position[1] != self.board_height - 1:
                if board[head_position[0]][head_position[1] + 1] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0], head_position[1] + 1)))
                    if score != 0:
                        if score <= best_score:
                            best_score = score
                            move = Move.RIGHT
            if head_position[1] != 0:
                if board[head_position[0]][head_position[1] - 1] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0], head_position[1] - 1)))
                    if score != 0:
                        if score <= best_score:
                            move = Move.LEFT
            return move

        if direction == Direction.SOUTH:
            best_score = math.inf
            if head_position[1] != self.board_height - 1:
                if board[head_position[0]][head_position[1] + 1] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0], head_position[1] + 1)))
                    if score != 0:
                        best_score = score
                        move = Move.STRAIGHT
            if head_position[0] != 0:
                if board[head_position[0] - 1][head_position[1]] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0] - 1, head_position[1])))
                    if score != 0:
                        if score <= best_score:
                            best_score = score
                            move = Move.RIGHT
            if head_position[0] != self.board_width - 1:
                if board[head_position[0] + 1][head_position[1]] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0] + 1, head_position[1])))
                    if score != 0:
                        if score <= best_score:
                            move = Move.LEFT

            return move

        if direction == Direction.WEST:
            best_score = math.inf
            if head_position[0] != 0:
                if board[head_position[0] - 1][head_position[1]] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0] - 1, head_position[1])))
                    if score != 0:
                        best_score = score
                        move = Move.STRAIGHT
            if head_position[1] != 0:
                if board[head_position[0]][head_position[1] - 1] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0], head_position[1] - 1)))
                    if score != 0:
                        if score <= best_score:
                            best_score = score
                            move = Move.RIGHT
            if head_position[1] != self.board_height - 1:
                if board[head_position[0]][head_position[1] + 1] == GameObject.EMPTY:
                    score = len(self.get_neighbors(board, (head_position[0], head_position[1]+1)))
                    if score != 0:
                        if score <= best_score:
                            move = Move.LEFT

            return move

    def should_redraw_board(self):
        """
        This function indicates whether the board should be redrawn. Not drawing to the board increases the number of
        games that can be played in a given time. This is especially useful if you want to train you agent. The
        function is called before the get_move function.

        :return: True if the board should be redrawn, False if the board should not be redrawn.
        """
        return self.redraw

    def should_grow_on_food_collision(self):
        """
        This function indicates whether the snake should grow when colliding with a food object. This function is
        called whenever the snake collides with a food block.

        :return: True if the snake should grow, False if the snake should not grow
        """

        self.path = None
        self.food_target = None
        #self.done = True

        return self.grow

    def on_die(self, head_position, board, score, body_parts):
        """This function will be called whenever the snake dies. After its dead the snake will be reincarnated into a
        new snake and its life will start over. This means that the next time the get_move function is called,
        it will be called for a fresh snake. Use this function to clean up variables specific to the life of a single
        snake or to host a funeral.

        :param head_position: (x, y) position of the head at the moment of dying.

        :param board: two dimensional array representing the board of the game at the moment of dying. The board
        given does not include information about the snake, only the food position(s) and wall(s) are listed.

        :param score: score at the moment of dying.

        :param body_parts: the array of the locations of the body parts of the snake. The last element of this array
        represents the tail and the first element represents the body part directly following the head of the snake.
        When the snake runs in its own body the following holds: head_position in body_parts.
        """

        self.done = False
        self.explore = 0.8
        if self.counter >= 1000:
            self.counter -= 100

        # f = open("lr.txt", "w")
        #
        # f.write(str(self.counter))

        self.path = None
