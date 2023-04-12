# Deep Q Learning TicTacToe Game

import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Initialize all player characters, training metrics, and reward metrics
BLANK = ' '
AI_PLAYER = 'O'
HUMAN_PLAYER = 'X'
# Number of games AI vs AI plays for training
TRAINING_EPOCHS = 10000
# training epsilon for AI vs AI matches during training
TRAINING_EPSILON = 0.4
# If the AI wins while training
REWARD_WIN = 10
# If the AI loses while training
REWARD_LOSE = -10
# If the AI ties while training
REWARD_TIE = 0


class Player:

    # Create the 3x3 board
    @staticmethod
    def show_board(board):
        print('|'.join(board[0:3]))
        print('|'.join(board[3:6]))
        print('|'.join(board[6:9]))


'''
The Human Player class contains all the functions required for the human player
Contains one functions: make_move(), reward()
'''
class HumanPlayer(Player):

    # pass reward functions since human players doesn't actually need reward
    def reward(self, value, board):
        pass

    # The make move function handles the logic when the human player
    # has to make a move on the board
    def make_move(self, board, moves_made):

        while True:
            try:
                # prints out the board
                self.show_board(board)
                # input for move based on cell index
                move = input('Your next move (cell index 1-9): ')
                move = int(move)
                # check if move is not within board range or
                # if a piece is already placed in that cell
                if not (move - 1 in range(9)) or move in moves_made:
                    raise ValueError
            # Error catch for invalid move
            except ValueError:
                print('Invalid move. Try again:\n')
            else:
                # Keeps track of moves already made
                moves_made.append(move)
                # returns the valid move
                return move - 1


'''
The AI Player class contains all the functions required for the AI player
Contains five functions: __init__(), available_moves(), get_q(), make_move(), reward()
'''
class AIPlayer(Player):

    # Initialize all parameters for deep reinforcement learning algorithm
    def __init__(self, epsilon=0.4, alpha=0.3, gamma=0.9):
        # this is the epsilon parameter of the model: the probability of exploration
        self.EPSILON = epsilon
        # learning rate
        self.ALPHA = alpha
        # discount parameter for the future reward (rewards now are better than rewards in the future)
        self.GAMMA = gamma

        # the Q(s,a) function is a neural network (this is how we eliminate the huge data structure)
        # single hidden layer with 32 neurons - output is 1 which is the optimal move (so the index)
        # input neurons: 36
        self.q = Sequential()
        self.q.add(Dense(32, input_dim=36, activation='relu'))
        self.q.add(Dense(1, activation='relu'))
        self.q.compile(optimizer='adam', loss='mean_squared_error')

        # previous move during the game
        self.move = None
        # board in the previous iteration
        self.board = (' ',) * 9

    # find all moves that are the available or empty cells on the grid (board)
    def available_moves(self, board):
        return [i for i in range(9) if board[i] == ' ']

    # one-hot encode the board for the neural network to train on effectievly
    def encode_input(self, board, action):
        # we represent the (s,a) pair with a one-dimensional array (one-hot representation)
        vector_representation = []

        # one-hot encoding for the 3 states
        # [1,0,0] - it means the given cell has X ticker
        # [0,1,0] - it means the given cell has O ticker
        # [0,0,1] - it means the given cell has ' ' ticker so empty
        # every single cell on the board (9 cells) has 3 values because of this representation
        # so there are 9x3=27 values
        for cell in board:
            for ticker in ['X', 'O', ' ']:
                if cell == ticker:
                    vector_representation.append(1)
                else:
                    vector_representation.append(0)

        # one-hot encoding of the action - array with size 9
        # [1,0,0,0,0,0,0,0,0] - it means putting X to the first cell
        # [0,1,0,0,0,0,0,0,0] - it means putting X to the second cell
        for move in range(9):
            if action == move:
                vector_representation.append(1)
            else:
                vector_representation.append(0)

        # all together there are 27+9=36 values (the input layer has 36 neurons)
        return np.array([vector_representation])

    # The make move function handles the logic when the AI player has to make a move on the board
    # make a random move with epsilon probability (exploration) or pick the action with
    # highest Q value (exploitation)
    def make_move(self, board, moves_made):
        self.board = tuple(board)
        actions = self.available_moves(board)

        # action with epsilon probability
        if random.random() < self.EPSILON:
            # this is in index (0-8 board cell related to index)
            self.move = random.choice(actions)
            return self.move

        # take the action with the highest Q value
        q_values = [self.get_q(self.board, a) for a in actions]
        max_q_value = max(q_values)

        # if multiple best actions, choose one at random
        if q_values.count(max_q_value) > 1:
            best_actions = [i for i in range(len(actions)) if q_values[i] == max_q_value]
            best_move = actions[random.choice(best_actions)]
        # there is just a single best move (best action)
        else:
            best_move = actions[q_values.index(max_q_value)]

        # keep track of moves made within moves_made list
        moves_made.append(best_move+1)
        self.move = best_move
        # return AI valid move
        return self.move

    # Q(s,a) - we have the input (s,a) representation and the neural network will make
    # a prediction (returns the Q value - this value will be learned through training)
    def get_q(self, state, action):
        return self.q.predict([self.encode_input(state, action)], batch_size=1)

    # let's evaluate a given state: so update the Q(s,a) table regarding s state and a action
    def reward(self, reward, board):
        if self.move:
            prev_q = self.get_q(self.board, self.move)
            max_q_new = max([self.get_q(tuple(board), a) for a in self.available_moves(self.board)])
            # train the neural network with the new (s,a) and Q value
            self.q.fit(self.encode_input(self.board, self.move),
                       prev_q + self.ALPHA * ((reward + self.GAMMA * max_q_new) - prev_q),
                       epochs=3, verbose=0)

        self.move = None
        self.board = None


'''
The TicTacToe class handles the actual game logic for TicTacToe
Contains three functions: __init__(), play(), is_game_over()
'''
class TicTacToe:

    # Initialize all game parameters like players, trun, and the board
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.first_player_turn = random.choice([True, False])
        self.board = [' '] * 9

    # Game loop for players alternating turns and return with winner or tie game state
    # also assigns rewards to the players
    def play(self):

        moves_made = []

        # this is the game loop
        while True:
            if self.first_player_turn:
                player = self.player1
                other_player = self.player2
                player_tickers = (AI_PLAYER, HUMAN_PLAYER)
            else:
                player = self.player2
                other_player = self.player1
                player_tickers = (HUMAN_PLAYER, AI_PLAYER)

            # check the state of the game (win, lose, or draw)
            game_over, winner = self.is_game_over(player_tickers)

            # game is over: handle rewards
            # other_player doesn't require a reward since it's the human player
            if game_over:
                if winner == player_tickers[0]:
                    player.show_board(self.board[:])
                    print('%s won!' % player.__class__.__name__)
                    player.reward(REWARD_WIN, self.board[:])
                    other_player.reward(REWARD_LOSE, self.board[:])
                if winner == player_tickers[1]:
                    player.show_board(self.board[:])
                    print('%s won!' % other_player.__class__.__name__)
                    other_player.reward(REWARD_WIN, self.board[:])
                    player.reward(REWARD_LOSE, self.board[:])
                else:
                    player.show_board(self.board[:])
                    print('Tie!')
                    player.reward(REWARD_TIE, self.board[:])
                    other_player.reward(REWARD_TIE, self.board[:])
                break

            # next player's turn in the next iteration
            self.first_player_turn = not self.first_player_turn

            # actual player's move best (based on Q(s,a) table for AI player)
            move = player.make_move(self.board, moves_made)
            self.board[move] = player_tickers[0]

    # Check game over conditions
    # If 3 in a row in a row or column or diagonal or all cells filled for tie
    def is_game_over(self, player_tickers):
        # consider both players (X and O players - these are the tickers)
        for player_ticker in player_tickers:

            # check horizontal dimensions (so the rows)
            for i in range(3):
                if self.board[3 * i + 0] == player_ticker and \
                        self.board[3 * i + 1] == player_ticker and \
                        self.board[3 * i + 2] == player_ticker:
                    return True, player_ticker

            # check vertical dimensions (so the columns)
            for j in range(3):
                if self.board[j + 0] == player_ticker and \
                        self.board[j + 3] == player_ticker and \
                        self.board[j + 6] == player_ticker:
                    return True, player_ticker

            # check the diagonal dimensions (top left to bottom right + top right to bottom left)
            if self.board[0] == player_ticker and self.board[4] == player_ticker and\
                    self.board[8] == player_ticker:
                return True, player_ticker

            if self.board[2] == player_ticker and self.board[4] == player_ticker and\
                    self.board[6] == player_ticker:
                return True, player_ticker

        # finally we can deal with the 'draw' cases
        if self.board.count(' ') == 0:
            return True, None
        # game is not over
        else:
            return False, None

'''
Initial entry point for the file
'''
if __name__ == '__main__':
    # Set play game state
    play_game = True

    # Set two AI players for training
    ai_player_1 = AIPlayer()
    ai_player_2 = AIPlayer()

    print('Training the AI player(s)...')

    # Intialize training epsilon
    ai_player_1.EPSILON = TRAINING_EPSILON
    ai_player_1.EPSILON = TRAINING_EPSILON

    # train for x = epochs = games
    for i in range(TRAINING_EPOCHS):
        print("Training iteration %s" % i)
        game = TicTacToe(ai_player_1, ai_player_2)
        game.play()

    print('\nTraining is Done...\n')

    # epsilon=0 means no exploration - it will use the Q(s,a) function to make the moves
    # this wil be used for AI vs Human games
    ai_player_1.EPSILON = 0
    human_player = HumanPlayer()

    while play_game:
        # Initialize and play game
        game = TicTacToe(ai_player_1, human_player)
        game.play()
        # Ask user for rematch
        answer = input('Would you like to play again... Y or N? ')
        if answer == 'Y':
            play_game = True
        else:
            play_game = False
