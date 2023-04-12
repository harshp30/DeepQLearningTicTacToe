# Q-Learning and Deep Q-Learning TicTacToe Game Implementation

**An implementation of a TicTacToe game using Q Reinforcement Learning and Deep Q Reinforcement Learning**

---

**Theoretical Topics:** *Reinforcement Learning, Q-Learning, Deep Q-Learning, Game Theory*

**Tools / Technologies:** *Python, Tensorflow, Keras, NumPy, PyCharm*

---

### Technical Explanation:

1. Q-Learning TicTacToe Implementation  -- [QLearningTicTacToe.py](https://github.com/harshp30/DeepQLearningTicTacToe/blob/main/src/QLearningTicTacToe.py)

    *What is Reinforcement Learning?*

    > Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.

    Reinforcement learning can applied to games such as TicTacToe and chess. It is one of the core machine learning approaches alongside supervised and unsupervised learning

    In this example we are reinforcement learning through a TicTacToe game implementation

    *What is Q-Learning?*

    > Q-learning is a model-free reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment, and it can handle problems with stochastic transitions and rewards without requiring adaptations.

    The agent in Q-Learning knows only about the current game state and possible moves. Using just those two pieces of information it runs many trials (games) and slowly learns optimal moves for the game. At the end of each game the agent is given a positive or negtaive reward based on outcome.

    *Explanation*

    The Q-Learning Reinforcement approach is related to online learning which is when the agent imporves it's behaviour through learning from the histroy of interactions with the environement.

    In order to understand reinforcement learning some definitions need to be clarified...

    -  `Enviornment: This can be thought of the game board within which the game is being played (Ex. TicTacToe board).`
    -  `Agent: This can be thought of as the player making moves in a game.`
    - `Reward: This is a integer value given to the agent at the end of each game. A positive reward is typically given for a win, negtaive reward for a loss, and no reward (0) for a tie.`
    - `State: This is the game state so the current state of the environemnt (Ex. a TicTacToe board with a X down in the middle).`
    - `Action: This is action a performed by the agent, basically the move played.`

    ![Reinforcement Learning](images/reinforcement.png)

    We also have a `Q*(s,a)` fucntion which represents the discounted future reward when we perform an action `a` in state `s` and optimize from tha starting of the game to the end.

    - The function is sorted by the best possible scores at the end of the game after performing action `a`
    - Finding the optimal policy is just a matter of finding the `argmax(Q*(s,a))`

    `Q(s,a) = Q(s,a) [previous] + α(R(s,a) + γ[max, a']Q(s',a') - Q(s,a))`

    - If α=1 we have the Bellman-Equation
    - The learning rate controls how much the difference between Q(s,a) and newly proposed Q(s',a') value the state takes into account
    - Gamma γ is the discount factor within range [0,1]
        - Usually a rule of thumb that rewards in the present are preferred vs the future possible rewards
        - γ~0 -> Agent tends to consider only immediate rewards
        - γ~1 -> Agent tends to consider future rewards more
    - Alpha α i sthe learning rate within the [0,1]
        - Controls how much of the difference between previous Q-values and newly proposed Q-valu is taken into account


    We end up with the following training loop:

    - Initalize gamma `γ`, alpha `α`, and reward `R + Q(s,a)` with zeros
    - For each training session
        - select random initial state
        - while the terminal state has not ended
            - select one among all possible `a` action for current state `s`
            - use the `a` action to go to the nezt `s'` state
            - calculate the max `Q(s',a')` value for the next `s'` state based on all possible actions
            - set the next state as the current one so `s=s'`
        - end loop
    - end current training session

    *Demo*

    Below is a video demonstration of the Q-Learning TicTacToe game:

    [![Q-Learning TicTacToe Video Demo](https://img.youtube.com/vi/501jwEofMY4/maxresdefault.jpg)](https://www.youtube.com/watch?v=501jwEofMY4&ab)
    

2. Deep Q-Learning TicTacToe Implementation  -- [DeepQLearningTicTacToe.py](https://github.com/harshp30/DeepQLearningTicTacToe/blob/main/src/DeepQLearningTicTacToe.py)

     *What is Deep Q-Learning?*

     > Deep Q-Learning uses Experience Replay to learn in small batches in order to avoid skewing the dataset distribution of different states, actions, rewards, and next_states that the neural network will see

     The primary difference is that we use neural networks instead of Q-tables
     - So we use a neural network as the `Q(s,a)` function, the input is just `a` state and the output is the Q-values for each possible state
     - The network ranks the possible actions to perform for a given state

     We can use the mean-squared loss function for optimization purposes

     `L = 1/2[r + γ(max(Q(a',a')) - Q(s,a))]^2`

     *Explanation*

    The implementation is very similar to the Q-Learning implementation, as described above the primary difference is the use a neural network for the Q values

    We have to take into account the opsilon-greedy algroithm

    In order to use the epsilon-greedy algorithm in a useful way we should do the following
    - `ε = 1` -> explore other actions (even if they seem worse)
    - `ε = 0` -> choose the est possible known action only

    This is related to the epxloration-exploitation problem where we initlaize `ε = 1` at the beginning and on every iteration decrease the `ε` value. During actual gameplay we set `ε = 0`

    Another topic to mention is remember + replay

    The algorithm has a tendnecy to forget past experience because it overwrites it with new experiences
    - We need a list of previous experiences and observations to retrain the model with the previous experiences
    - Replay means training the neural network with experiences in the memory and usually we sample reanodm elements from the previous memeory (batch size)

     *Demo*

    Below is a video demonstration of the Deep Q-Learning TicTacToe game:

    [![Q-Learning TicTacToe Video Demo](https://img.youtube.com/vi/sFYxJI9XsGg/maxresdefault.jpg)](https://www.youtube.com/watch?v=sFYxJI9XsGg&ab)


---

### Next Steps:

- Increase epoch value to improve model
- Integrate CUDA to train on GPU as opposed to CPU
- Create GUI for gameplay interface

---

### Additional Notes:
- The gameplay dmeos do not show all possible outocmes of games, in order for me to consistently win I would train the model on lower epochs so it's bad at the game (as seen in the Deep Q-Learning Implementation Demo Video)