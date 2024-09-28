Steps:

1. Build a neural net with 16 input nodes. 16 being the number of values on the 2048 game (4x4 grid).  The number of hidden layers and the number of neurons in them can be chosen later as you only need enough to induce non-linearity in the network. 
2. The output layer should have 4 nodes, each outputting the actions values of all four actions for the input state. Choose the action according to an epsilon greedy policy. And propagate the error back into the network. ERROR = [r + max(Q(s’,a’)) - Q(s,a)]
3. This error will, be our loss L1. And this is propagated back into the network to update the weights. 
4. We will maintain two networks, a sample network which will constantly be updated and a target network which gives us the action values of the future states


The Atari DQN paper was followed for this implementation. 
MILESTONE: A high score of 512
