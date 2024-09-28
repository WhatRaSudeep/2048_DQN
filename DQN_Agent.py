#This VerSION Focusses on the newlog 17/06/24 of notion

from new_game_CLASS import game
from torch import nn, save, tensor
import torch
import random
import numpy as np
import matplotlib as plt

FILLING_STEPS = 10
PLANNING_STEPS = 10 

def conv_action(action):
    if action == '0':
        return 'w'
    elif action == '1':
        return 'a'
    elif action == '2':
        return 's'
    elif action == '3':
        return 'd'
    else:
        return None

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.replay_memory = []      
    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x))
        x = nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, x):
        return self.forward(x).argmax().item()
    
    def backprop(self, x, y): 
        self.optimizer.zero_grad()
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)  # Change to MSELoss
        loss.backward()
        self.optimizer.step()
        return loss.item()


def planning(replay_memory):
        if len(replay_memory) > FILLING_STEPS:
            for i in range(PLANNING_STEPS):
                transition = random.sample(replay_memory,1)
                (board_tensor_init, reward, board_tensor_final) = transition[0]
                q_value_fin = target_net.forward(board_tensor_final).max().item()


                epsilon = random.random()
                q_value = target_net.forward(board_tensor_init)
                if random.random() < 0: #i am keeping it completely random for now
                    q_value = q_value.max().item()
                else:
                    q_value = q_value[random.randint(0,3)].item()



                q_update_loss = reward + 0.01 * q_value_fin  - q_value
    
                # Prepare target tensor for MSELoss
                target = target_net.forward(board_tensor).detach()
                target[0] = q_update_loss  # Assume 0 is the index for the relevant action
    
                # Perform backpropagation
                target_net.backprop(board_tensor, target)

        else:
            print("filling the replay memory")
   
# def reward_func(prev_state, next_state):
#     prev_sum = np.array(prev_state).sum()
#     next_sum = np.array(next_state).sum()
#     prev_max = np.array(prev_state).max()
#     next_max = np.array(next_state).max()
#     ret = 0
#     if prev_state == next_state:
#         ret-=1
#     if prev_sum < next_sum:
#         ret+=0
#     elif prev_max < next_max:
#         ret+=1

#     return ret


replay_memory =[]

target_net = Net()
training_net = Net()
board = game.boardstart()
board_old = 1
board_old_old = board#any random value that is not equal to board, so that it passes through the first if statement
board_old_old_old = board
i = 0
maxi = 0
gameover = False
failures = 0


while True:
    # print('Iteration:', i)
    reward = -1

#     #the board old thing is fuckall, i will remove it later
#     board_old_old_old = board_old_old
#     board_old_old = board_old
# #i want to print only the final board of each episode
#     board_old = board
#     board = game.add_piece(board)


    if board_old!= board:
        board = game.add_piece(board)#this should only add a piece of the previous move

        board_old_old_old = board_old_old  
        board_old_old = board_old
        board_old = board                #wait i think there might be a bug here, lmao#yea so initially there was board_old=board, so we would lose memory
        #only update previous states if valid move was made.
        
    else:
        failures+=1
        # print("Failures:", failures)#it is failing a lot, but for now i will ignore this.
        # reward-=1 #it looks like -1 is not sufficient as the ai still plays the same move forever. 
        reward-=15

    

    if gameover is True:#technically this should never happen if the ai plays all the valid moves.
        #i adjusted the above variable for a gameover variable
        board = game.boardstart()
        board = game.add_piece(board)

        game.display(board_old_old_old)
        game.display(board_old_old)
        game.display(board_old)
        print("NOOOOOOOOOO")
        print("maximum = ", maxi)
        print("failures= ", failures)

        maxi = 0
        gameover = False
        failures = 0
#And yes my hypothesis was right, this block of code was never executed

    # game.display(board)
    # print("/n")


#add your check for valid moves and make sure the ai only plays valid moves
#this should happen exactly after adding a new piece
    if game.checkforvalidmoves(board) == False:
        print("NoValidMoves")
        reward-=50
        game.display(board)
        gameover = True



    
    # Flatten the board for the neural network
    board_flat = np.array(board).flatten()
    # maxi = 0
    if maxi<np.max(board_flat):
        maxi = np.max(board_flat)

    # Convert to tensor and ensure correct dtype
    board_tensor = tensor(board_flat, dtype=torch.float32)
    
    user = target_net.predict(board_tensor)
    user = conv_action(str(user))

    #check if user is valid, if we allow only valid moves then the  if game.move(board, user) == board: reward += -1 would not be needed.

    # if game.movenotvalid(board, user):#we should add a piece only if the previous move was valid, so i will add a new variable called valid
    #     reward += -1                  #and i will update the validity of the previous move. 
    
                                      #or honestly I might just do a if board_old==board, then dont add piece   
                                      # I will go with this for now, so i commented out the movenotvalid method.    


    # print(user)
    # game.display(board)
    # print("User:",user)

    if failures ==5000:
        game.display(board)
        print(failures)
        break

    # if game.move(board, user) == board:
    #     reward += -10
    #     print("should'nt be happening")#ok, so shouldn't be happening and the first if statement are the same things
    #ok so one good thing is that it tries out different move after an invalid move. and we are not adding any piece after an invalid move.
    #i am adding this reward in the first statement itself.

    init_val = np.array(board).sum()
    max_val_init = np.array(board).max()
    board = game.move(board, user)
    final_val = np.array(board).sum()
    max_val_final = np.array(board).max()

    if max_val_init < max_val_final:
        reward +=1
    

    # reward += reward_func(board, game.move(board, user))




        # Flatten the board for the neural network
    board_flat_final = np.array(board).flatten()
    
    # Convert to tensor and ensure correct dtype
    board_tensor_final = tensor(board_flat_final, dtype=torch.float32)

    q_value_final = target_net.forward(board_tensor_final).max().item()
    
    epsilon = random.random()
    q_value = target_net.forward(board_tensor)
    if random.random() < 0: #i am keeping it completely random for now
        q_value = q_value.max().item()
    else:
        q_value = q_value[random.randint(0,3)].item()

    q_update_loss = reward + 0.01 * q_value_final - q_value
    
    # Prepare target tensor for MSELoss
    target = target_net.forward(board_tensor).detach()
    target[0] = q_update_loss  # Assume 0 is the index for the relevant action
    
    # Perform backpropagation
    target_net.backprop(board_tensor, target)
    
    replay_memory.append((board_tensor, reward, board_tensor_final))
    planning(replay_memory)

    i += 1
    k = 8
    if i % k == 0:
        training_net.load_state_dict(target_net.state_dict())
    # if i % 1000 == 0:
    #     print('Saving model...')
    #     save(target_net.state_dict(), 'model.pth')
    #     print('Model saved')
