#THE NEW GAME CLASSS
class game:

    def move(board, move):
        board_new = game.boardstart()
        if move == 'w':
            merge =[]
            for j in range(0,4):
                merge.append([])
                for i in range(0,4):
                    if board[i][j] != 0:
                        if len(merge[j]) == 0:
                            merge[j].append([board[i][j]])
                        elif merge[j][len(merge[j])-1]==[board[i][j]]:
                            merge[j].pop()
                            merge[j].append([board[i][j], board[i][j]])
                        else:
                            merge[j].append([board[i][j]])
            
            for j in range(0,4):
                for i in range(0, len(merge[j])):
                    board_new[i][j] = game.sum_list(merge[j][i])
        
        elif move == 's':
            merge =[]
            for j in range(0,4):
                merge.append([])
                for i in [3,2,1,0]:
                    if board[i][j] != 0:
                        if len(merge[j]) == 0:
                            merge[j].append([board[i][j]])
                        elif merge[j][len(merge[j])-1]==[board[i][j]]:
                            merge[j].pop()
                            merge[j].append([board[i][j], board[i][j]])
                        else:
                            merge[j].append([board[i][j]])
                        
            for j in range(0,4):
                for i in range(3, 3-len(merge[j]), -1):
                    board_new[i][j] = game.sum_list(merge[j][3-i])
        
        elif move == 'a':
            merge =[]
            for i in range(0,4):
                merge.append([])
                for j in range(0,4):
                    if board[i][j] != 0:
                        if len(merge[i]) == 0:
                            merge[i].append([board[i][j]])
                        elif merge[i][len(merge[i])-1]==[board[i][j]]:
                            merge[i].pop()
                            merge[i].append([board[i][j], board[i][j]])
                        else:
                            merge[i].append([board[i][j]])
            for i in range(0,4):
                for j in range(0, len(merge[i])):
                    board_new[i][j] = game.sum_list(merge[i][j])
        
        elif move == 'd':
            merge =[]
            for i in range(0,4):
                merge.append([])
                for j in [3,2,1,0]:
                    if board[i][j] != 0:
                        if len(merge[i]) == 0:
                            merge[i].append([board[i][j]])
                        elif merge[i][len(merge[i])-1]==[board[i][j]]:
                            merge[i].pop()
                            merge[i].append([board[i][j], board[i][j]])
                        else:
                            merge[i].append([board[i][j]])
            for i in range(0,4):
                for j in range(3, 3-len(merge[i]), -1):
                    board_new[i][j] = game.sum_list(merge[i][3-j])

        else:
            return board
        
        return board_new            


                    
    def pos_zeroes(board):
        pos = []
        for i in range(0,4):
            for j in range(0,4):
                if board[i][j] == 0:
                    pos.append([i,j])
        return pos

    def add_piece(board):
        pos = game.pos_zeroes(board)
        if len(pos) == 0:
            print("NO MORE MOVES")
            return False
        else:
            from random import choice
            pos = choice(pos)
            board[pos[0]][pos[1]] = choice([2,2,2,2,2,2,2,2,2,4])
            return board
        
    def sum_list(list):
        sum = 0
        for i in list:
            sum += i
        return sum

    def boardstart():
        
        board = [[0 for i in range(4)] for j in range(4)]
        return board

    def display(board):
        for j in range(4):
            print(board[j])


    #The new additions

    def checkforvalidmoves(board):#Checks for valid moves.
        for i in ['w','a','s','d']:
            if game.move(board,i) != board:
                return True
        return False

    def movenotvalid(board,action):
        if game.move(board,action) != board:
            return False
        else:
            return True