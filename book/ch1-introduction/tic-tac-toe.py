#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS    #棋盘


class State:
    def __init__(self):
        # the board is represented by an n * n array,
        # 1 represents a chessman of the player who moves first,
        # -1 represents a chessman of another player
        # 0 represents an empty position
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.hash_val = None
        self.end = None

    # compute the hash value for one state, it's unique
    #用哈希值作为棋盘状态的索引
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):  #按从左到右从上到下遍历棋盘数组，一个9位数，按3进制转化成十进制，作为哈希值
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    # check whether a player has won the game, or it's a tie
    def is_end(self):
        if self.end is not None:
            return self.end
        results = []
        # check row     若某行/某列/某斜线三数之和为3，则说明3个都是1，first赢。同理-3
        for i in range(BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # check columns
        for i in range(BOARD_COLS):
            results.append(np.sum(self.data[:, i]))

        # check diagonals
        trace = 0   #对角
        reverse_trace = 0   #反对角
        for i in range(BOARD_ROWS):
            trace += self.data[i, i]
            reverse_trace += self.data[i, BOARD_ROWS - 1 - i]
        results.append(trace)
        results.append(reverse_trace)

        for result in results:
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end

        # whether it's a tie
        sum_values = np.sum(np.abs(self.data))
        if sum_values == BOARD_SIZE:    #下满了，且不是上面两种情况，就是和局
            self.winner = 0
            self.end = True
            return self.end

        # game is still going on
        self.end = False    #否则继续
        return self.end

    # @symbol: 1 or -1
    # put chessman symbol in position (i, j)
    def next_state(self, i, j, symbol): #根据当前状态，下棋位置和下棋者，获得下一状态
        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    # print the board
    def print_state(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

def get_all_states_impl(current_state, current_symbol, all_states):#这函数递归实在是太难理解了
    for i in range(BOARD_ROWS):     #每个位置分别有0 1 -1三种可能
        for j in range(BOARD_COLS):
            if current_state.data[i][j] == 0:
                new_state = current_state.next_state(i, j, current_symbol)#获得新状态
                new_hash = new_state.hash() #获得新状态的key
                if new_hash not in all_states:  #检查是否重复
                    is_end = new_state.is_end()
                    all_states[new_hash] = (new_state, is_end)  #存了两次is_end，state里面也有is_end
                    if not is_end:  #如果结束了，就不再递归，也就是说后面的空位不需要填满，筛掉了后面可能的状态，减少了总的状态数
                        get_all_states_impl(new_state, -current_symbol, all_states)#递归调用

'''上函数：模拟对弈，你一步我一步，所以1和-1交替赋值给棋盘。回合制，两人轮流选择下棋的位置。这是符合实际问题的。当还没结束，就让另一个继续下，且
     是按从左到右从上到下的顺序找到空位来下。若该状态已经结束，就会让当前symbol下在其他位置，即遍历所有位置。
     这样是考虑了状态的先后顺序的，比如某状态是结束，然后symbol不下在该位置，下在别的位置，但可能最后又回到了
     同一个结束状态，只不过两个人在其他位置多下了几次。这对于一开始的agent来说，更晚结束的状态的确是存在且有意
     意义的，但到后面agent更聪明了，肯定是会倾向于到达更早结束的状态,更晚的状态其实就是输了，因为对手不会同样失误的
下函数：按从左到右从上到下，每个位置分别赋值1 -1 ，实现对棋盘状态的穷举。此状态集是上述状态集的父集，包含更多的
     状态。但是包含许多无意义的状态，看似是交替赋值1 -1，但实际上会出现一个人连下两次的情况。比如1 -1 1 -1 1 -1 0 0 0状态，
     在(3,1)位置下1，则结束，那就转到在（3,1）下-1，但前一个下的也是-1，这是不合理的。所以先选位置，对同一位置先后赋予1 和-1
     就会出现这种情况。按上函数，先选1或-1，再去选位置，就是让当前symbol一定要选个位置下，不能因为下了结束，就不下这
     一步让另一个人下，这样才能保证一定是交替进行的。'''

def get_allstates_my(current_state, all_states, a,symbol):
    i = a // BOARD_COLS
    j = a % BOARD_COLS
    new_state = current_state.next_state(i, j, symbol)#获得新状态
    new_hash = new_state.hash() #获得新状态的key
    if new_hash not in all_states:  #检查是否重复
        is_end = new_state.is_end()
        all_states[new_hash] = (new_state, is_end)  #存了两次is_end，state里面也有is_end
        if not is_end:  #如果结束了，就不再递归，也就是说后面的空位不需要填满，筛掉了后面可能的状态，减少了总的状态数
            a=1+a
            get_allstates_my(new_state, all_states,a,-symbol)

    new_state = current_state.next_state(i, j, -symbol)
    new_hash = new_state.hash() 
    if new_hash not in all_states:  
        is_end = new_state.is_end()
        all_states[new_hash] = (new_state, is_end)  
        if not is_end: 
            a=1+a
            get_allstates_my(new_state, all_states,a,symbol)

def get_all_states():
    current_symbol = 1
    current_state = State() #第一个状态，所有位置都是0，尤其是第一个位置是0.其他位置是0的都会被动的被考虑到
    all_states = dict() #字典，只需要用新的键作索引并赋值，就是添加新的键值对
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    #get_allstates_my(current_state, all_states,0,1)
    return all_states


# all possible board configurations
all_states = get_all_states()


class Judger:
    # @player1: the player who will move first, its chessman will be 1
    # @player2: another player with a chessman -1
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)
        self.current_state = State()

    def reset(self):
        self.p1.reset()
        self.p2.reset()

    def alternate(self):#交替玩家函数
        while True:
            yield self.p1   #返回p1对象
            yield self.p2

    # @print_state: if True, print each board during the game
    def play(self, print_state=False):
        alternator = self.alternate()
        self.reset()    #重置状态
        current_state = State() #初始化状态
        self.p1.set_state(current_state)    #因为两个玩家分别建立对状态的estimation，所以每个玩家都要有一份状态
        self.p2.set_state(current_state)
        if print_state:
            current_state.print_state()
        while True: #开始下棋
            player = next(alternator)   #next真正调用yield函数
            i, j, symbol = player.act()
            next_state_hash = current_state.next_state(i, j, symbol).hash()#next_state只有棋盘状态，没有is_end
            current_state, is_end = all_states[next_state_hash]
            self.p1.set_state(current_state)#更新状态
            self.p2.set_state(current_state)
            if print_state:
                current_state.print_state()
            if is_end:
                return current_state.winner


# AI player
class Player:
    # @step_size: the step size to update estimations
    # @epsilon: the probability to explore
    def __init__(self, step_size=0.1, epsilon=0.1):
        self.estimations = dict()   #字典
        self.step_size = step_size
        self.epsilon = epsilon
        self.states = []    #states是列表，每一盘游戏存储从初始到结束的每一步的状态
        self.greedy = []
        self.symbol = 0

    def reset(self):
        self.states = []
        self.greedy = []

    def set_state(self, state):
        self.states.append(state)
        self.greedy.append(True)

    def set_symbol(self, symbol):
        self.symbol = symbol
        for hash_val in all_states:
            state, is_end = all_states[hash_val]
            if is_end:
                if state.winner == self.symbol:
                    self.estimations[hash_val] = 1.0
                elif state.winner == 0:
                    # we need to distinguish between a tie and a lose
                    self.estimations[hash_val] = 0.5
                else:
                    self.estimations[hash_val] = 0
            else:
                self.estimations[hash_val] = 0.5

    # update value estimation
    def backup(self):
        states = [state.hash() for state in self.states]

        for i in reversed(range(len(states) - 1)):#要玩完一盘才会更新一次状态值
            state = states[i]
            td_error = self.greedy[i] * (#从后往前，用后一个状态去更新前一个状态
                self.estimations[states[i + 1]] - self.estimations[state]
            )
            self.estimations[state] += self.step_size * td_error

    # choose an action based on the state
    def act(self):
        state = self.states[-1] #返回最后一个元素，即最新的状态
        next_states = []    #只是储存状态对应的关键字
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    next_positions.append([i, j])   #当前状态可以下的位置
                    next_states.append(state.next_state(i, j, self.symbol).hash())  #相应的在该位置下棋后的状态的关键字

        if np.random.rand() < self.epsilon: #在可以下的位置随机选择
            action = next_positions[np.random.randint(len(next_positions))]#在长度范围内随机选索引，再根据索引获得对应位置
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        for hash_val, pos in zip(next_states, next_positions):
            values.append((self.estimations[hash_val], pos))#根据关键字找到对应状态的estimation
        # to select one of the actions of equal value at random due to Python's sort is stable
        np.random.shuffle(values)#先打乱顺序，因为sort排序是稳定的，即值相同的两个元素在排序前后顺序是不变的，如果不打乱就不会选到后一个元素
        values.sort(key=lambda x: x[0], reverse=True)#用values的第1列(estimation)作为排序依据
        action = values[0][1]#选择第一行（estimation最大）的第2列元素（位置）赋值给action
        action.append(self.symbol)#加上是谁下的
        return action

    def save_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.estimations, f)

    def load_policy(self):
        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.estimations = pickle.load(f)


# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']#字母位置与棋盘位置对应
        self.state = None

    def reset(self):
        pass

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol

    def act(self):
        self.state.print_state()#每次都打印棋盘
        key = input("Input your position:") #获取键盘输入
        data = self.keys.index(key) #在keys里查找对应字母的索引
        i = data // BOARD_COLS
        j = data % BOARD_COLS   #转换成二维棋盘坐标
        return i, j, self.symbol


def train(epochs, print_every_n=500):
    player1 = Player(epsilon=0.01)
    player2 = Player(epsilon=0.01)  #构造电脑玩家对象，epsilon-greedy 策略
    judger = Judger(player1, player2)   #构造裁判（上帝/环境）对象
    player1_win = 0.0
    player2_win = 0.0
    for i in range(1, epochs + 1):
        winner = judger.play(print_state=False)
        if winner == 1:
            player1_win += 1#统计双方赢的次数
        if winner == -1:
            player2_win += 1
        if i % print_every_n == 0:#每多少回合打印一次双方赢的比例
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        player1.backup()    #玩完一盘才更新一次状态值，且是对本次游戏出现过的状态进行更新
        player2.backup()
        judger.reset()  #重置状态
    #player1.save_policy()
    #player2.save_policy()


def compete(turns):
    player1 = Player(epsilon=0)
    player2 = Player(epsilon=0)
    judger = Judger(player1, player2)
    player1.load_policy()   #载入训练好的状态值表
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(turns):
        winner = judger.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        judger.reset()
    #输出两个电脑的胜率，验证policy的最优性，保证不会让对手赢
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))


# The game is a zero sum game. If both players are playing with an optimal strategy, every game will end in a tie.
# So we test whether the AI can guarantee at least a tie if it goes second.
def play():
    while True:
        player1 = HumanPlayer()
        player2 = Player(epsilon=0)
        judger = Judger(player1, player2)
        player2.load_policy()   #电脑载入policy
        winner = judger.play()
        if winner == player2.symbol:
            print("You lose!")
        elif winner == player1.symbol:
            print("You win!")
        else:
            print("It is a tie!")


if __name__ == '__main__':
    train(int(1e1))     #训练得到状态值函数表并保存
    compete(int(1e1))   #载入表，利用表使两个机器人对战，验证是否能不相上下
    play()      #人机对战
 