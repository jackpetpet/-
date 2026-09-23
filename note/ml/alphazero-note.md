# 从代码理解 AlphaZero

> AlphaZero 源码学习笔记

---

## 0. 写在前面

* 有一些强化学习的基础所以就想看看alphazero
* 听完课之后觉得对一些细节还不是很理解，就去找了类似的源代码
* 本文参考的是五子棋的AlphaZero
* 本文主要记录自己的理解过程

> 本文主要用于学习和源码分析，代码来自开源项目，版权及许可证遵循原项目要求。

**原项目：**

https://github.com/junxiaosong/AlphaZero_Gomoku#update-2018117-supports-training-with-pytorch

---

# 1. AlphaZero 整体结构

## 1.1 我一开始对 AlphaZero 的理解
大概就是先训练神经网络，再自我对弈再迭代更新参数的过程

---

## 1.2 AlphaZero 整体流程

```text
                 ┌──────────────────────┐
                 │   Policy-Value Net   │
                 │      神经网络         │
                 └──────────┬───────────┘
                            │
                    输出 Policy + Value
                            │
                            ↓
                 ┌──────────────────────┐
                 │         MCTS         │
                 │     蒙特卡洛树搜索    │
                 └──────────┬───────────┘
                            │
                       搜索当前局面
                            │
                            ↓
                 ┌──────────────────────┐
                 │      Self-Play       │
                 │       自我对弈         │
                 └──────────┬───────────┘
                            │
                     产生训练数据
                            │
                            ↓
                  (state, π, z)
                            │
                            ↓
                 ┌──────────────────────┐
                 │       Training       │
                 │       网络训练         │
                 └──────────┬───────────┘
                            │
                     更新网络参数 θ
                            │
                            └──────────────┐
                                           │
                                           ↓
                                  新的 Policy-Value Net
                                           │
                                           └────→ 再次 Self-Play
```

---

# 2. Board：棋盘状态

## 2.1 Board 是干什么的？
这个是定义棋盘一些规则，功能还有如何确定胜利，棋盘为8*8，连成5个就算获胜，分为玩家1,2，states用于存状态，availables里是棋盘中还空的坐标，move_to_location是将标号转化成棋盘的坐标，
location_to_move和前面的相反，current_state函数返回4*8*8，其中的4分别表示当前玩家的状态，对方玩家的状态，最后一步走的坐标还有为下一步要走的玩家，如果是2的整数倍，那下一步就走先出的人，
其他就相反，do_move函数确定现在要走的人（也就是轮到谁出棋了），has_a_winner就是确定输赢函数，输赢有四种状态，横连五子，竖连五子，还有左斜和右斜，game_end函数是确定游戏是否结束，平局返回-1
最后的函数是得到当前玩家。
总结来看就是定义了棋盘的规则还有表达方式还有就是当前玩家和上一个玩家，输入给神经网络的也是4*8*8的，这里4代表的是仅含当前玩家的棋谱，仅含对方玩家的棋谱，上一个走的状态，现在要走的玩家。

---

### 💻 对应代码

```python
class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

```

---

# 3. MCTS：蒙特卡洛树搜索
## 3.1 MCTS 是什么？
MCTS（Monte Carlo Tree Search，蒙特卡洛树搜索）是一种通过不断模拟和搜索来选择动作的方法。
简单来说，MCTS 并不是直接告诉我们“下一步一定应该走哪里”，而是从当前状态出发，反复尝试不同的走法，并根据搜索过程中得到的结果，逐渐判断哪些动作更值得选择。
在 AlphaZero 中，MCTS 不再单纯依靠随机模拟，而是结合神经网络提供的两个信息：
- **Policy（策略）**：告诉 MCTS 哪些动作更值得优先探索；
- **Value（价值）**：评价当前局面的好坏。
MCTS 会不断进行搜索，并在搜索过程中更新每个动作的访问次数和价值估计。搜索完成后，再根据各个动作被访问的次数形成最终的动作概率分布。
因此，可以把 AlphaZero 中的 MCTS 简单理解为：
神经网络负责提供“方向”，MCTS 负责进行“进一步搜索和验证”。
整个过程可以概括为：

当前棋盘状态
↓
神经网络给出 Policy 和 Value
↓
MCTS 根据这些信息选择值得探索的动作
↓
不断进行多次搜索
↓
统计各个动作的访问情况
↓
得到最终的动作概率
↓
选择下一步动作

---

## 3.2 一次 MCTS 搜索做什么？

整体可以理解成：

```text
Selection
    ↓
Expansion
    ↓
Evaluation
    ↓
Backup
```

---

## 3.3 Selection

### 💻 对应代码

```python
    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
```
选择那个被选中次数最多的动作

---

## 3.4 Expansion

### 💻 对应代码

```python
    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
```
得到子节点的得分
---


## 3.5 Evaluation

### 💻 对应代码

```python
    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
```
根据公式计算这个结果，返回的值就是打的分数
---

## 3.6 Backup

### 💻 对应代码

```python
    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
```
更新_Q参数，更新_n_visits被访问次数，下面这个函数是继续往上更新

---

# 4. Q：动作价值

## 4.1 Q 是什么？动作价值是做了这一系列动作得到的奖励

---

### 📐 数学公式

$$
Q(s,a)=\frac{W(s,a)}{N(s,a)}
$$

---

### 💻 对应源码

```python
self._Q += (leaf_value - self._Q) / self._n_visits
```
---
这个式子是化简过的，可以这样理解，平均加上现在这个参数还离平均差多少的平均，也就是在平均的基础上我可以加多少减多少。
---

# 5. U：探索项

## 5.1 为什么有了 Q 还需要 U？

U代表探索，必须要有探索不然和神经网络直接输出没有区别，因为神经网络输出是有局限的

---

### 📐 公式

$$
U(s,a)=c_{\text{puct}}
P(s,a)
\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}
$$
---
### 💻 对应代码

```python
self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
```
被访问的次数越多，被探索的几率就越小

---

# 6. Q + U：MCTS 到底怎么选择动作？
$$
a^*
=
\arg\max_a
\left[
Q(s,a)+U(s,a)
\right]
$$
## 6.1 Q 和 U 分别代表什么？

```text
Q
↓
利用已有搜索结果

U
↓
鼓励探索
```
---

## 6.2 Q + U 是概率吗？

不是概率是分数，可以理解为得的分数
---

# 7. Policy-Value Network

## 7.1 网络输入什么？

经验池中存的（state_batch,mcts_probs,winner_batch）
---

## 7.2 网络输出什么？

```text
State
  ↓
Network
  ↓
┌────────────┐
│            │
Policy      Value
P             V
```

---

### 💻 对应代码

```python
class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.data[0], entropy.data[0]
        #for pytorch version >= 0.5 please use the following line instead.
        #return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)

```

---

### 📝 我的理解
PolicyValueNet分为三个模块，policy_value_fn--用于MCTS，train_step--用于训练网络，save_model--用于保存模型
policy_value_fn得到act_probs,value ，分别是动作概率和价值，在合法的位置得到概率，再将这些信息交给MCTS
train_step在经验回放中取出一条信息，包括state_batch,mcts_probs,winner_batch然后state_batch通过网络得到log_act_probs,value
用于计算value_loss，policy_loss,然后更新网络参数。
---

# 8. Self-Play

## 8.1 什么是 Self-Play？
自己与自己下棋，得到一条路径的最终胜负    
---

$$
(s,\pi,z)
$$

---

### 💻 对应代码

```python
        def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # store the data
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)
```
自己和自己下棋的过程，需要下到分出输赢，赢了的每个步骤都付奖励为1，输了的每个步骤都付奖励为-1，最后又将这些数据存入经验池，用于训练神经网络

---

# 9. Training

## 9.1 训练数据从哪里来？

```text
Self-Play
    ↓
(s, π, z)
    ↓
Replay Buffer
    ↓
Random Sample
    ↓
Training
```
---

## 9.2 winner_batch 是什么？
winner_batch就是代表winner,这个状态下搜索树自对弈后的最终赢家也就是赢的话值为1，输为-1
---

### 💻 对应代码

```python
winner_batch = [data[2] for data in mini_batch]
```

---

## 9.3 Value Loss

### 💻 对应代码

```python
value_loss = F.mse_loss(
    value.view(-1),
    winner_batch
)
```

### 📐 公式

$$
L_{value}=(V-z)^2
$$

## 9.4 Policy Loss

policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))

---

# 10. Data Augmentation

## 10.1 为什么要旋转和翻转？
旋转和翻转可以得到更多的数据也不会改变棋局的胜负，只会改变一些位置的表达，总共要翻转4次



### 💻 对应代码

```python
    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data
```

---


# 11. Training Pipeline

## 11.1 run() 到底做了什么？
run函数的步骤是先进行自对弈的过程，收集数据到经验池，如果经验池的数据收集够了的话就去更新训练网络的参数，更新后的网络再进行下一轮的子对弈
训练若干个batch之后去评估模型，在当前模型中是让MCTS+PolicyValueNet和Pure MCTS对弈并保存模型，当前模型评估超越了历史最佳模型则更新并保存最佳模型
当模型在当前评估难度下达到 100% 胜率时，提高 Pure MCTS 的搜索次数，从而增加后续评估的难度

---

### 💻 对应代码

```python
    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

```
流程如下：
                  ┌─────────────────────┐
                  │  当前 PolicyValueNet │
                  └──────────┬──────────┘
                             ↓
                     MCTS + Self-Play
                             ↓
                    collect_selfplay_data
                             ↓
                       产生训练数据
                             ↓
                       data_buffer
                             ↓
                   数据量是否足够？
                       ↙           ↘
                     否             是
                     ↓               ↓
                  继续收集      policy_update
                                     ↓
                              更新神经网络参数
                                     ↓
                                     └──────────┐
                                                │
                                                ↓
                                       下一轮 Self-Play
                                                │
                                                │
                     ┌──────────────────────────┘
                     ↓
              每隔 check_freq 次
                     ↓
               policy_evaluate
                     ↓
                  win_ratio
                     ↓
              ┌──────┴──────┐
              ↓             ↓
        保存当前模型     超过历史最佳？
                            ↓
                           是
                            ↓
                    保存 best_policy
                            ↓
                    是否达到 100%？
                            ↓
                           是
                            ↓
                  增加 Pure MCTS 搜索次数
---

# 12. 把整个 AlphaZero 串起来

```text
             当前状态 s
                  ↓
        Policy-Value Network
                  ↓
             P + V
                  ↓
                 MCTS
                  ↓
            多次搜索
                  ↓
              π
                  ↓
                落子
                  ↓
              Self-Play
                  ↓
            游戏结束
                  ↓
              得到 z
                  ↓
            (s, π, z)
                  ↓
              Training
                  ↓
             新 Network
                  ↓
                循环
```

---


