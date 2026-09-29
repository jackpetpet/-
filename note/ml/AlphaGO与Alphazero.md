Alphazero究竟是有什么更亮点的地方，当AlphaGO与Alphazero对弈时实现100:0的局面，
在学Alphazero的时候，因为本身本人听课是听的AlphaGO但看的代码是后者，所以一直觉得就应该这样，下面让我们一起来看看

首先就是AlphaGO和Alphazero的框架
AlphaGO
```
                         ┌────────────── AlphaGo ──────────────┐
                         │                                     │
                    输入棋盘状态                                │
                         │                                     │
                         ↓                                     │
              ┌──────────┴──────────┐                          │
              ↓                     ↓                          │
       Policy Network         Value Network                    │
              │                     │                          │
              ↓                     ↓                          │
       预测下一步动作概率       评估当前局面价值                  │
              │                     │                          │
              └──────────┬──────────┘                          │
                         ↓                                     │
                        MCTS                                   │
                         │                                     │
             ┌───────────┼────────────┐                        │
             ↓           ↓            ↓                        │
          Selection   Expansion    Evaluation                  │
             │           │            │                        │
             │           ↓            │                        │
             │       新增节点         │                         │
             │                        │                        │
             │                 ┌──────┴──────┐                 │
             │                 ↓             ↓                 │
             │           Value Network   Rollout               │
             │                 │             │                 │
             │                 └──────┬──────┘                 │
             │                        ↓                        │
             └────────────────── Backup                        │
                                      │                        │
                                      ↓                        │
                              更新搜索树统计                    │
                                      │                        │
                                      ↓                        │
                              选择最终落子                      │
                                      │                        │
                                      ↓                        │
                                  进行对局                      │
                                      │                        │
                                      ↓                        │
                              获得最终胜负结果                   │
                                      │                         │
                    ┌─────────────────┴─────────────────┐       │
                    ↓                                   ↓       │
              人类棋谱监督学习                    自我对弈强化学习 │
                    │                                   │       │
                    ↓                                   ↓       │
             学习人类落子策略                      更新网络参数    │
                    │                                   │       │
                    └─────────────────┬─────────────────┘       │
                                      ↓                         │
                                更强的 Policy                    │
                                更强的 Value                     │
                                      │                         │
                                      └────────→ MCTS ──────────┘
```
Alphazero
```
                      ┌──────────── AlphaGo Zero ────────────┐
                      │                                      │
                 初始棋盘状态                                 │
                      │                                      │
                      ↓                                      │
                一个神经网络                                  │
                      │                                      │
                      ↓                                      │
               共享特征提取                                   │
                      │                                      │
                ┌─────┴─────┐                                │
                ↓           ↓                                │
           Policy Head   Value Head                          │
                │           │                                │
                ↓           ↓                                │
          动作概率 P        局面价值 V                        |
                │           │                                │
                └─────┬─────┘                                │
                      ↓                                      │
                     MCTS                                    │
                      │                                      │
          ┌───────────┼────────────┐                          │
          ↓           ↓            ↓                          │
       Selection   Expansion    Evaluation                    │
          │           │            │                          │
          │           ↓            │                          │
          │        新增节点        │                           │
          │                        │                          │
          │                    神经网络                        │
          │                        │                          │
          │                 ┌──────┴──────┐                   │
          │                 ↓             ↓                   │
          │              Policy         Value                 │
          │                 │             │                   │
          │                 └──────┬──────┘                   │
          │                        ↓                          │
          └──────────────────── Backup                        │
                                   │                          │
                                   ↓                          │
                           更新搜索树统计                      │
                                   │                          │
                                   ↓                          │
                           得到 MCTS 策略 π                    │
                                   │                          │
                                   ↓                          │
                              选择动作                         │
                                   │                          │
                                   ↓                          │
                              执行动作                         │
                                   │                          │
                                   ↓                          │
                              自我对弈                         │
                                   │                          │
                                   ↓                          │
                            获得最终胜负 z                     │
                                   │                          │
                    ┌──────────────┴──────────────┐           │
                    ↓                             ↓           │
                 状态 s                        策略 π          │
                    │                             │           │
                    └──────────────┬──────────────┘           │
                                   ↓                          │
                            训练神经网络                       │
                                   │                          │
                         ┌─────────┴─────────┐                │
                         ↓                   ↓                │
                    Policy目标            Value目标            │
                         │                   │                │
                         ↓                   ↓                │
                       π                  z                   │
                         │                   │                │
                         └─────────┬─────────┘                │
                                   ↓                          │
                             更新网络参数                      │
                                   │                          │
                                   └──────→ MCTS ──────────────┘
```
看完架构图的对比，我们发现有以下词，人类行为模仿和同一神经网络，这分别是对方不同的东西
人类行为模仿也叫行为克隆，也就是让神经网络去学习高手的招式和下棋风格，从而变得强大，局限性在于如果有一个从未见过的局面，那它就不智能了，可以说是克隆
再者我觉得的缺点就是还有，因为棋手对弈难免会有些地方没考虑到，导致给网络的是个不是最优下法的学习，会有一些混淆。
Alphazero去掉这个其实在预期之中，因为我们知道蒙特卡洛树搜索的过程，也就是对于一个状态的几乎要搜索上千次，对于对弈来说，上千次已经挺多的了，几乎模拟完

所有的路径了然后挑选个最好的，如果我们本身不去模仿人的行为，而是发挥自己的算力特长，那效果会不会更好
答案是会的，这就是alphazero所做的一个改进

下来我们通过代码来看这两个的不同地方

首先第一点：
AlphaGO的训练数据是人类棋谱和self-play数据
Alphazero直接是self-play数据
其次：
AlphaGO是两个独立网络，策略网络和价值网络，因为策略网络和价值网络需要的特征很相似，所以如果把它们结合在一起会更好，更不用一个更新完更另一个，只需要更新同一套参数

Alphazero是一个网络两个输出
```python
class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2*board_width*board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2*self.board_width*self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))
        return x_act, x_val


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
```
这个的漂亮之处在于，两种任务可以共享棋盘特征，并减少重复的特征提取

再者：
MCTS的评估方式：
AlphaGO
```
MCTS
 │
 ├── Neural Network
 │      ↓
 │    Value
 │
 └── Rollout
        ↓
     模拟下棋
        ↓
     得到结果
```
是通过神经网络得到价值函数的值，然后再随机模拟下棋，这个自对弈的过程用一个轻量的rollout policy，根据当前局面快速选择动作
```python
def _playout(state):

    # =========================
    # 1. Selection
    # =========================
    node = root

    while not node.is_leaf():

        action = node.select()
        state.do_move(action)

    # =========================
    # 2. Expansion
    # =========================
    action_probs = policy_network(state)

    node.expand(action_probs)

    # =========================
    # 3. Evaluation
    # =========================

    # ① Value Network
    value = value_network(state)

    # ② Rollout
    rollout_state = copy.deepcopy(state)

    while not rollout_state.game_end():

        # 注意：
        # 这里不是再跑 MCTS
        # 也不是再跑 Value Network
        #
        # 而是一个非常快的 rollout policy
        probs = rollout_policy(rollout_state)

        action = sample(probs)

        rollout_state.do_move(action)

    rollout_result = get_winner(rollout_state)

    # =========================
    # 4. 综合
    # =========================
    leaf_value = combine(
        value,
        rollout_result
    )

    # =========================
    # 5. Backup
    # =========================
    node.update_recursive(leaf_value)

```
可以理解为以下过程，这也就是为什么与alphaGO Master与柯洁下完棋后，柯洁说在alphaGO的身上看见了许多先贤和对手，甚至看到了自己的影子。
```
当前棋盘
   ↓
观察候选位置附近的棋形
   ↓
提取一些局部特征
   ↓
计算每个动作的分数
   ↓
softmax
   ↓
得到动作概率
Alphazero
```
```
MCTS
 │
 └── Neural Network
        │
     ┌──┴──┐
     ↓     ↓
  Policy  Value
```
神经网络也会输出策略和价值，所以这个过程的自对弈和AlphaGO有些区别，下面是代码
```python
class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
```
AlphaZero 自对弈时，通常可以理解为两个 MCTSPlayer 轮流下棋；每个 MCTSPlayer 内部各自持有一个 MCTS 搜索树，而两个 MCTS 使用同一个 Policy-Value 神经网络。有点像自我对抗的意思了，我觉得主要在这里，因为如果对手和自己旗鼓相当，对于alphazero来说就是学习最好的时候，这些数据也最有用。

如果AlphaGO去掉刚开始的行为克隆，那其实也无法战胜alphazero，主要在于alphazero的对手太强大了就是它自己，再加上alphazero的算力更充足更大，人类是无法战胜alphazero的，就像没有人能一直跑过汽车，算力和脑力相比，就如同发动机和全身肌肉相比。













