"""
优化版井字棋强化学习系统
基于原代码重构，包含数据结构优化、算法改进、模块化设计等
"""

import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import yaml
from pathlib import Path
from enum import IntEnum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 配置类
@dataclass
class Config:
    """训练配置参数"""
    total_trials: int = 30000
    epsilon_start: float = 0.1
    epsilon_end: float = 0.0
    epsilon_decay_step: int = 20000
    learning_rate: float = 0.1
    eval_window_size: int = 500
    eval_step_size: int = 250
    patience: int = 10
    eval_interval: int = 1000
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, filepath: str):
        """从YAML文件加载配置"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, filepath: str):
        """保存配置到YAML文件"""
        with open(filepath, 'w') as f:
            yaml.dump(self.__dict__, f)


# 游戏常量
class GameResult(IntEnum):
    """游戏结果枚举"""
    ONGOING = 0
    PLAYER1_WIN = 1
    PLAYER2_WIN = 2
    DRAW = 3


class GameLogic:
    """游戏逻辑类，处理棋盘操作和胜负判断"""

    WINNING_LINES = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
        [0, 4, 8], [2, 4, 6]  # 对角线
    ]

    SYMMETRY_TRANSFORMS = [
        lambda x: x,  # 原始
        lambda x: [x[2], x[5], x[8], x[1], x[4], x[7], x[0], x[3], x[6]],  # 旋转90度
        lambda x: [x[8], x[7], x[6], x[5], x[4], x[3], x[2], x[1], x[0]],  # 旋转180度
        lambda x: [x[6], x[3], x[0], x[7], x[4], x[1], x[8], x[5], x[2]],  # 旋转270度
        lambda x: [x[2], x[1], x[0], x[5], x[4], x[3], x[8], x[7], x[6]],  # 垂直翻转
        lambda x: [x[6], x[7], x[8], x[3], x[4], x[5], x[0], x[1], x[2]],  # 水平翻转
        lambda x: [x[0], x[3], x[6], x[1], x[4], x[7], x[2], x[5], x[8]],  # 主对角线翻转
        lambda x: [x[8], x[5], x[2], x[7], x[4], x[1], x[6], x[3], x[0]]  # 副对角线翻转
    ]

    @staticmethod
    def get_canonical_state(state: np.ndarray) -> tuple:
        """获取状态的规范形式（考虑对称性）"""
        state_list = state.tolist()
        canonical = min(
            tuple(transform(state_list))
            for transform in GameLogic.SYMMETRY_TRANSFORMS
        )
        return canonical

    @staticmethod
    def get_available_moves(state: np.ndarray) -> np.ndarray:
        """获取可用的移动位置"""
        return np.where(state == 0)[0]

    @staticmethod
    def check_winner(state: np.ndarray, player_index: int) -> bool:
        """检查指定玩家是否获胜"""
        for line in GameLogic.WINNING_LINES:
            if all(state[pos] == player_index for pos in line):
                return True
        return False

    @staticmethod
    def is_terminal(state: np.ndarray) -> Tuple[bool, int]:
        """检查是否为终止状态，返回(是否终止, 获胜者)"""
        # 检查玩家1是否获胜
        if GameLogic.check_winner(state, 1):
            return True, GameResult.PLAYER1_WIN
        # 检查玩家2是否获胜
        if GameLogic.check_winner(state, 2):
            return True, GameResult.PLAYER2_WIN
        # 检查是否平局
        if 0 not in state:
            return True, GameResult.DRAW
        # 游戏继续
        return False, GameResult.ONGOING

    @staticmethod
    def get_state_key(state: np.ndarray) -> str:
        """获取状态的字符串表示（用于字典键）"""
        return ''.join(str(int(x)) for x in state)


class Agent:
    """强化学习智能体"""

    def __init__(self, player_index: int, epsilon: float = 0.1,
                 alpha: float = 0.1, use_symmetry: bool = True):
        """
        初始化智能体

        Args:
            player_index: 玩家标识 (1或2)
            epsilon: 探索率
            alpha: 学习率
            use_symmetry: 是否使用对称性优化
        """
        self.player_index = player_index
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_symmetry = use_symmetry

        # 使用字典存储状态价值，节省内存
        self.value_table = defaultdict(float)
        self.stored_state = None
        self.stored_state_key = None

        # 训练统计
        self.training_steps = 0

        logger.info(f"Agent {player_index} initialized with epsilon={epsilon}, alpha={alpha}")

    def reset(self):
        """重置智能体状态"""
        self.stored_state = None
        self.stored_state_key = None

    def get_state_value(self, state: np.ndarray) -> float:
        """获取状态价值"""
        if self.use_symmetry:
            state_key = GameLogic.get_canonical_state(state)
        else:
            state_key = tuple(state)

        return self.value_table[state_key]

    def set_state_value(self, state: np.ndarray, value: float):
        """设置状态价值"""
        if self.use_symmetry:
            state_key = GameLogic.get_canonical_state(state)
        else:
            state_key = tuple(state)

        self.value_table[state_key] = value

    def choose_action(self, state: np.ndarray, available_moves: np.ndarray,
                      training: bool = True) -> int:
        """选择行动"""
        current_epsilon = self.epsilon if training else 0.0

        # ε-greedy策略
        if training and np.random.random() < current_epsilon:
            # 随机探索
            return np.random.choice(available_moves)
        else:
            # 选择最优行动
            best_move = None
            best_value = -np.inf

            for move in available_moves:
                # 模拟执行行动
                next_state = state.copy()
                next_state[move] = self.player_index

                # 获取状态价值
                state_value = self.get_state_value(next_state)

                if state_value > best_value:
                    best_value = state_value
                    best_move = move

            # 如果所有行动价值相同，随机选择一个
            if best_move is None:
                return np.random.choice(available_moves)

            return best_move

    def update_value(self, next_state: np.ndarray, reward: float = 0.0):
        """更新价值函数"""
        if self.stored_state is None or self.stored_state_key is None:
            return

        # 计算TD误差
        current_value = self.get_state_value(next_state)
        previous_value = self.get_state_value(self.stored_state)
        td_error = reward + current_value - previous_value

        # 更新价值
        new_value = previous_value + self.alpha * td_error
        self.set_state_value(self.stored_state, new_value)

    def move(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """执行移动并更新学习"""
        available_moves = GameLogic.get_available_moves(state)

        if len(available_moves) == 0:
            return state.copy()

        # 存储当前状态（用于后续学习）
        self.stored_state = state.copy()
        self.stored_state_key = GameLogic.get_state_key(state)

        # 选择行动
        chosen_move = self.choose_action(state, available_moves, training)

        # 执行行动
        next_state = state.copy()
        next_state[chosen_move] = self.player_index

        # 更新价值函数
        self.update_value(next_state)

        return next_state

    def save(self, filepath: str):
        """保存智能体状态"""
        data = {
            'player_index': self.player_index,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'value_table': dict(self.value_table),
            'training_steps': self.training_steps
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """加载智能体状态"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.player_index = data['player_index']
        self.epsilon = data['epsilon']
        self.alpha = data['alpha']
        self.value_table = defaultdict(float, data['value_table'])
        self.training_steps = data['training_steps']
        logger.info(f"Agent loaded from {filepath}")


class TicTacToeGame:
    """井字棋游戏管理器"""

    def __init__(self, agent1: Agent, agent2: Agent, config: Config):
        self.agent1 = agent1
        self.agent2 = agent2
        self.config = config
        self.winner_history = []
        self.state_history = []

        # 创建检查点目录
        if config.save_checkpoints:
            Path(config.checkpoint_dir).mkdir(exist_ok=True)

    def play_game(self, training: bool = True) -> int:
        """进行一局游戏"""
        # 重置智能体
        self.agent1.reset()
        self.agent2.reset()

        # 初始化棋盘
        state = np.zeros(9, dtype=np.int8)
        current_player = self.agent1
        opponent = self.agent2
        winner = GameResult.ONGOING

        move_count = 0
        game_states = [state.copy()]

        while winner == GameResult.ONGOING:
            # 当前玩家行动
            next_state = current_player.move(state, training)
            move_count += 1

            # 检查游戏是否结束
            is_terminal, game_result = GameLogic.is_terminal(next_state)

            if is_terminal:
                winner = game_result

                # 给获胜玩家奖励
                if winner == GameResult.PLAYER1_WIN:
                    self.agent1.update_value(next_state, reward=1.0)
                    self.agent2.update_value(state, reward=-1.0)
                elif winner == GameResult.PLAYER2_WIN:
                    self.agent2.update_value(next_state, reward=1.0)
                    self.agent1.update_value(state, reward=-1.0)
                elif winner == GameResult.DRAW:
                    # 平局给予小奖励
                    self.agent1.update_value(next_state, reward=0.1)
                    self.agent2.update_value(next_state, reward=0.1)

            # 交换玩家
            current_player, opponent = opponent, current_player
            state = next_state
            game_states.append(state.copy())

        # 记录游戏历史
        self.winner_history.append(winner)
        self.state_history.append(game_states)

        return winner

    def train(self) -> Dict[str, List[float]]:
        """训练智能体"""
        logger.info("开始训练...")

        best_win_rate = 0
        patience_counter = 0
        stats = {
            'agent1_win_rate': [],
            'agent2_win_rate': [],
            'draw_rate': [],
            'agent1_value_size': [],
            'agent2_value_size': []
        }

        for trial in range(self.config.total_trials):
            # 动态调整epsilon
            if trial < self.config.epsilon_decay_step:
                decay = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_step
                self.agent1.epsilon = self.config.epsilon_start - decay * trial
                self.agent2.epsilon = self.config.epsilon_start - decay * trial
            else:
                self.agent1.epsilon = self.config.epsilon_end
                self.agent2.epsilon = self.config.epsilon_end

            # 进行一局游戏
            self.play_game(training=True)

            # 定期评估和保存
            if (trial + 1) % self.config.eval_interval == 0:
                win_rates = self.evaluate_recent(self.config.eval_window_size)
                agent1_win_rate = win_rates[GameResult.PLAYER1_WIN]

                # 记录统计信息
                stats['agent1_win_rate'].append(agent1_win_rate)
                stats['agent2_win_rate'].append(win_rates[GameResult.PLAYER2_WIN])
                stats['draw_rate'].append(win_rates[GameResult.DRAW])
                stats['agent1_value_size'].append(len(self.agent1.value_table))
                stats['agent2_value_size'].append(len(self.agent2.value_table))

                logger.info(f"Trial {trial + 1}/{self.config.total_trials} - "
                            f"Agent1 Win Rate: {agent1_win_rate:.3f}, "
                            f"Value Table Size: {len(self.agent1.value_table)}")

                # 早停检查
                if agent1_win_rate > best_win_rate:
                    best_win_rate = agent1_win_rate
                    patience_counter = 0

                    # 保存检查点
                    if self.config.save_checkpoints:
                        self.save_checkpoint(trial + 1)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        logger.info(f"早停触发于第 {trial + 1} 次训练")
                        break

        logger.info("训练完成!")
        return stats

    def evaluate_recent(self, window_size: int = 500) -> Dict[int, float]:
        """评估最近若干局的表现"""
        if len(self.winner_history) < window_size:
            window = self.winner_history
        else:
            window = self.winner_history[-window_size:]

        total_games = len(window)
        win_rates = {
            GameResult.PLAYER1_WIN: sum(1 for w in window if w == GameResult.PLAYER1_WIN) / total_games,
            GameResult.PLAYER2_WIN: sum(1 for w in window if w == GameResult.PLAYER2_WIN) / total_games,
            GameResult.DRAW: sum(1 for w in window if w == GameResult.DRAW) / total_games
        }

        return win_rates

    def save_checkpoint(self, trial: int):
        """保存检查点"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{trial}.pkl"
        data = {
            'trial': trial,
            'agent1': self.agent1,
            'agent2': self.agent2,
            'winner_history': self.winner_history,
            'config': self.config
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"检查点保存于 {checkpoint_path}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.agent1 = data['agent1']
        self.agent2 = data['agent2']
        self.winner_history = data['winner_history']
        self.config = data['config']
        logger.info(f"检查点加载自 {filepath}，训练步数: {data['trial']}")


class Visualizer:
    """可视化类"""

    @staticmethod
    def plot_training_stats(stats: Dict[str, List[float]], config: Config):
        """绘制训练统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 胜率图
        x_vals = np.arange(len(stats['agent1_win_rate'])) * config.eval_interval
        axes[0, 0].plot(x_vals, stats['agent1_win_rate'], 'b-', linewidth=2, label='Agent1 Win Rate')
        axes[0, 0].plot(x_vals, stats['agent2_win_rate'], 'r-', linewidth=2, label='Agent2 Win Rate')
        axes[0, 0].plot(x_vals, stats['draw_rate'], 'y-', linewidth=2, label='Draw Rate')
        axes[0, 0].set_xlabel('Training Trials')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_title('Win/Draw Rates Over Training')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 价值表大小
        axes[0, 1].plot(x_vals, stats['agent1_value_size'], 'b-', linewidth=2, label='Agent1')
        axes[0, 1].plot(x_vals, stats['agent2_value_size'], 'r--', linewidth=2, label='Agent2')
        axes[0, 1].set_xlabel('Training Trials')
        axes[0, 1].set_ylabel('Value Table Size')
        axes[0, 1].set_title('State Value Table Size')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 最近500局胜率分布
        if len(stats['agent1_win_rate']) > 0:
            recent_size = min(20, len(stats['agent1_win_rate']))
            recent_agent1 = stats['agent1_win_rate'][-recent_size:]
            recent_agent2 = stats['agent2_win_rate'][-recent_size:]
            recent_draw = stats['draw_rate'][-recent_size:]

            x = np.arange(recent_size)
            width = 0.25
            axes[1, 0].bar(x - width, recent_agent1, width, label='Agent1', color='blue', alpha=0.7)
            axes[1, 0].bar(x, recent_agent2, width, label='Agent2', color='red', alpha=0.7)
            axes[1, 0].bar(x + width, recent_draw, width, label='Draw', color='yellow', alpha=0.7)
            axes[1, 0].set_xlabel('Recent Evaluation Points')
            axes[1, 0].set_ylabel('Rate')
            axes[1, 0].set_title('Recent Performance (Last 20 Evaluations)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 示例棋盘
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Example Board States:', fontsize=12, fontweight='bold')

        # 绘制简单棋盘示例
        example_board = np.array([1, 0, 2, 0, 1, 0, 2, 0, 1])
        Visualizer._draw_board(axes[1, 1], example_board, x_offset=0.1, y_offset=0.7, size=0.2)
        axes[1, 1].text(0.1, 0.65, 'Player1: X, Player2: O', fontsize=10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _draw_board(ax, board: np.ndarray, x_offset: float, y_offset: float, size: float):
        """绘制棋盘"""
        cell_size = size / 3

        # 绘制网格
        for i in range(4):
            ax.plot([x_offset, x_offset + size],
                    [y_offset + i * cell_size, y_offset + i * cell_size], 'k-', linewidth=2)
            ax.plot([x_offset + i * cell_size, x_offset + i * cell_size],
                    [y_offset, y_offset + size], 'k-', linewidth=2)

        # 绘制棋子
        for i in range(9):
            row = i // 3
            col = i % 3

            center_x = x_offset + col * cell_size + cell_size / 2
            center_y = y_offset + (2 - row) * cell_size + cell_size / 2

            if board[i] == 1:  # Player 1 (X)
                offset = cell_size / 4
                ax.plot([center_x - offset, center_x + offset],
                        [center_y - offset, center_y + offset], 'b-', linewidth=3)
                ax.plot([center_x - offset, center_x + offset],
                        [center_y + offset, center_y - offset], 'b-', linewidth=3)
            elif board[i] == 2:  # Player 2 (O)
                radius = cell_size / 4
                circle = plt.Circle((center_x, center_y), radius, color='red', fill=False, linewidth=3)
                ax.add_patch(circle)

    @staticmethod
    def plot_winning_rates_original_style(winner_history: List[int], config: Config):
        """按照原始代码风格绘制胜率图"""
        trial = len(winner_history)
        step = config.eval_step_size
        duration = config.eval_window_size

        def calculate_rates(history):
            rates1 = []
            rates2 = []
            rates3 = []

            for i in range(0, trial - duration + 1, step):
                window = history[i:i + duration]
                rates1.append(sum(1 for w in window if w == GameResult.PLAYER1_WIN) / duration)
                rates2.append(sum(1 for w in window if w == GameResult.PLAYER2_WIN) / duration)
                rates3.append(sum(1 for w in window if w == GameResult.DRAW) / duration)

            return rates1, rates2, rates3

        if trial >= duration:
            rate1, rate2, rate3 = calculate_rates(winner_history)

            fig, ax = plt.subplots(figsize=(12, 7))
            x_vals = np.arange(len(rate1)) * step / 1000

            plt.plot(x_vals, rate1, linewidth=4, marker='.', markersize=20,
                     color="#0071B7", label="Agent1")
            plt.plot(x_vals, rate2, linewidth=4, marker='.', markersize=20,
                     color="#DB2C2C", label="Agent2")
            plt.plot(x_vals, rate3, linewidth=4, marker='.', markersize=20,
                     color="#FAB70D", label="Draw")

            max_x = max(x_vals) if len(x_vals) > 0 else 30
            plt.xticks(np.arange(0, max_x + 1, max_x / 3),
                       np.round(np.arange(0, max_x + 1, max_x / 3), 1),
                       fontsize=20)
            plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)
            plt.xlabel("Trials (x1k)", fontsize=24)
            plt.ylabel("Winning Rate", fontsize=24)
            plt.legend(loc="best", fontsize=18)
            plt.tick_params(width=3, length=8)
            ax.spines[:].set_linewidth(3)
            plt.grid(True, alpha=0.3)
            plt.title("Training Progress (Original Style)", fontsize=26, pad=20)

            plt.tight_layout()
            plt.show()
        else:
            logger.warning(f"历史数据不足，需要至少 {duration} 局游戏，当前只有 {trial} 局")


def main():
    """主函数"""
    # 创建配置
    config = Config(
        total_trials=30000,
        epsilon_start=0.1,
        epsilon_end=0.0,
        epsilon_decay_step=20000,
        learning_rate=0.1,
        eval_window_size=500,
        eval_step_size=250,
        patience=10,
        eval_interval=1000,
        save_checkpoints=True,
        checkpoint_dir="checkpoints"
    )

    # 保存配置到文件
    config.to_yaml("config.yaml")
    logger.info("配置已保存到 config.yaml")

    # 创建智能体
    agent1 = Agent(player_index=1, epsilon=config.epsilon_start,
                   alpha=config.learning_rate, use_symmetry=True)
    agent2 = Agent(player_index=2, epsilon=config.epsilon_start,
                   alpha=config.learning_rate, use_symmetry=True)

    # 创建游戏管理器
    game = TicTacToeGame(agent1, agent2, config)

    # 训练
    stats = game.train()

    # 可视化结果
    Visualizer.plot_training_stats(stats, config)
    Visualizer.plot_winning_rates_original_style(game.winner_history, config)

    # 保存最终模型
    agent1.save("agent1_final.pkl")
    agent2.save("agent2_final.pkl")

    # 输出最终统计
    final_stats = game.evaluate_recent(1000)
    logger.info(f"最终统计 (最近1000局):")
    logger.info(f"  Agent1 胜率: {final_stats[GameResult.PLAYER1_WIN]:.3f}")
    logger.info(f"  Agent2 胜率: {final_stats[GameResult.PLAYER2_WIN]:.3f}")
    logger.info(f"  平局率: {final_stats[GameResult.DRAW]:.3f}")
    logger.info(f"  Agent1 状态表大小: {len(agent1.value_table)}")
    logger.info(f"  Agent2 状态表大小: {len(agent2.value_table)}")


if __name__ == "__main__":
    main()