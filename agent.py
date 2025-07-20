# /agent.py

import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    """
    Q-러닝 알고리즘을 구현하는 에이전트 클래스.
    """
    def __init__(self, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay_rate=0.999, min_exploration_rate=0.05):
        """
        에이전트의 하이퍼파라미터와 Q-테이블을 초기화합니다.
        
        Args:
            learning_rate (float): 학습률 (alpha)
            discount_factor (float): 할인율 (gamma)
            exploration_rate (float): 초기 탐험 확률 (epsilon)
            exploration_decay_rate (float): 탐험 확률 감소율
            min_exploration_rate (float): 최소 탐험 확률
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.epsilon_min = min_exploration_rate
        
        # Q-테이블을 defaultdict로 생성하여 새로운 상태를 쉽게 추가
        # 값은 [up_value, stop_value] 형태의 리스트
        self.q_table = defaultdict(lambda: [0.0, 0.0])
        
        # 행동을 인덱스로 변환하기 위한 맵
        self.action_map = {'up': 0, 'stop': 1}
        self.actions = ['up', 'stop']

    def choose_action(self, state):
        """
        Epsilon-Greedy 정책에 따라 행동을 선택합니다.
        
        Args:
            state (tuple): 현재 상태 (level, attempts_left)
        
        Returns:
            str: 선택된 행동 'up' 또는 'stop'
        """
        if np.random.rand() < self.epsilon:
            # 탐험: 무작위로 행동 선택
            return np.random.choice(self.actions)
        else:
            # 활용: Q-값이 가장 높은 행동 선택
            q_values = self.q_table[state]
            return self.actions[np.argmax(q_values)]

    def learn(self, state, action, reward, next_state, done):
        """
        Q-테이블을 벨만 방정식을 이용해 업데이트합니다.
        Q(s, a) = Q(s, a) + alpha * [R + gamma * max_a' Q(s', a') - Q(s, a)]
        
        Args:
            state (tuple): 현재 상태
            action (str): 수행한 행동
            reward (float): 받은 보상
            next_state (tuple): 다음 상태
            done (bool): 게임 종료 여부
        """
        action_idx = self.action_map[action]
        old_value = self.q_table[state][action_idx]
        
        # 다음 상태에서 얻을 수 있는 최대 Q-값
        # 게임이 종료(done)되었으면 미래 가치는 0
        next_max = np.max(self.q_table[next_state]) if not done else 0
        
        # 새로운 Q-값 계산
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action_idx] = new_value

    def decay_epsilon(self):
        """탐험 확률(epsilon)을 감소시킵니다."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename="q_table.pkl"):
        """학습된 Q-테이블을 파일로 저장합니다."""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename="q_table.pkl"):
        """파일에서 Q-테이블을 불러옵니다."""
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: [0.0, 0.0], pickle.load(f))
        self.epsilon = 0  # 불러온 모델은 활용만 하도록 epsilon을 0으로 설정
        print(f"Q-table loaded from {filename}")