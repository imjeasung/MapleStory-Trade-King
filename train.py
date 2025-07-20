# /train.py

from environment import GameEnvironment
from agent import QLearningAgent
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def plot_results(q_table, rewards_per_episode):
    """
    학습이 끝난 후 결과를 시각화하는 함수.
    1. 에피소드별 보상 변화 그래프
    2. 학습된 정책 히트맵
    """
    plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
    # plt.rcParams['font.family'] = 'AppleGothic' # Mac
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 보상 변화 그래프 (이동 평균 사용)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # 노이즈를 줄이기 위해 1000개 구간의 이동 평균을 계산
    moving_avg_rewards = np.convolve(rewards_per_episode, np.ones(1000)/1000, mode='valid')
    plt.plot(moving_avg_rewards)
    plt.title("에피소드별 보상 변화 (이동 평균)")
    plt.xlabel("에피소드 (x1000 구간)")
    plt.ylabel("평균 보상")
    plt.grid(True)

    # 2. 정책 히트맵
    plt.subplot(1, 2, 2)
    # (레벨, 남은횟수)에 따른 최적 행동(0=stop, 1=up)을 저장할 배열
    policy_map = np.zeros((8, 100)) 
    for level in range(1, 9): # 레벨 1 ~ 8
        for attempts in range(1, 101): # 남은 횟수 1 ~ 100
            state = (level, attempts)
            if q_table[state][0] < q_table[state][1]: # 'up'의 Q-값이 더 높으면
                policy_map[level-1, attempts-1] = 1 # 'up'을 1로 표시

    plt.imshow(policy_map, cmap='coolwarm', aspect='auto', origin='lower', extent=[1, 100, 1, 8])
    plt.colorbar(ticks=[0, 1]).set_ticklabels(['STOP', 'UP'])
    plt.title("학습된 최적 정책")
    plt.xlabel("남은 시도 횟수")
    plt.ylabel("현재 레벨")
    
    plt.tight_layout()
    plt.show()


def train():
    """
    QLearningAgent를 학습시키는 메인 함수
    """
    env = GameEnvironment(max_attempts=100)
    agent = QLearningAgent(
        learning_rate=0.1, 
        discount_factor=0.99, 
        exploration_rate=1.0, 
        exploration_decay_rate=0.9999,
        min_exploration_rate=0.01
    )
    thousand = 10000000
    num_episodes = thousand*5
    rewards_per_episode = [] # 에피소드별 총 보상을 저장할 리스트

    print("Starting training...")
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        rewards_per_episode.append(total_reward)
        agent.decay_epsilon()

    print("Training finished.")
    agent.save_q_table("q_table.pkl")
    
    # 학습 완료 후 결과 시각화 함수 호출
    print("Plotting results...")
    plot_results(agent.q_table, rewards_per_episode)


if __name__ == "__main__":
    train()