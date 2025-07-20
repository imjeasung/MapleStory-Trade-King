# /evaluate.py

import os
from collections import defaultdict
from environment import GameEnvironment
from agent import QLearningAgent

def evaluate_agent(agent, num_episodes=10000):
    """
    학습된 에이전트의 성능을 시뮬레이션을 통해 평가합니다.
    """
    env = GameEnvironment()
    
    total_final_level = 0
    distress_count = 0
    max_level_count = 0
    final_level_distribution = defaultdict(int)

    print(f"\n📊 {num_episodes}회의 시뮬레이션을 통해 AI 성능 평가를 시작합니다...")

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 학습된 정책(Q-테이블)에 따라 최적의 행동만 선택
            action = agent.choose_action(state)
            state, _, done = env.step(action)
        
        # 에피소드 종료 후 결과 기록
        final_level = env.level
        total_final_level += final_level
        final_level_distribution[final_level] += 1
        
        if final_level == 0:
            distress_count += 1
        elif final_level == 9:
            max_level_count += 1

    # 평가 결과 출력
    avg_level = total_final_level / num_episodes
    distress_rate = (distress_count / num_episodes) * 100
    max_level_rate = (max_level_count / num_episodes) * 100

    print("\n--- AI 성능 평가 결과 ---")
    print(f"평균 도달 레벨: {avg_level:.2f}")
    print(f"최고 레벨(9) 도달 확률: {max_level_rate:.2f}%")
    print(f"조난(레벨 0) 확률: {distress_rate:.2f}%")
    print("\n--- 레벨별 종료 분포 ---")
    for level, count in sorted(final_level_distribution.items()):
        rate = (count / num_episodes) * 100
        print(f"레벨 {level}: {count}회 ({rate:.2f}%)")
    print("-------------------------\n")


def play_with_agent(agent):
    """
    사용자가 직접 게임을 플레이하며 AI의 추천과 비교합니다.
    """
    env = GameEnvironment()
    state = env.reset()
    done = False

    print("\n🎮 AI와 함께 플레이하기를 시작합니다.")
    print("매 턴마다 AI의 추천을 보고 당신의 선택을 결정하세요. ('u' = up, 's' = stop)")

    while not done:
        level, attempts_left = state
        
        print("\n" + "="*30)
        print(f"현재 상태: 레벨 {level}, 남은 횟수 {attempts_left}")
        
        # AI의 추천 행동 표시
        ai_action = agent.choose_action(state).upper()
        print(f"🤖 AI의 추천: {ai_action}")
        
        # 사용자 입력 받기
        user_choice = ''
        while user_choice not in ['u', 's']:
            user_choice = input("당신의 선택은? [u/s]: ").lower()

        action = 'up' if user_choice == 'u' else 'stop'
        
        # 사용자의 선택으로 게임 진행
        state, reward, done = env.step(action)
        
        # 결과 출력
        if action == 'stop':
            print(f"\n🛑 게임을 중단했습니다. 최종 레벨: {env.level}")
        elif reward == -100:
            print(f"\n💥 조난당했습니다! 게임이 종료됩니다.")
        elif env.level > level:
             print(f"\n🎉 성공! 레벨이 {env.level}로 상승했습니다.")
        elif env.level < level:
             print(f"\n😢 실패... 레벨이 {env.level}로 하락했습니다.")
        else: # 레벨 2에서 실패한 경우
             print(f"\n😅 실패... 레벨이 유지됩니다.")

    print("\n--- 게임 종료 ---")


def main():
    """
    메인 실행 함수: 모드 선택 메뉴 제공
    """
    Q_TABLE_FILE = "q_table.pkl"
    if not os.path.exists(Q_TABLE_FILE):
        print(f"오류: 모델 파일({Q_TABLE_FILE})을 찾을 수 없습니다.")
        print("'train.py'를 실행하여 모델을 먼저 학습시켜 주세요.")
        return

    agent = QLearningAgent()
    agent.load_q_table(Q_TABLE_FILE)
    
    while True:
        print("\n--- 모델 평가 및 테스트 ---")
        print("1. AI 성능 자동 평가 (10,000회 시뮬레이션)")
        print("2. AI와 함께 플레이하기")
        print("3. 종료")
        choice = input("원하는 모드를 선택하세요 [1/2/3]: ")

        if choice == '1':
            evaluate_agent(agent)
        elif choice == '2':
            play_with_agent(agent)
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 입력입니다. 1, 2, 3 중에서 선택해주세요.")


if __name__ == "__main__":
    main()