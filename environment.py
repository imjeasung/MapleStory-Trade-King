# /environment.py

import numpy as np

class GameEnvironment:
    """
    강화학습 게임의 환경을 정의하는 클래스.
    (솔 에르다 조각 보상 체계로 수정됨)
    """
    def __init__(self, max_attempts=100):
        self.max_attempts = max_attempts
        
        # 레벨별 [성공, 실패, 조난] 확률 정의
        self.probabilities = {
            1: [1.0, 0.0, 0.0],
            2: [0.6, 0.4, 0.0],
            3: [0.5, 0.5, 0.0],
            4: [0.4, 0.6, 0.0],
            5: [0.307, 0.693, 0.0],
            6: [0.205, 0.765, 0.03],
            7: [0.103, 0.857, 0.04],
            8: [0.05, 0.90, 0.05],
        }
        
        # 💡 주요 변경 사항: 레벨별 '솔 에르다 조각' 보상 맵
        self.reward_map = {
            1: 0, 
            2: 1,
            3: 3,
            4: 6,
            5: 10,
            6: 15,
            7: 25,
            8: 150,
            9: 500,
            0: -500 # 조난 시 큰 페널티 (최대 보상만큼의 손실)
        }
        
        self.reset()

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.level = 1
        self.attempts_left = self.max_attempts
        self.is_done = False
        return (self.level, self.attempts_left)

    def step(self, action):
        """
        에이전트의 행동(action)을 받아 게임을 한 스텝 진행합니다.
        """
        if self.is_done:
            return (self.level, self.attempts_left), 0, True

        # 💡 주요 변경 사항: 보상 계산 로직 수정
        if action == 'stop':
            self.is_done = True
            # stop을 선택하면 현재 레벨에 해당하는 조각 수를 보상으로 받음
            reward = self.reward_map.get(self.level, 0)
            return (self.level, self.attempts_left), reward, self.is_done

        if action == 'up':
            if self.level == 9: 
                self.is_done = True
                return (self.level, self.attempts_left), self.reward_map[9], self.is_done

            self.attempts_left -= 1
            
            probs = self.probabilities[self.level]
            outcome = np.random.choice(['success', 'fail', 'distress'], p=probs)
            
            reward = 0  # 단계 진행 중에는 보상 없음

            if outcome == 'success':
                self.level += 1
                if self.level == 9:
                    # 최고 레벨 달성 시 조각 보상
                    reward = self.reward_map[9]
                    self.is_done = True
            elif outcome == 'fail':
                if self.level > 2:
                    self.level -= 1
            elif outcome == 'distress':
                # 조난 시 페널티
                self.level = 0
                reward = self.reward_map[0] 
                self.is_done = True

            if self.attempts_left == 0 and not self.is_done:
                self.is_done = True
                # 시도 횟수 소진 시, 최종 레벨의 조각 수를 보상으로 받음
                reward = self.reward_map.get(self.level, 0)

            return (self.level, self.attempts_left), reward, self.is_done
        
        return (self.level, self.attempts_left), 0, self.is_done