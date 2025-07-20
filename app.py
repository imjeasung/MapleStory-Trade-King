# /app.py

from flask import Flask, render_template, request, flash
from agent import QLearningAgent
import os

# Flask 애플리케이션 생성
app = Flask(__name__)
# flash 메시지를 사용하기 위한 secret_key 설정
app.secret_key = 'supersecretkey'

# Q-러닝 에이전트 생성 및 학습된 Q-테이블 불러오기
agent = QLearningAgent()
Q_TABLE_FILE = "q_table.pkl"

if os.path.exists(Q_TABLE_FILE):
    agent.load_q_table(Q_TABLE_FILE)
    print("성공: 학습된 모델(q_table.pkl)을 불러왔습니다.")
else:
    print(f"경고: {Q_TABLE_FILE} 파일을 찾을 수 없습니다. 'train.py'를 먼저 실행하여 모델을 생성해주세요.")
    # 파일이 없을 경우, 조언 기능이 제대로 동작하지 않으므로 빈 테이블로 시작
    # 실제 운영 시에는 여기서 서버를 중단하거나 에러 처리를 하는 것이 좋음

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/get_advice', methods=['POST'])
def get_advice():
    """사용자 입력을 받아 최적의 행동을 조언하고 결과를 페이지에 표시합니다."""
    try:
        # form에서 'level'과 'attempts' 값을 정수형으로 가져옴
        level = int(request.form['level'])
        attempts = int(request.form['attempts'])

        # 현재 상태 정의
        state = (level, attempts)
        
        # Q-테이블에서 현재 상태에 대한 Q-값들을 가져옴
        # defaultdict 이므로 키가 없으면 [0.0, 0.0]을 반환
        q_values = agent.q_table[state]
        
        up_value = q_values[agent.action_map['up']]
        stop_value = q_values[agent.action_map['stop']]

        # Q-값이 더 높은 행동을 추천
        if up_value > stop_value:
            recommendation = 'UP'
        else:
            recommendation = 'STOP'
            
        # 계산된 추천 결과를 포함하여 페이지를 다시 렌더링
        return render_template('index.html', recommendation=recommendation)

    except (ValueError, KeyError):
        # 입력값이 유효하지 않은 경우
        flash("유효한 숫자를 입력해주세요.", "error")
        return render_template('index.html')

if __name__ == '__main__':
    # Flask 개발 서버 실행 (디버그 모드 활성화)
    app.run(debug=True)