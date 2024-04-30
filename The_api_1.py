from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
from catboost import CatBoostRegressor, Pool
import pandas as pd
# from tensorflow.keras.models import load_model

import catboost

# catboost 모델 로드
loaded_model = catboost.CatBoost()
loaded_model.load_model('bbbro_sample_model.h5')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

logging.basicConfig(level=logging.INFO)

sample_data = {
    'GENDER': '여',
    'AGE_GRP': 30.0,
    'TRAVEL_STYL_1': 3,# 자연 vs 도시
    'TRAVEL_STYL_2': 2,# 숙박 vs 당일
    'TRAVEL_STYL_3': 1,# 새로운지역 vs ~
    'TRAVEL_STYL_4': 2,
    'TRAVEL_STYL_5': 6,
    'TRAVEL_STYL_6': 4,
    'TRAVEL_STYL_7': 2,
    'TRAVEL_STYL_8': 7,
    'TRAVEL_MOTIVE_1': 1,
    'TRAVEL_COMPANIONS_NUM': 1.0,
    'TRAVEL_MISSION_INT': 22,
    'VISIT_AREA_NM': 'whatever',
    'DGSTFN': 4.0
}

DB = {
    'GENDER': ['남', '여', '여', '남', '남', '남', '여', '여', '남', '남'],
    'AGE_GRP': [30.0, 40.0, 30.0, 30.0, 20.0, 40.0, 30.0, 40.0, 20.0, 30.0],
    'TRAVEL_STYL_1': [4, 4, 4, 7, 6, 6, 1, 2, 7, 3],
    'TRAVEL_STYL_2': [4, 3, 4, 1, 1, 6, 3, 2, 4, 3],
    'TRAVEL_STYL_3': [2, 6, 4, 1, 1, 2, 1, 1, 3, 5],
    'TRAVEL_STYL_4': [4, 3, 3, 1, 3, 4, 4, 2, 4, 5],
    'TRAVEL_STYL_5': [6, 2, 4, 7, 4, 4, 4, 2, 3, 1],
    'TRAVEL_STYL_6': [5, 2, 4, 7, 6, 2, 2, 2, 7, 5],
    'TRAVEL_STYL_7': [6, 7, 5, 2, 3, 4, 1, 1, 6, 5],
    'TRAVEL_STYL_8': [6, 2, 7, 4, 2, 2, 7, 7, 3, 2],
    'TRAVEL_MOTIVE_1': [1, 6, 1, 1, 3, 1, 3, 3, 2, 6],
    'TRAVEL_COMPANIONS_NUM': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0],
    'TRAVEL_MISSION_INT': [5, 28, 6, 2, 11, 1, 3, 9, 6, 21],
    'VISIT_AREA_NM': ['프로방스마을', '병점역 1호선', '더현대서울', '강릉중앙시장', '청계천', '새절역 6호선', '장안순대국', '유일닭강정', '숙이네닭발', '삼송역 3호선'],
    'DGSTFN': [4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 3.0]
}

@app.route('/')
def index():
    return 'Server is running'

@app.route('/vote', methods=["POST"])
def predict():
    # JSON 데이터를 받음
    data = request.get_json()
    logging.info("Received vote data: %s", data)

    if not isinstance(data, list):
        logging.error("Data format error: Data should be a list")
        return jsonify({'error': 'Data format error: Data should be a list'}), 400

    # 예시 데이터
    # data = [
    #     {'answer': 'Y', 'questionId': 1},
    #     {'answer': 'Y', 'questionId': 2},
    #     {'answer': 'Y', 'questionId': 3},
    #     {'answer': 'Y', 'questionId': 4}
    # ]

# 질문 ID에 따라 답변을 매핑하는 딕셔너리 생성
    answer_mapping = {item['questionId']: (2 if item['answer'] == 'Y' else 8) for item in data}

    results = pd.DataFrame([], columns=['AREA', 'SCORE'])
    for item in data:
        # 모델 입력을 위한 사용자 데이터 준비
        input_data = [
            '여',
            30,
            answer_mapping.get(1, 3),  # TRAVEL_STYL_1, 기본값 3
            answer_mapping.get(2, 2),  # TRAVEL_STYL_2, 기본값 2
            answer_mapping.get(3, 1),  # TRAVEL_STYL_3, 기본값 1
            answer_mapping.get(4, 2),  # TRAVEL_STYL_4, 기본값 2
            6, # TRAVEL_STYL_5 고정 값
            4, # TRAVEL_STYL_6 고정 값
            2, # TRAVEL_STYL_7 고정 값
            7, # TRAVEL_STYL_8 고정 값
            1, # TRAVEL_MOTIVE_1 고정 값
            1, # TRAVEL_COMPANIONS_NUM 고정 값
            22, # TRAVEL_MISSION_INT 고정 값
            4 # DGSTFN 고정 값
        ]
        # 여기서 모델을 사용하여 점수를 예측 (가정)
        score = loaded_model.predict(input_data)
        
        # 결과에 추가
        results = pd.concat([results, pd.DataFrame([[item['VISIT_AREA_NM'], score]], columns=['AREA', 'SCORE'])])
    
    # 점수가 높은 상위 10개 지역 정렬
    top_results = results.sort_values('SCORE', ascending=False)[:10].to_dict('records')

    # Node.js 서버로 결과 데이터를 전송
    node_js_endpoint = 'http://localhost:3000/receive_data'
    response = requests.post(node_js_endpoint, json=top_results)
    logging.info("Sending data to Node.js server: %s", results)  # 로그에 전송 데이터 기록
    
    if response.status_code == 200:
        logging.info("Data successfully sent to the Node.js server.")
    else:
        logging.error("Failed to send data to the Node.js server.")

    # 결과를 JSON 형태로 반환
    return jsonify(results)

if __name__ == "__main__":
    app.run()