# from flask import Flask
# from flask import jsonify, request
# from flask_cors import CORS, cross_origin
# app = Flask(__name__)
# import tensorflow as tf
# import transformers
# import numpy as np
# import re
# import pandas as pd
# import catboost
# from catboost import CatBoostRegressor
#
# # if __name__ == '__main__':
# #     app.run()
#
# loaded_model = CatBoostRegressor().load_model('bbbro_sample_model.h5')
# # @app.route('/predict', method = ["GET", "POST"])
# # def predict():
# #     data = {"requested" : "request"}
# #
# #     params = request.json
# #     sample_title = params["input"]
#
#
# # catboost 모델 로드
# # loaded_model = catboost.CatBoost()
# # loaded_model.load_model('bbbro_sample_model.h5')
#
# loaded_model = CatBoostRegressor().load_model('bbbro_sample_model.h5')
#
# # 샘플 데이터 준비
# sample_data = {
#     'GENDER': '여',
#     'AGE_GRP': 30.0,
#     'TRAVEL_STYL_1': 3,  # 자연 vs 도시
#     'TRAVEL_STYL_2': 2,  # 숙박 vs 당일
#     'TRAVEL_STYL_3': 1,  # 새로운지역 vs ~
#     'TRAVEL_STYL_4': 2,
#     'TRAVEL_STYL_5': 6,
#     'TRAVEL_STYL_6': 4,
#     'TRAVEL_STYL_7': 2,
#     'TRAVEL_STYL_8': 7,
#     'TRAVEL_MOTIVE_1': 1,
#     'TRAVEL_COMPANIONS_NUM': 1.0,
#     'TRAVEL_MISSION_INT': 22,
#     'VISIT_AREA_NM': 'whatever',
#     'DGSTFN': 4.0
# }
#
# # 샘플 데이터를 catboost 모델이 이해할 수 있는 형태로 변환
# # sample_df = pd.DataFrame(sample_data, index=[0])
#
# # 예측 수행
# # score = loaded_model.predict(sample_df)
# # print(f"예측 결과: {score}")
#
#
# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
#
#
# @app.route('/')
# def index():
#     return 'xdd_rat'
#
#
# # @app.route('/predict', methods = ["GET", "POST"])
# # def predict():
# #     data = {"requested" : "request"}
#
# #     params = request.json
# #     sample_df = params["input"]
# #     sample_df = pd.DataFrame(sample_data, index=[0])
# #     score = loaded_model.predict(sample_df)
# #     score=str(int(score * 10000))
# #     data["score"] = score
# #     return jsonify(data)
#
# # if __name__ == "__main__":
# #     app.run()
# # def predict():
# #     data = request.json  # 클라이언트로부터 받은 데이터
# #     sample_df = pd.DataFrame([data])  # 수정된 부분: 직접 제공된 샘플 데이터 대신 요청의 JSON 데이터를 사용
# #     score = loaded_model.predict(sample_df)
# #     score = str(int(score[0] * 10000))  # 수정된 부분: score가 배열로 반환될 수 있으니 첫 번째 요소 사용
# #     return jsonify({"score": score})
#
# # if __name__ == "__main__":
# #     app.run(debug=True)  # 디버그 모드 활성화
#
# # 서버 측에서 미리 정의된 지역명 리스트
#
# # 데이터를 딕셔너리 형태로 정의합니다.
# data = {
#     'GENDER': ['남', '여', '여', '남', '남', '남', '여', '여', '남', '남'],
#     'AGE_GRP': [30.0, 40.0, 30.0, 30.0, 20.0, 40.0, 30.0, 40.0, 20.0, 30.0],
#     'TRAVEL_STYL_1': [4, 4, 4, 7, 6, 6, 1, 2, 7, 3],
#     'TRAVEL_STYL_2': [4, 3, 4, 1, 1, 6, 3, 2, 4, 3],
#     'TRAVEL_STYL_3': [2, 6, 4, 1, 1, 2, 1, 1, 3, 5],
#     'TRAVEL_STYL_4': [4, 3, 3, 1, 3, 4, 4, 2, 4, 5],
#     'TRAVEL_STYL_5': [6, 2, 4, 7, 4, 4, 4, 2, 3, 1],
#     'TRAVEL_STYL_6': [5, 2, 4, 7, 6, 2, 2, 2, 7, 5],
#     'TRAVEL_STYL_7': [6, 7, 5, 2, 3, 4, 1, 1, 6, 5],
#     'TRAVEL_STYL_8': [6, 2, 7, 4, 2, 2, 7, 7, 3, 2],
#     'TRAVEL_MOTIVE_1': [1, 6, 1, 1, 3, 1, 3, 3, 2, 6],
#     'TRAVEL_COMPANIONS_NUM': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0],
#     'TRAVEL_MISSION_INT': [5, 28, 6, 2, 11, 1, 3, 9, 6, 21],
#     'VISIT_AREA_NM': ['프로방스마을', '병점역 1호선', '더현대서울', '강릉중앙시장', '청계천', '새절역 6호선', '장안순대국', '유일닭강정', '숙이네닭발', '삼송역 3호선'],
#     'DGSTFN': [4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 3.0]
# }
#
# # pandas DataFrame을 생성합니다.
# df_filter = pd.DataFrame(data)
#
# area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()
#
#
# # area_names
#
# # pandas DataFrame을 생성합니다.
# # df_filter = pd.DataFrame(data)
#
# # area_names = ["창덕궁",
# #     "광화문광장",
# #     "소피텔 앰배서더 서울",
# #     "남이섬",
# #     "수서역",
# #     "창경궁",
# #     "남산서울타워",
# #     "서울역",
# #     "덕수궁",
# #     "송도센트럴파크"]
#
# @app.route('/predict', methods=["POST"])
# def predict():
#     sample_data = {
#         'GENDER': '여',
#         'AGE_GRP': 30.0,
#         'TRAVEL_STYL_1': 3,  # 자연 vs 도시
#         'TRAVEL_STYL_2': 2,  # 숙박 vs 당일
#         'TRAVEL_STYL_3': 1,  # 새로운지역 vs ~
#         'TRAVEL_STYL_4': 2,
#         'TRAVEL_STYL_5': 6,
#         'TRAVEL_STYL_6': 4,
#         'TRAVEL_STYL_7': 2,
#         'TRAVEL_STYL_8': 7,
#         'TRAVEL_MOTIVE_1': 1,
#         'TRAVEL_COMPANIONS_NUM': 1.0,
#         'TRAVEL_MISSION_INT': 22,
#         'VISIT_AREA_NM': 'whatever',
#         'DGSTFN': 4.0
#     }
#     # data = request.json  # 클라이언트로부터 받은 데이터
#     # sample_data = data  # 여행자 정보
#
#     # sample_data
#
#     df_filter = pd.DataFrame(data)
#
#     area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()
#
#     results = pd.DataFrame([], columns=['AREA', 'SCORE'])
#
#     for area in area_names['VISIT_AREA_NM']:
#         input = list(sample_data.values())
#
#         input.append(area)
#
#         score = loaded_model.predict(input)
#
#         # # 입력 데이터를 모델에 맞게 조정
#         # input_df = pd.DataFrame([input_data], columns=list(traveler.keys()) + ['AREA'])
#
#         # # CatBoost 모델에 입력 전 데이터 프레임의 'AREA' 컬럼을 범주형으로 변환
#         # # (모델 학습 시 'AREA'가 범주형으로 처리되었다면 이 부분은 무시됩니다)
#         # input_df['AREA'] = input_df['AREA'].astype('category')
#
#         # score = loaded_model.predict(input_df)[0]  # 모델 예측
#
#         results = pd.concat([results, pd.DataFrame([[area, score]], columns=['AREA', 'SCORE'])])
#
#     # 점수가 높은 상위 5개 지역 반환
#     top_results = results.sort_values('SCORE', ascending=False).head(5)
#
#     return jsonify(top_results.to_dict(orient='records'))  # 결과를 JSON 형태로 반환
#
#
# if __name__ == "__main__":
#     app.run(debug=True)


#
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse, abort
import requests

app = Flask(__name__)
api = Api(app)

@app.route('/')
def hello_world():
    return 'Hello World!'

Todos = {
    'todo1': {"task": "exercise"},
    'todo2': {'task': "eat delivery food"},
    'todo3': {'task': 'watch movie'}
}


class TodoList(Resource):
    def get(self):
        return Todos

# flask<->node
@app.route('/test', methods=['POST'])
def test():
    file_name = request.args.get('file_name')
    return jsonify({'message': 'Received', 'file_name': file_name})

@app.route('/trigger-node', methods=['POST'])
def trigger_node():
    # Node.js 서버의 URL
    node_url = 'http://localhost:3000/'

    # Node.js 서버로 보낼 데이터
    data = {'file_name': 'example.jpg'}

    # Node.js 서버로 POST 요청을 보냅니다.
    response = requests.post(node_url, json=data)

    # Node.js 서버로부터의 응답을 반환합니다.
    return jsonify({'message': 'Data sent to Node.js', 'response': response.text})


api.add_resource(TodoList, '/todos/')
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
