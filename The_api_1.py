from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
from catboost import CatBoostRegressor
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import json
from flask import Flask, Response

#몽고디비 아틀라스 연동
uri = "mongodb+srv://lime:dhcoms65!@cluster-capstone-001.afkiqw3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-Capstone-001"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client.capstone #capstone이라는 이름의 디비 접속
# place_info = list(db.place.find({'VISIT_AREA_NM':'돌카롱 중문점'}))

# Create a new client and connect to the server
# client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# catboost 모델 로드
loaded_model = CatBoostRegressor()
loaded_model.load_model('catboost1_model.cbm')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

logging.basicConfig(level=logging.INFO)

# Data = {
#     # 'GENDER': ['남', '여', '여', '남', '남', '남', '여', '여', '남', '남'],
#     # 'AGE_GRP': [30.0, 40.0, 30.0, 30.0, 20.0, 40.0, 30.0, 40.0, 20.0, 30.0],
#     # 'TRAVEL_STYL_1': [4, 4, 4, 7, 6, 6, 1, 2, 7, 3],
#     # 'TRAVEL_STYL_2': [4, 3, 4, 1, 1, 6, 3, 2, 4, 3],
#     # 'TRAVEL_STYL_3': [2, 6, 4, 1, 1, 2, 1, 1, 3, 5],
#     # 'TRAVEL_STYL_4': [4, 3, 3, 1, 3, 4, 4, 2, 4, 5],
#     # 'TRAVEL_STYL_5': [6, 2, 4, 7, 4, 4, 4, 2, 3, 1],
#     # 'TRAVEL_STYL_6': [5, 2, 4, 7, 6, 2, 2, 2, 7, 5],
#     # 'TRAVEL_STYL_7': [6, 7, 5, 2, 3, 4, 1, 1, 6, 5],
#     # 'TRAVEL_STYL_8': [6, 2, 7, 4, 2, 2, 7, 7, 3, 2],
#     # 'TRAVEL_MOTIVE_1': [1, 6, 1, 1, 3, 1, 3, 3, 2, 6],
#     # 'TRAVEL_COMPANIONS_NUM': [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0],
#     # 'TRAVEL_MISSION_INT': [5, 28, 6, 2, 11, 1, 3, 9, 6, 21],
#     'VISIT_AREA_NM': ['돌하르방식당', '오는정김밥', '물고기자리', '올래국수', '제주해녀의집'],
#     # 'DGSTFN': [4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 3.0]
# }

# 데이터를 pandas DataFrame으로 변환
# Data = pd.DataFrame(Data)

# categorical_features_names = [
#     # 'GENDER',
#     # 'AGE_GRP',
#     # 'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
#     # 'TRAVEL_MOTIVE_1',
#     # 'TRAVEL_COMPANIONS_NUM',
#     # 'TRAVEL_MISSION_INT',
#     'VISIT_AREA_NM',
#     # 'DGSTFN',
# ]

# df_filter = df[~df['TRAVEL_MISSION_CHECK'].isnull()].copy()

# df_filter.loc[:, 'TRAVEL_MISSION_INT'] = df_filter['TRAVEL_MISSION_CHECK'].str.split(';').str[0].astype(int)

# df_filter

# df_filter = Data[[
#     # 'GENDER',
#     # 'AGE_GRP',
#     # 'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
#     # 'TRAVEL_MOTIVE_1',
#     # 'TRAVEL_COMPANIONS_NUM',
#     # 'TRAVEL_MISSION_INT',
#     'VISIT_AREA_NM',
#     # 'DGSTFN',
# ]]

# df_filter = df_filter.dropna()

# df_filter[categorical_features_names[1:-1]] = df_filter[categorical_features_names[1:-1]].astype(int)

# area_names = df_filter[['VISIT_AREA_NM']].drop_duplicates()

# area_names

# df_filter


@app.route('/')
def index():
    area_names_list = list(db.question_1.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}).limit(10))
    area_names_df = pd.DataFrame(area_names_list)
    print(area_names_df)
    print(area_names_list)
    return 'Server is running'

# Database = df_filter[['VISIT_AREA_NM']].drop_duplicates()

@app.route('/vote', methods=["POST"])
def predict():
    # JSON 데이터를 받음
    data = request.get_json()
    logging.info("Received vote data: %s", data)

    if not isinstance(data, list):
        logging.error("Data format error: Data should be a list")
        return jsonify({'error': 'Data format error: Data should be a list'}), 400

# 질문 ID에 따라 답변을 매핑하는 딕셔너리 생성
    answer_mapping = {item['questionId']: (2 if item['answer'] == 'Y' else 8) 
                  for item in data if 1 <= item['questionId'] <= 4}
    answer_mapping_else = {item['questionId']: 
                  (float(item['answer']) if item['answer'] == '6' else 
                   (item['answer'] if item['answer'] == '5' else 
                    (int(item['answer']) if item['answer'] == '7' else item['answer'])))
                  for item in data}


    traveler = {
    'GENDER': answer_mapping_else.get(5, "남"),
    'AGE_GRP': answer_mapping_else.get(6, 20.0),
    'TRAVEL_STYL_1': answer_mapping.get(1, 2),
    'TRAVEL_STYL_2': answer_mapping.get(2, 2),
    'TRAVEL_STYL_3': answer_mapping.get(3, 2),
    'TRAVEL_STYL_4': answer_mapping.get(4, 2),
    'TRAVEL_STYL_5': 2,
    'TRAVEL_STYL_6': 2,
    'TRAVEL_STYL_7': 2,
    'TRAVEL_STYL_8': 3,
    'TRAVEL_MOTIVE_1': 8,
    'TRAVEL_COMPANIONS_NUM': 0.0,
    'TRAVEL_MISSION_INT': 3,
    }
    # area_names = db.place.find({}, {'VISIT_AREA_NM': 1}) 
    area_names_list = list(db.question_1.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}).limit(10)) # _id 필드는 기본적으로 포함되므로, 제외하고자 할 때 명시적으로 지정해야 합니다.
    results_list = []
    area_names_df = pd.DataFrame(area_names_list)
    print(area_names_list)
    for document in area_names_list:
        area = document['VISIT_AREA_NM']
        input_data = list(traveler.values())
        input_data.append(area)
        score = loaded_model.predict([input_data])[0]  # 가정: 예측 결과의 첫 번째 값이 실제 점수
        
        results_list.append([area, score])
    results = pd.DataFrame(results_list, columns=['AREA', 'SCORE'])

    top_results = results.sort_values('SCORE', ascending=False)[:5]
    top_results_with_coords = top_results.merge(area_names_df, 
                                            left_on='AREA', 
                                            right_on='VISIT_AREA_NM', 
                                            how='left').drop(columns=['VISIT_AREA_NM'])
    top_results_json = top_results_with_coords.to_json(orient='records', force_ascii=False)
    
    # top_area_names = [result for result in top_results['AREA']]
    # print(top_results_with_coords)
    # response = json.dumps({'AREA': top_area_names}, ensure_ascii=False)
    # temporary_data = top_area_names.to_dict('records')
    node_js_endpoint = 'http://localhost:3000/receive_data1'
    
    response = requests.post(node_js_endpoint, json=json.loads(top_results_json))

    # Node.js 서버로 결과 데이터를 전송
    # node_js_endpoint = 'http://localhost:3000/receive_data'
    # response = requests.post(node_js_endpoint, json=temporary_data)
    logging.info("Sending data to Node.js server: %s", results)  # 로그에 전송 데이터 기록
    
    if response.status_code == 200:
        logging.info("Data successfully sent to the Node.js server.")
    else:
        logging.error("Failed to send data to the Node.js server.")

    # 결과를 JSON 형태로 반환
    return Response(response, content_type="application/json; charset=utf-8")

if __name__ == "__main__":
    app.run()