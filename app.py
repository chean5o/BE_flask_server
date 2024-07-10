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

uri = "mongodb+srv://lime:dhcoms65!@cluster-capstone-001.afkiqw3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster-Capstone-001"

client = MongoClient(uri, server_api=ServerApi('1'))
db = client.capstone  # capstone이라는 이름의 디비 접속

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# catboost 모델 로드
loaded_model = CatBoostRegressor()
loaded_model.load_model('catboost_recommend_model.cbm')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

logging.basicConfig(level=logging.INFO)

# area_names_list1 = list(db.question_1.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}).limit(10))
# area_names_list2 = list(db.question_5.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}).limit(10))
# area_names_list3 = list(db.question_6.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}).limit(10))
# area_names_list4 = list(db.question_8.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}).limit(10))

global_top_results_with_coords = pd.DataFrame()
questionId_7_answer = None  # questionId가 7인 항목의 answer를 저장할 변수 초기화
questionId_1_answer = None
questionId_2_answer = None
questionId_4_answer = None
questionId_3_answer = None


@app.route('/')
def index():
    area_names_list = list(db.question_1.find(
        {
            "price": {"$lte": 100000}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
            "VISIT_AREA_TYPE_CD": {"$in": [1, 2, 7]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
        },
        {
            'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
            '_id': 0  # 반환할 필드 지정
        }
    ).limit(10))
    area_names_df = pd.DataFrame(area_names_list)
    # print(area_names_df)
    print(area_names_list)
    area_names_list = list(db.question_1.find(
        {
            "price": {"$lte": 100000},  # 조건: 가격이 price_limit1 이하인 경우
            "VISIT_AREA_TYPE_CD": {"$in": [3, 4, 6]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
        },
        {
            'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'VISIT_AREA_TYPE_CD': 1, '_id': 0  # 반환할 필드 지정
        }
    ).limit(10))
    print(area_names_list)
    return 'Server is running'


@app.route('/data', methods=['GET'])
def get_data():
    global global_top_results_with_coords
    if global_top_results_with_coords is not None:
        if check_area_count(global_top_results_with_coords):
            # DataFrame을 JSON 형식으로 변환
            top_results_json = global_top_results_with_coords.to_json(orient='records', force_ascii=False)
            print(top_results_json)
            global_top_results_with_coords = None
            return Response(top_results_json, content_type="application/json; charset=utf-8")
        else:
            # check_area_count 조건을 만족하지 않는 경우
            return jsonify({'error': '조건을 만족하는 데이터가 없습니다.'}), 400
    else:
        # global_top_results_with_coords가 None인 경우
        return jsonify({'error': '데이터가 없습니다.'}), 400


@app.route('/vote', methods=["POST"])
def predict():
    data = request.get_json()
    logging.info("Received vote data: %s", data)

    if not isinstance(data, list):
        logging.error("Data format error: Data should be a list")
        return jsonify({'error': 'Data format error: Data should be a list'}), 400

    answer_mapping = {item['questionId']: (2 if item['answer'] == 'Y' else 8)
                      for item in data if 1 <= item['questionId'] <= 4}
    answer_mapping_else = {item['questionId']:
                               (float(item['answer']) if item['answer'] == '6' else
                                (item['answer'] if item['answer'] == '5' else
                                 (int(item['answer']) if item['answer'] == '7' else item['answer'])))
                           for item in data}

    # questionId_7_answer = None   # questionId가 7인 항목의 answer를 저장할 변수 초기화
    global questionId_7_answer

    for item in data:
        if item.get('questionId') == 7:  # questionId가 7인 항목을 찾음
            answer_text = item.get('answer', '')  # answer 값을 가져옴, 없으면 빈 문자열 반환
            # "원", 쉼표, 공백 제거
            answer_text = answer_text.replace("원", "").replace(",", "").strip()
            try:
                questionId_7_answer = int(answer_text)  # 정수로 변환
                break  # 값을 찾았으니 루프 종료
            except ValueError:
                # answer 값을 정수로 변환할 수 없는 경우(예: answer가 숫자가 아닌 경우)
                logging.error("Conversion error: answer of questionId 7 cannot be converted to int")
                questionId_7_answer = None

    print(questionId_7_answer)  # k에 원하는 값 할당
    price_limit1 = questionId_7_answer * 1 / 36  # k의 1/16 계산
    price_limit2 = questionId_7_answer * 5 / 48  # k의 1/16 계산
    price_limit4 = questionId_7_answer * 2 / 36  # k의 1/16 계산
    price_limit3 = questionId_7_answer * 8 / 48  # k의 1/16 계산
    # price_limit4 = questionId_7_answer * 2/36  # k의 1/16 계산
    print(price_limit1)

    questionId_1_answer = answer_mapping.get(1, 5)
    questionId_2_answer = answer_mapping.get(2, 5)
    # questionId_4_answer = answer_mapping.get(4, 5)
    questionId_4_answer = answer_mapping.get(4, 5)
    print(questionId_1_answer)
    if questionId_1_answer == 2:
        area_names_list1 = list(db.question_1.find(
            {
                "price": {"$lte": price_limit1}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
                "VISIT_AREA_TYPE_CD": {"$in": [1, 2, 7]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
            },
            {
                'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
                '_id': 0  # 반환할 필드 지정
            }
        ).limit(10))
    else:
        area_names_list1 = list(db.question_1.find(
            {
                "price": {"$lte": price_limit1}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
                "VISIT_AREA_TYPE_CD": {"$in": [3, 4, 6]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
            },
            {
                'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
                '_id': 0  # 반환할 필드 지정
            }
        ).limit(10))
    print(questionId_2_answer)
    if questionId_2_answer == 2:
        area_names_list2 = list(db.question_1.find(
            {
                "price": {"$lte": price_limit2}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
                "VISIT_AREA_TYPE_CD": {"$in": [1, 7]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
            },
            {
                'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
                '_id': 0  # 반환할 필드 지정
            }
        ).limit(10))
    else:
        area_names_list2 = list(db.question_1.find(
            {
                "price": {"$lte": price_limit2}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
                "VISIT_AREA_TYPE_CD": {"$in": [5, 6]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
            },
            {
                'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
                '_id': 0  # 반환할 필드 지정
            }
        ).limit(10))
    print(questionId_4_answer)
    if questionId_4_answer == 2:
        area_names_list4 = list(db.question_1.find(
            {
                "price": {"$lte": price_limit4}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
                "VISIT_AREA_TYPE_CD": {"$in": [1, 2]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
            },
            {
                'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
                '_id': 0  # 반환할 필드 지정
            }
        ).limit(10))
    else:
        area_names_list4 = list(db.question_1.find(
            {
                "price": {"$lte": price_limit4}, "tot_review": {"$exists": True},  # 조건: 가격이 price_limit1 이하인 경우
                "VISIT_AREA_TYPE_CD": {"$in": [8, 6]}  # 조건: VISIT_AREA_TYPE_CD가 '02' 또는 '03'인 경우
            },
            {
                'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price': 1, 'tot_review': 1, 'VISIT_AREA_TYPE_CD': 1,
                '_id': 0  # 반환할 필드 지정
            }
        ).limit(10))

    # area_names_list1 = list(db.question_1.find(
    #     {"price": {"$lte": price_limit1}},  # price가 k의 1/16 이하인 조건 추가
    #     {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price':1, '_id': 0}  # 반환할 필드 지정
    # ))
    # area_names_list2 = list(db.question_5.find(
    #     {"price": {"$lte": price_limit2}},  # price가 k의 1/16 이하인 조건 추가
    #     {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price':1, '_id': 0}  # 반환할 필드 지정
    # ))
    # area_names_list4 = list(db.question_8.find(
    #     {"price": {"$lte": price_limit4}},  # price가 k의 1/16 이하인 조건 추가
    #     {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price':1, '_id': 0}  # 반환할 필드 지정
    # ))
    area_names_list3 = list(db.question_1.find(
        {"price": {"$lte": price_limit3}, "tot_review": {"$exists": True}},  # price가 k의 1/16 이하인 조건 추가
        {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'tot_review': 1, 'price': 1, '_id': 0}  # 반환할 필드 지정
    ).limit(10))
    # # area_names_list4 = list(db.question_8.find(
    #     {"price": {"$lte": price_limit4}},  # price가 k의 1/16 이하인 조건 추가
    #     {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, 'price':1, '_id': 0}  # 반환할 필드 지정
    # ))
    # area_names_list2 = list(db.question_5.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}))
    # area_names_list3 = list(db.question_6.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}))
    # area_names_list4 = list(db.question_8.find({}, {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}))

    traveler1 = {
        'GENDER': answer_mapping_else.get(5, "남"),
        'AGE_GRP': answer_mapping_else.get(6, 20.0),
        'TRAVEL_STYL_1': answer_mapping.get(1, 5),
        # 'TRAVEL_STYL_2': 5,
        # 'TRAVEL_STYL_3': 5,
        # 'TRAVEL_STYL_4': 5,
        'TRAVEL_STYL_5': 5,
        'TRAVEL_STYL_6': 5,
        # 'TRAVEL_STYL_7': 5,
        'TRAVEL_STYL_8': 5,
        'TRAVEL_MOTIVE_1': 8,
        # 'TRAVEL_COMPANIONS_NUM': 0.0,
        'TRAVEL_MISSION_INT': 3,
    }

    traveler2 = {
        'GENDER': answer_mapping_else.get(5, "남"),
        'AGE_GRP': answer_mapping_else.get(6, 20.0),
        'TRAVEL_STYL_1': 5,
        # 'TRAVEL_STYL_2': 5,
        # 'TRAVEL_STYL_3': 5,
        # 'TRAVEL_STYL_4': 5,
        'TRAVEL_STYL_5': answer_mapping.get(2, 5),
        'TRAVEL_STYL_6': 5,
        # 'TRAVEL_STYL_7': 5,
        'TRAVEL_STYL_8': 5,
        'TRAVEL_MOTIVE_1': 8,
        # 'TRAVEL_COMPANIONS_NUM': 0.0,
        'TRAVEL_MISSION_INT': 3,
    }

    traveler4 = {
        'GENDER': answer_mapping_else.get(5, "남"),
        'AGE_GRP': answer_mapping_else.get(6, 20.0),
        'TRAVEL_STYL_1': 5,
        # 'TRAVEL_STYL_2': 5,
        # 'TRAVEL_STYL_3': 5,
        # 'TRAVEL_STYL_4': 5,
        'TRAVEL_STYL_5': 5,
        'TRAVEL_STYL_6': 5,
        # 'TRAVEL_STYL_7': 5,
        'TRAVEL_STYL_8': answer_mapping.get(4, 5),
        'TRAVEL_MOTIVE_1': 8,
        # 'TRAVEL_COMPANIONS_NUM': 0.0,
        'TRAVEL_MISSION_INT': 3,
    }

    traveler3 = {
        'GENDER': answer_mapping_else.get(5, "남"),
        'AGE_GRP': answer_mapping_else.get(6, 20.0),
        'TRAVEL_STYL_1': 5,
        # 'TRAVEL_STYL_2': 5,
        # 'TRAVEL_STYL_3': 5,
        # 'TRAVEL_STYL_4': 5,
        'TRAVEL_STYL_5': 5,
        'TRAVEL_STYL_6': answer_mapping.get(4, 5),
        # 'TRAVEL_STYL_7': 5,
        'TRAVEL_STYL_8': answer_mapping.get(3, 5),
        'TRAVEL_MOTIVE_1': 8,
        # 'TRAVEL_COMPANIONS_NUM': 0.0,
        'TRAVEL_MISSION_INT': 3,
    }

    # question_function_map = {
    # 1: lambda: get_top_areas(area_names_list1, traveler1, 2),
    # 2: lambda: get_top_areas(area_names_list2, traveler2, 2),
    # # 추가적인 questionId와 함수 매핑은 여기에 추가
    # }

    # 데이터에서 모든 questionId를 추출
    received_question_ids = [item['questionId'] for item in data]

    specific_question_id = 1
    if specific_question_id not in received_question_ids:
        print(
            f"The questionId {specific_question_id} has not yet been received. Skipping get_top_areas function calls.")
    else:
        # questionId 1이 존재하면, get_top_areas 함수 실행
        # area_names_list1 = list(db.question_1.find(
        # {"price": {"$lte": price_limit1}},  # price가 k의 1/16 이하인 조건 추가
        # {'VISIT_AREA_NM': 1, 'X_COORD': 1, 'Y_COORD': 1, '_id': 0}  # 반환할 필드 지정
        # ))
        get_top_areas(area_names_list1, traveler1, 2)
        print(global_top_results_with_coords)
        received_question_ids.remove(specific_question_id)

    specific_question_id = 2
    if specific_question_id not in received_question_ids:
        print(
            f"The questionId {specific_question_id} has not yet been received. Skipping get_top_areas function calls.")
    else:
        # questionId 2이 존재하면, get_top_areas 함수 실행
        get_top_areas(area_names_list2, traveler2, 2)
        print(global_top_results_with_coords)
        received_question_ids.remove(specific_question_id)

    specific_question_id = 4
    if specific_question_id not in received_question_ids:
        print(
            f"The questionId {specific_question_id} has not yet been received. Skipping get_top_areas function calls.")
    else:
        # questionId 2이 존재하면, get_top_areas 함수 실행
        get_top_areas(area_names_list4, traveler4, 2)
        print(global_top_results_with_coords)
        # received_question_ids.remove(specific_question_id)

    specific_question_id = 3
    if specific_question_id not in received_question_ids:
        print(
            f"The questionId {specific_question_id} has not yet been received. Skipping get_top_areas function calls.")
    else:
        # questionId 2이 존재하면, get_top_areas 함수 실행
        get_top_areas(area_names_list3, traveler3, 3)
        print(global_top_results_with_coords)
        received_question_ids.remove(specific_question_id)
        received_question_ids.remove(4)

    # get_top_areas(area_names_list2, traveler2, 2)
    # print(global_top_results_with_coords)
    # get_top_areas(area_names_list3, traveler3, 3)
    # print(global_top_results_with_coords)
    # get_top_areas(area_names_list4, traveler4, 2)
    # print(global_top_results_with_coords)

    # if check_area_count(global_top_results_with_coords):
    #     if global_top_results_with_coords is not None:
    #     # DataFrame을 JSON 형식으로 변환
    #     top_results_json = global_top_results_with_coords.to_json(orient='records', force_ascii=False)

    # return Response(response=top_results_json,
    #             status=200,
    #             mimetype="application/json; charset=utf-8")

    # if global_top_results_with_coords is not None:
    #     if check_area_count(global_top_results_with_coords):
    #         # DataFrame을 JSON 형식으로 변환
    #         top_results_json = global_top_results_with_coords.to_json(orient='records', force_ascii=False)
    #         print(top_results_json)
    #         return Response(top_results_json, content_type="application/json; charset=utf-8")
    #     else:
    #         # 조건을 만족하지 않는 경우
    #         return jsonify({'error': '조건을 만족하는 데이터가 없습니다.'}), 400
    # else:
    #     # 데이터프레임이 None인 경우
    #     return jsonify({'error': '데이터가 없습니다.'}), 400


# global_top_results_with_coords = None

# def get_top_areas(area_names_list, traveler, top_n):

#     global global_top_results_with_coords
#     results_list = []

#     # 각 지역에 대해 점수 계산
#     for document in area_names_list:
#         area = document['VISIT_AREA_NM']
#         input_data = list(traveler.values())
#         input_data.append(area)
#         score = loaded_model.predict([input_data])[0]  # 모델을 사용하여 점수 예측
#         results_list.append([area, score, document['X_COORD'], document['Y_COORD']])

#     # 결과 DataFrame 생성 및 점수 기준으로 상위 N개 선택
#     results = pd.DataFrame(results_list, columns=['AREA', 'SCORE','X_COORD', 'Y_COORD'])
#     top_results = results.sort_values('SCORE', ascending=False)[:top_n]

#     # 좌표 정보를 포함한 최종 결과 생성
#     area_names_df = pd.DataFrame(area_names_list)
#     global_top_results_with_coords = top_results.merge(area_names_df,
#                                                 left_on='AREA',
#                                                 right_on='VISIT_AREA_NM',
#                                                 how='left').drop(columns=['VISIT_AREA_NM'])

# 초기에 global_top_results_with_coords를 빈 DataFrame으로 선언
# global_top_results_with_coords = pd.DataFrame()

def get_top_areas(area_names_list, traveler, top_n):
    global global_top_results_with_coords
    results_list = []

    # 각 지역에 대해 점수 계산
    for document in area_names_list:
        area = document['VISIT_AREA_NM']
        input_data = list(traveler.values())
        input_data.append(area)
        score = loaded_model.predict([input_data])[0]  # 모델을 사용하여 점수 예측
        results_list.append(
            [area, score, document['X_COORD'], document['Y_COORD'], document['price'], document['tot_review']])

    # 결과 DataFrame 생성 및 점수 기준으로 상위 N개 선택
    results = pd.DataFrame(results_list, columns=['AREA', 'SCORE', 'X_COORD', 'Y_COORD', 'price', 'tot_review'])
    top_results = results.sort_values('SCORE', ascending=False)[:top_n]

    # 좌표 정보를 포함한 최종 결과 생성
    area_names_df = pd.DataFrame(area_names_list)
    new_results_with_coords = top_results.merge(area_names_df,
                                                left_on='AREA',
                                                right_on='VISIT_AREA_NM',
                                                how='left').drop(columns=['VISIT_AREA_NM', 'X_COORD_y', 'Y_COORD_y'])

    # 기존 결과에 새로운 결과를 누적
    global_top_results_with_coords = pd.concat([global_top_results_with_coords, new_results_with_coords],
                                               ignore_index=True)


def check_area_count(df):
    """DataFrame의 'AREA' 컬럼의 행 수가 9개 이상인지 확인합니다."""
    return df['AREA'].count() >= 9


# def send_to_kotl_app():
#     global global_top_results_with_coords  # 전역 변수 사용을 명시
#     kotl_js_endpoint = 'http://localhost:3000/data'

#     if global_top_results_with_coords is not None:
#         # DataFrame을 JSON 형식으로 변환
#         top_results_json = global_top_results_with_coords.to_json(orient='records', force_ascii=False)

#         # Node.js 서버로 전송
#         headers = {'Content-Type': 'application/json'}  # JSON 형식임을 명시
#         response = requests.post(kotl_js_endpoint, data=top_results_json.encode('utf-8'), headers=headers)

#         # 서버 응답 확인
#         if response.status_code == 200:
#             print("데이터가 성공적으로 전송되었습니다.")
#             global_top_results_with_coords = None  # 데이터 전송 후 전역 변수 초기화
#         else:
#             print(f"전송 실패: {response.status_code}")
#     else:
#         print("전송할 데이터가 없습니다.")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)