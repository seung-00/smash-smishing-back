import pickle
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# SVM 모델 호출
svm_model = joblib.load('model/svm_model.pkl')


def predict(message):
    '''
    parameter: message(문자메시지)
    return: (proba[확률], prediction[분류결과])
    '''
    # ================== 예측 모델 ======================
    # loaded_vectorizer: vectorizer 모델 호출

    loaded_vectorizer = pickle.load(open('model/test_vec.pickle', 'rb'))

    # INPUT DATA를 loaded_vectorizer로 변환시켜줌
    message_data = loaded_vectorizer.transform([message])

    # 예측 label
    prediction = svm_model.predict(message_data)
    # 예측 label에 대한 확률
    # proba = svm_model.predict_proba(message_data)

    if (prediction == 1):
        return 1
    else:
        return 0


@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.json['message']  # json 데이터를 받아옴
        print(message)
        predict_result = predict(message)
        print(predict_result)
        response_data = {"prediction": predict_result}
        return jsonify(response_data)

    else:
        return jsonify({'test': 1})


@app.route('/')
def hello():
    return 'Hello'


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
