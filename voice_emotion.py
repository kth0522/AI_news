from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
# init a model, let's use SVC
my_model = SVC()
# pass my model to EmotionRecognizer instance
# and balance the dataset
rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
# train the model
rec.train()

# loads the best estimators from `grid` folder that was searched by GridSearchCV in `grid_search.py`,
# and set the model to the best in terms of test score, and then train it

import os
import csv
import re

# 입력으로 주어진 폴더 내의 모든 WAV 파일 경로 가져오기
def get_wav_files(folder_path):
    wav_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

# 파일 이름에서 숫자를 추출하는 함수
def extract_number(file_name):
    # 파일 이름에서 확장자를 제외한 부분 추출
    name_without_extension = os.path.splitext(file_name)[0]

    # 파일 이름에서 숫자 부분 추출
    match = re.search(r"\d+", name_without_extension)
    if match:
        return int(match.group())
    else:
        return float("inf")  # 숫자가 없는 경우 가장 큰 값을 반환하여 맨 뒤에 정렬되도록 설정

# 폴더 내의 WAV 파일에 대한 예측 수행 및 결과 저장
def save_predictions_to_csv(folder_path, rec, output_folder):
    wav_files = get_wav_files(folder_path)

    # WAV 파일을 파일 이름 숫자를 기준으로 정렬
    wav_files.sort(key=extract_number)

    # CSV 파일에 저장할 데이터 리스트 초기화
    data = [("File Name", "Prediction")]

    for wav_file in wav_files:
        prediction = rec.predict(wav_file)
        file_name = os.path.basename(wav_file)
        data.append((file_name, prediction))

    # 결과를 저장할 폴더 생성 (폴더가 이미 존재하는 경우 무시)
    os.makedirs(output_folder, exist_ok=True)

    # CSV 파일 경로 설정
    output_csv_path = os.path.join(output_folder, "predictions.csv")

    # CSV 파일 저장
    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

# 특정 폴더 내의 WAV 파일에 대한 예측 결과를 CSV 파일로 저장
folder_path = "./data/cropped_segments"  # 실제 폴더 경로로 대체해야 합니다.
output_folder = "./outputs"  # 저장할 폴더 경로 지정
save_predictions_to_csv(folder_path, rec, output_folder)

