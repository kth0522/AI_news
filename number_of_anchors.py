import os
import glob
from deepface import DeepFace
from deepface.basemodels import VGGFace
import pandas as pd
import matplotlib.pyplot as plt

backends = [
  'opencv',
  'ssd',
  'dlib',
  'mtcnn',
  'retinaface',
  'mediapipe',
  'yolov8',
  'yunet',
]

model = VGGFace.loadModel()

# 주어진 폴더 경로
folder_path = "./separate_frames/human_pm7.2_0001"

# 결과를 저장할 DataFrame
results_df = pd.DataFrame(columns=["image_name", "number_of_faces", "match_found"])

filepaths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")),
                   key=lambda name: int(name.split("_")[-1].split(".")[0]))

for filepath in filepaths:
    filename = os.path.basename(filepath)
    image_name = os.path.splitext(filename)[0]  # 확장자 제거

    # "frame_n.jpg" 형식의 파일명일 경우, "n"만 추출
    if image_name.startswith("frame_"):
        image_name = int(image_name.split("_")[1])  # convert to integer for correct sorting

    match_found = False  # Initialize match_found as False
    number_of_faces = 0  # Initialize number_of_faces as 0

    try:
        face_objs = DeepFace.extract_faces(img_path=filepath,
            target_size=(224, 224),
            detector_backend=backends[4]
        )

        number_of_faces = len(face_objs)  # Update number of faces

        for face_obj in face_objs:
            face = face_obj["face"]
            plt.imshow(face)
            plt.axis("off")
            plt.savefig('temp_face.jpg')
            plt.close()  # 이미지가 완전히 저장될 때까지 기다립니다.
            dfs = DeepFace.find(img_path="temp_face.jpg", db_path="./face_database",
                                model_name='VGG-Face', distance_metric='cosine')
            # match_found: 결과가 비어있지 않으면 True, 그렇지 않으면 False
            match_found = not dfs[0].empty if isinstance(dfs, list) and dfs else False

    except ValueError:
        print(f"Face could not be detected in {filename}. Skipping this image.")

    results_df = results_df.append({"image_name": image_name, "number_of_faces": number_of_faces, "match_found": match_found}, ignore_index=True)

# 결과를 CSV 파일로 저장
results_df.to_csv('match_results.csv', index=False)
