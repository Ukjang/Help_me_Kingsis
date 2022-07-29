import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as io
from glob import glob
import os
from utils import Frontalize


def extraction_lip_point(img):
    
    index = list(range(48, 68))

    img = cv2.imread(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img_gray)

    # 얼굴 검출 안 되는 경우
    if len(faces) == 0:
            lip_point = []

    # 얼굴 검출 되는 경우
    else:
        face_size = 0
        num = 0
        for n, face in enumerate(faces):
            if face.area() > face_size:
                face_size = face.area()
                num = n
        landmark_model = dlib.shape_predictor('./model/Lip_motion/shape_predictor_68_face_landmarks.dat')
        lm = landmark_model(img, faces[num])
        lm_point = [[p.x, p.y] for p in lm.parts()]
        lm_point = np.array(lm_point)
        lip_point = [tuple(lm_point[index][i]) for i in range(len(index))]

        # Translation
        lip_point_translation = [(lip_point[i][0]-lip_point[9][0], lip_point[i][1]-lip_point[9][1]) for i in range(20)]

        # Aligning
        rad = math.atan2(lip_point_translation[3][0], lip_point_translation[3][1])
        lip_point_aligning = []
        for i in range(20):
            lip_point_aligning.append((math.cos(rad)*lip_point_translation[i][0] - math.sin(rad)*lip_point_translation[i][1],
                                    math.sin(rad)*lip_point_translation[i][0] + math.cos(rad)*lip_point_translation[i][1]))
            
        # Scaling
        lip_point_scaling = lip_point_aligning / np.linalg.norm(lip_point_aligning)

    return lip_point_scaling

def make_target_dir(video_name, selected_dir):
    video_cap = cv2.VideoCapture('./static/' + video_name + '.mp4')
    cnt = 0
    try:
        os.mkdir('./static/test')
    except:
        pass
        print('Directory is already existed')
    try:
        while video_cap.isOpened():
            ret, img = video_cap.read()
            cv2.imwrite('./static/test/' + '%d.jpg' % cnt, img)
            cnt += 1
    except:
        video_cap.release()

    select_frames = len(glob(selected_dir + '/*.jpg'))
    if len(os.listdir('./static/test')) > select_frames:
        while len(os.listdir('./static/test')) != select_frames:
            cnt -= 1
            os.remove(f"./static/test/{cnt}.jpg")
    elif len(os.listdir('./static/test')) < select_frames:
        while len(os.listdir('./static/test')) != select_frames:
            gray_img = np.empty((240, 320), dtype=np.uint8)
            cv2.imwrite('./static/test/' + '%d.jpg' % cnt, gray_img)
            cnt += 1
    else:
        pass


def lip_motion_analysis(video_idx, target_dir, dir_lst):
    # video index directory 이미지 추출
    dir_index = dir_lst[video_idx]
    dir_path = f'./static/Study_Dir/{dir_index}th_Study_Dir/'
    imgs = sorted(glob(dir_path+'*.jpg'), key=os.path.getctime)
    # 추출된 이미지를 정면화
    model3D = Frontalize.ThreeD_Model('./model/Lip_motion/model3Ddlib.mat', 'model_dlib')
    cnt = 0
    try:
        os.mkdir(f'./static/Study_Dir/{dir_index}th_Study_Dir/frontal')
    except:
        pass
        print('Directory is already existed')

    for img in imgs:
        if len(imgs) == len(os.listdir(f'./static/Study_Dir/{dir_index}th_Study_Dir/frontal')):
            print('Frames are already existed')
            break
        try:
            img = cv2.imread(img, 1)
            img = Frontalize.center_image(img, IMAGE_SIZE=540)
            lmarks = Frontalize.get_landmarks(img)
            proj_matrix, camera_matrix, rmat, tvec = Frontalize.estimate_camera(model3D, lmarks[0])
            eyemask = np.asarray(io.loadmat('./model/Lip_motion/eyemask.mat')['eyemask'])
            frontal_raw, frontal_sym = Frontalize.frontalize(img, proj_matrix, model3D.ref_U, eyemask)

            new_img = frontal_sym[:,:,::-1]
        # error 발생 시, 기존의 img로 저장
        # 현재 색상과 사이즈를 바꾸는 것을 기존 색상과 사이즈로 저장하도록 변경
            cv2.imwrite(f'./static/Study_Dir/{dir_index}th_Study_Dir/frontal/' + '%d.jpg' %cnt, new_img)
            cnt += 1
        except:
            cv2.imwrite(f'./static/Study_Dir/{dir_index}th_Study_Dir/frontal/' + '%d.jpg' %cnt, img)

    # 정면화된 이미지에서 입술 랜드마크 검출
    target_frames = sorted(glob(f'./static/Study_Dir/{dir_index}th_Study_Dir/frontal/' + '*.jpg'), key=os.path.getctime)
    lip_point_target = []
    for i in range(0, len(target_frames), 10):
        lip_point = extraction_lip_point(target_frames[i])
        lip_point_target.append(lip_point)

    # 사용자 프레임 이미지에서 입술 랜드마크 검출
    frame_lst = sorted(glob(target_dir + '/*.jpg'), key=os.path.getctime)
    lip_point_test = []
    for i in range(0, len(frame_lst), 10):
        lip_point = extraction_lip_point(frame_lst[i])
        lip_point_test.append(lip_point)

    # 좌표 차이값 계산
    if len(lip_point_target) == len(lip_point_test):
        print('원본 영상과 학습자 영상의 프레임 수가 일치합니다.')
        lip_point_target = np.array(lip_point_target)
        lip_point_test = np.array(lip_point_test)
        sum_lst = []
        for target, test in zip(lip_point_target, lip_point_test):
            try:
                frame_score_lst = []
                frame_score = np.square(target - test)
                for point in frame_score:
                    point_score = np.sqrt(np.sum(point))
                    frame_score_lst.append(point_score)
                sum_lst.append(np.sum(frame_score_lst)/len(frame_score_lst))
            except:
                pass
        score = (np.sum(sum_lst))/(len(sum_lst))

        score = 100 - (score*3000)
        print(f'입술 분석 결과: {score}')
    else:
        print('원본 영상과 학습자 영상의 프레임 수가 일치하지 않습니다.')
    
    return score, frame_score_lst