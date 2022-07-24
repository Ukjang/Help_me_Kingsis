import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as io

from utils import Frontalize


def draw_lipline_on_image(img, faces):
    face_size = 0
    num = 0
    for n, face in enumerate(faces):
        if face.area() > face.size:
            face_size = face.area()
            num = n
    landmark_model = dlib.shape_predictor('..\model\Lip_motion\shape_predictor_68_face_landmarks.dat')
    lm = landmark_model(img, faces[num])
    lm_point = [[p.x, p.y] for p in lm.parts()]
    lm_point = np.array(lm_point)
    lip_point = [tuple(lm_point[index][i] for i in range(len(index)))]

    return lm_point, lip_point

def procrustes_analysis(lip_point):
    
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
    video_cap = cv2.VideoCapture('../data/' + video_name + '.mp4')
    cnt = 0
    try:
        os.mkdir('../data/test')
    except:
        pass
        print('Directory is already existed')
    try:
        while video_cap.isOpened():
            ret, img = video_cap.read()
            cv2.imwrite('../data/test/' + '%d.jpg' % cnt, img)
            cnt += 1
    except:
        video_cap.release()

    select_frame = len(glob(selected_dir + '/*.jpg'))
    if len(os.listdir('../data/test')) > len(select_frame):
        while len(os.listdir('../data/test')) != len(select_frame):
            os.remove(f"../data/test/{cnt}.jpg")
            cnt -= 1
    elif len(os.listdir('../data/test')) < len(select_frame):
        while len(os.listdir('../data/test')) != len(select_frame):
            gray_img = np.empty((240, 320), dtype=np.uint8)
            cv2.imwrite('../data/test/' + '%d.jpg' % cnt, img)
    else:
        pass


def lip_motion_analysis(video_idx, target_dir, dir_lst):
    # video index directory 이미지 추출
    dir_index = dir_lst[video_idx]
    dir_path = f'./data/Study_Dir/{dir_index}th_Dir/'
    imgs = glob(dir_path+'*.jpg')
    # 추출된 이미지를 정면화
    model3D = Frontalize.ThreeD_Model('../model/Lip_motion/model3Ddlib.mat', 'model_dlib')
    cnt = 0
    for img in imgs:
        cv.imread(dir_path + img, 1)
        img = Frontalize.center_image(img, IMAGE_SIZE=300)
        lmarks = Frontalize.get_landmarks(img)
        proj_matrix, camera_matrix, rmat, tvec = Frontalize.estimate_camera(model3D, lmarks[0])
        eyemask = np.asarray(io.loadmat('../model/Lip_motion/eyemask.mat')['eyemask'])
        frontal_raw, frontal_sym = Frontalize.frontalize(img, proj_matrix, ref_U, eyemask)

        try:
            os.mkdir(f'../data/Study_Dir/{dir_index}th_Dir/frontal')
        except:
            pass
            print('Directory is already existed')

        new_img = frontal_sym[:,:,::-1]
        cv.imwrite(f'../data/Study_Dir/{dir_index}th_Dir/frontal/' + '%d.jpg' %cnt, new_img)
        cnt += 1
    # 정면화된 이미지에서 입술 랜드마크 검출
    index = list(range(48, 68))
    taget_frames = sorted(glob(f'../data/Study_Dir/{dir_index}th_Dir/frontal/' + '*.jpg'), keys=os.path.getctime)
    lip_point_target = []
    for frame in target_frames:
        img = cv2.imread(frame)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(img_gray)

        if len(faces) == 0:
            lip_point=[]
        else:
            lm_point, lip_point = draw_lipline_on_image(img, faces)
            lip_point_scaling = procrustes_analysis(lip_point)
            lip_point_target.append(lip_point_scaling)

    # 사용자 프레임 이미지에서 입술 랜드마크 검출
    frame_lst = sorted(glob(target_dir + '/*.jpg'), key=os.path.getctime)
    iip_point_test = []

    for frame in frame_lst:
        img = cv2.imread(frame)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_detector = dlib.get_frontal_face_detector()
        faces = face_detector(img_gray)

        if len(faces) == 0:
            lip_point=[]
        else:
            lm_point, lip_point = draw_lipline_on_image(img, faces)
            lip_point_scaling = procrustes_analysis(lip_point)
            lip_point_test.append(lip_point_scaling)
    if len(lip_point_target) == len(lip_point_test):
        print('원본 영상과 학습자 영상의 프레임 수가 일치합니다.')
        lip_point_target = np.array(lip_point_target)
        lip_point_test = np.array(lip_point_test)

        sum_lst = []
        for o, t in zip(iip_point_target, iip_point_test):
            try:
                sum_lst.append(np.square(o - t))
                score = math.sqrt(np.sum(sum_lst) / len(sum_lst))
            except:
                pass
        print(f'입술 분석 결과: {score}')
    else:
        print('원본 영상과 학습자 영상의 프레임 수가 일치하지 않습니다.')
    
    return score

