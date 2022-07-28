import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from hanspell import spell_checker
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import math
import cv2
import dlib

plt.rcParams["figure.figsize"] = (20,10)

def mfcc_visualize(x_sys, y_sys, x_user, y_user):
    if len(y_sys) > len(y_user):
        new_y_sys = y_sys[:len(y_user)]
        y_dif = abs(new_y_sys - y_user)
    else:
        new_y_user = y_user[:len(y_sys)]
        y_dif = abs(new_y_user - y_sys)
    max_y_dif = list(y_dif).index(max(y_dif)) # 최소 유사 지점
    min_y_dif = list(y_dif).index(min(y_dif)) # 최대 유사 지점

    max_y_dif_y = list(y_user)[max_y_dif]
    min_y_dif_y = list(y_user)[min_y_dif]

    font_1 = {
    'family':'Arial',
    'color':'blue',
    'style':'italic',
    'size': 25}
    font_2 = {
    'family':'Arial',
    'color':'red',
    'style':'italic',
    'size': 25}

    temp_df = pd.concat([pd.DataFrame(y_user), pd.DataFrame(y_sys)],axis=1)
    temp_df.columns = ['User', 'Clip']

    temp_df.plot.area(color='BGYR', alpha=0.3, stacked=False)
    plt.scatter(min_y_dif,min_y_dif_y, color='blue')
    plt.text(min_y_dif, min_y_dif_y,'Good',fontdict=font_1)
    plt.scatter(max_y_dif,max_y_dif_y, color='red')
    plt.text(max_y_dif, max_y_dif_y, 'Bad',fontdict=font_2)
    plt.ylim(0.2, 0.8)
    
    plt.savefig('imgs/MFCC.png')

def pronunciation_visualize(p_score):
    c = 20*np.random.randn(3) + p_score
    labels_1 = ['Chunk Spped', 'Pause Length', 'Nativity']
    labels_2 = ['Accent', 'Innotation','Prosody']
    labels = [labels_1, labels_2]

    for idx, label in zip(range(1, 3),labels):
        c = (20*np.random.randn(6) + p_score).reshape(2, 3)
        plt.figure()
        sns.barplot(x=c[idx-1], y=label, palette=['hls', 'husl'][idx-1], alpha=0.7)
        plt.savefig(f'imgs/Pronunciation_{idx}.png')

def text_recognition_visualize(sys_text, user_text):
    sys_search = sys_text.replace(' ', '')
    user_search = user_text.replace(' ', '')

    sys_check = spell_checker.check(sys_search).as_dict()['checked']
    user_check = spell_checker.check(user_search).as_dict()['checked']

    sys_list, user_list = sys_check.split(' '), user_check.split(' ')
    uncommon_sys = [WORD for WORD in sys_list if WORD not in user_list]                      # sys_list에 있으나 user_list에는 없는 단어
    uncommon_user = {IDX:WORD for IDX, WORD in enumerate(user_list) if WORD not in sys_list} # user_list에 있으나 sys_list에는 없는 단어

    incorrect = list()
    for SYS, USER in zip(uncommon_sys, uncommon_user.values()):
        for IDX, STR in enumerate(USER):
            if STR not in SYS:
                ease = ' '.join(user_list[:user_list.index(USER)]+[USER[:IDX]])
                incorrect.append([ease, ease+STR])
    
    lines = 'ㅡ' * int(len(sys_text)*0.8)
    width = len(sys_text) * 12

    img = Image.new("RGB",(width,80), (255,255,255))
    draw = ImageDraw.Draw(img)
    
    fnt = ImageFont.truetype(r'C:/windows/fonts/YES24GothicR.ttf', 13)

    draw.text((20,20), sys_check, font=fnt, fill=(0,0,0))
    draw.text((20,35), lines, font=fnt, fill=(13, 80, 71))
    draw.text((20,50), user_check, font=fnt, fill=(0,0,0))

    for FRONT, STR in reversed(incorrect):
        draw.text((20, 50), STR, font=fnt, fill=(255,0,0)) 
        draw.text((20, 50), FRONT, font=fnt, fill=(0,0,0))

    img.save('imgs/Text_Recognition.png')

    sllya = f'Correct Syllables : {len(sys_check)-len(incorrect)}/{len(sys_check)}'

    img = Image.new("RGB", (180, 40), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.text((20, 15), sllya, font=fnt, fill=(0, 0, 0))

    img.save('imgs/Syllables.png')

##########################################################################

def draw_lipline_on_image(img_dir):
    
    index = list(range(48, 68))
    color = (0, 255, 255)
    thickness = 2
    
    # 이미지 불러오기
    img = cv2.imread(img_dir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출(흑백이미지로)
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(img_gray)

    face_size = 0
    num = 0

    for i, face in enumerate(faces):
        if face.area() > face_size:
            face_size = face.area()
            num = i

    # 랜드마크 검출(컬러이미지로)
    landmark_model = dlib.shape_predictor('./model/Lip_motion/shape_predictor_68_face_landmarks.dat')
    lm = landmark_model(img, faces[num])

    # 랜드마크 좌표 추출
    lm_point = [[p.x, p.y] for p in lm.parts()]
    lm_point = np.array(lm_point)

    # 입술 좌표 추출
    lip_point = [tuple(lm_point[index][i]) for i in range(len(index))]
    
    # 입술 라인 그리기
    for i in range(len(lip_point)):

        # OUTLINE 마지막 좌표
        if i == 11:
            cv2.line(img, lip_point[i], lip_point[0], color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # INLINE 마지막 좌표
        elif i == 19:
            cv2.line(img, lip_point[i], lip_point[12], color=color, thickness=thickness, lineType=cv2.LINE_AA)

        else:
            cv2.line(img, lip_point[i], lip_point[i+1], color=color, thickness=thickness, lineType=cv2.LINE_AA)
            
    # 입술 좌표 y값 양수로 변경
    lip_point = [(lip_point[i][0], -lip_point[i][1]) for i in range(20)]
            
    return img, lip_point

def draw_lipline_on_graph(lip_point):
    
    lipOutline_x = []
    lipOutline_y = []
    lipInline_x = []
    lipInline_y = []

    for i, p in enumerate(lip_point):

        # OUTLINE
        if i < 11:
            lipOutline_x.append(p[0])
            lipOutline_y.append(p[1])

        elif i == 11:
            lipOutline_x.append(p[0])
            lipOutline_x.append(lip_point[0][0])
            lipOutline_y.append(p[1])
            lipOutline_y.append(lip_point[0][1])

        # INLINE
        elif i > 11 and i < 19:
            lipInline_x.append(p[0])
            lipInline_y.append(p[1])

        elif i == 19:
            lipInline_x.append(p[0])
            lipInline_x.append(lip_point[12][0])
            lipInline_y.append(p[1])
            lipInline_y.append(lip_point[12][1])
    
    return lipOutline_x, lipOutline_y, lipInline_x, lipInline_y

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

def Lip_Pronunciation_Analysis(lip_point, lip_point_test):
    
    lip_point_procrustes = procrustes_analysis(lip_point)
    lip_point_procrustes_test = procrustes_analysis(lip_point_test)
    
    score = np.sum(np.square(lip_point_procrustes - lip_point_procrustes_test))
    
    return score

def draw_Lip_Pronunciation_Analysis_graph(lip_point, lip_point_test, condi):
    
    # Translation
    lip_point_translation = [(lip_point[i][0]-lip_point[9][0], lip_point[i][1]-lip_point[9][1]) for i in range(20)]
    lip_point_translation_test = [(lip_point_test[i][0]-lip_point_test[9][0], lip_point_test[i][1]-lip_point_test[9][1]) for i in range(20)]

    # Aligning
    rad = math.atan2(lip_point_translation[3][0], lip_point_translation[3][1])
    lip_point_aligning = []
    for i in range(20):
        lip_point_aligning.append((math.cos(rad)*lip_point_translation[i][0] - math.sin(rad)*lip_point_translation[i][1],
                                   math.sin(rad)*lip_point_translation[i][0] + math.cos(rad)*lip_point_translation[i][1]))
    rad = math.atan2(lip_point_translation_test[3][0], lip_point_translation_test[3][1])
    lip_point_aligning_test = []
    for i in range(20):
        lip_point_aligning_test.append((math.cos(rad)*lip_point_translation_test[i][0] - math.sin(rad)*lip_point_translation_test[i][1],
                                   math.sin(rad)*lip_point_translation_test[i][0] + math.cos(rad)*lip_point_translation_test[i][1]))
        
    # Scaling
    lip_point_scaling = lip_point_aligning / np.linalg.norm(lip_point_aligning)
    lip_point_scaling_test = lip_point_aligning_test / np.linalg.norm(lip_point_aligning_test)

    # Graph
    plt.figure(figsize=(16, 10))
    
    lipOutline_x, lipOutline_y, lipInline_x, lipInline_y = draw_lipline_on_graph(lip_point_scaling)
    if condi=='Good':
        plt.title('Best Frame of Lip Motion', size=35)
    else:
        plt.title('Worst Frame of Lip Motion', size=35)
    plt.plot(lipOutline_x, lipOutline_y, 'steelblue')
    plt.plot(lipInline_x, lipInline_y, 'steelblue')
    lipOutline_x_test, lipOutline_y_test, lipInline_x_test, lipInline_y_test = draw_lipline_on_graph(lip_point_scaling_test)
    plt.plot(lipOutline_x_test, lipOutline_y_test, 'orange')
    plt.plot(lipInline_x_test, lipInline_y_test, 'orange')
    plt.axis('off')

    
    if condi=='Good':
        plt.savefig('imgs/Best_Lip.png')
    else:
        plt.savefig('imgs/Worst_Lip.png')

def lip_visualize(video_number, lst):
    y_max = max(lst)
    x_max = lst.index(y_max)

    y_min = min(lst)
    x_min = lst.index(y_min)

    font_1 = {
        'family':'Arial',
        'color':'blue',
        'style':'italic',
        'size': 25}
    font_2 = {
        'family':'Arial',
        'color':'red',
        'style':'italic',
        'size': 25}

    pd.DataFrame(lst).plot.area(color='mediumseagreen', alpha=0.4,legend=False)
    plt.scatter(x_min,y_min, color='blue')
    plt.text(x_min, y_min,'Good',fontdict=font_1)
    plt.scatter(x_max,y_max, color='red')
    plt.text(x_max, y_max, 'Bad',fontdict=font_2)
    plt.ylim([0, y_max+0.01])

    plt.savefig('imgs/Lip_motion.png')

    min_frame, max_frame = x_min * 10, x_max * 10

    num_frame = [min_frame, max_frame]
    condition = ['Good', 'Bad']
    for frame_n, condi in zip(num_frame,condition):
        img, lip_point = draw_lipline_on_image(f'./data/Study_dir/{video_number}th_Study_Dir/{frame_n}.jpg')
        img_test, lip_point_test = draw_lipline_on_image(f'./data/test/{frame_n}.jpg')

        draw_Lip_Pronunciation_Analysis_graph(lip_point, lip_point_test, condi=condi)

def radar_visualize(p_score, t_score, mfcc_score, l_score):
    df = pd.DataFrame(dict(
        r=[p_score, t_score, l_score, mfcc_score],
        theta=['Pronunciation','Text Recognition','Lip_motion',
            'Acoustic']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(autosize=False)
    fig.update_polars(angularaxis_rotation=135)
    fig.show()

    fig.write_image("imgs/RADAR.png")