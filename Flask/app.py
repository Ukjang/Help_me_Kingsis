from flask import Flask, render_template, request, redirect
from flask import current_app, make_response, url_for, flash
import cv2
import flask
import numpy as np
import urllib3, json, base64
from utils import CLOVA
from utils import contents_select
from utils import Frontalize
from utils import Lip_motion
from utils import Pronounce
from utils import recommendar
from utils import Spleeter
from utils import visualize
from utils import Wave

def reg_score(p_score, t_score, mfcc_score, l_score):
    sc_list = [p_score, t_score, mfcc_score, l_score]
    sc_list = np.array(sc_list)
    sc_mean = np.mean(sc_list)
    sc_std = np.std(sc_list)
    total_score = []
    for sc in sc_list:
        if abs(float(sc)-sc_mean) >= sc_std :
            total_score.append(sc_mean)
        else :
            total_score.append(sc)
    reg = np.mean(np.array(total_score))
    return reg

def total_infer(video_name, file_path, target_name, target_dir, video_index, source_audio, target_audio, video_num,export_dir, api_sys, api_user, pro_path, sys_text=None, selected_dir=None):
    lst, dialogue = contents_select.contents_select(filepath=file_path, video_name=video_name, exist=True)
    lets_study, lets_study_lip_lst = contents_select.create_study_dir(video_name, lst, dialogue=dialogue, object=None, exist=True)
    selected_dir = f'./data/Study_Dir/{lets_study[video_index]}th_Study_Dir'
    
    Lip_motion.make_target_dir(target_name, selected_dir)

    dir_lst = lets_study
    l_score, l_lst = Lip_motion.lip_motion_analysis(video_index, target_dir, dir_lst)
    
    Wave.make_wave_file(f'{video_num}th_video', f'./data/Study_Dir/{video_num}th_Study_Dir/')
    Wave.make_wave_file(videoname=target_name, directory=file_path) 

    mfcc_score, x_sys, y_sys, x_user, y_user = Wave.MFCC(source_audio, target_audio, export_dir)

    t_score, sys_text, user_text = Pronounce.text_recognition(api_sys, api_user)
    p_score = Pronounce.prounce_score(pro_path, sys_text)

    visualize.mfcc_visualize(x_sys, y_sys, x_user, y_user)
    visualize.pronunciation_visualize(p_score)
    visualize.text_recognition_visualize(sys_text, user_text)
    visualize.lip_visualize(video_num, l_lst)
    visualize.radar_visualize(p_score, t_score, mfcc_score, l_score)

    reg = reg_score(p_score, t_score, mfcc_score, l_score)

    return reg, p_score, t_score, mfcc_score, l_score

app = Flask(__name__)

@app.route("/")
def sub_00():
    return render_template("sub-00.html")

@app.route("/start")
def sub_01():
    return render_template("sub-01.html")

@app.route("/start2")
def sub_02():
    return render_template("sub-02.html")

@app.route("/start3")
def sub_03():
    return render_template("sub-03.html")

@app.route("/start4")
def sub_04():
    return render_template("sub-04.html")

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'GET':
        return render_template('/video.html')
    else:
        file = request.files['video_blob']
        filename = 'static/images/raw_video.avi'
        file.save(filename)
        return '0'

@app.route('/video_proc', methods=['POST'])
def video_proc():
    flash('녹화된 동영상을 이 부분에서 처리해주면 됩니다.')
    raw_file = 'static/images/raw_video.avi'
    cap = cv2.VideoCapture(raw_file)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # 또는 cap.get(3)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)             # 또는 cap.get(5)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # 코덱 정의, *'mp4v' == 'm', 'p', '4', 'v', 또는 *'DIVX'
    out = cv2.VideoWriter('static/images/raw_video.mp4', fourcc, fps, (int(width), int(height))) # VideoWriter 객체
    while True:
        ret, img = cap.read()
        if ret:
            out.write(cv2.flip(img, 1))     # 좌우 반전 시켜서 원 위치로 환원
            cv2.waitKey(33)                 # 30 fps
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return render_template("spinner.html")

@app.route("/start5", methods = ['POST', 'GET'])
def sub_05():
    if request.method == 'GET':
        file_path = './data/'
        video_name = 'system_video'
        target_name = 'user_video'
        video_index = 0
        video_num = 4
        target_dir = './data/test'
        source_audio = f'./data/Audio/{video_num}th_video.wav'
        target_audio = './data/Audio/user_video.wav'
        export_dir = './data/Audio/'
        api_sys = './data/Audio/api_system_audio.wav'
        api_user = './data/Audio/api_user_audio.wav'
        pro_path = './data/Audio/pronoun.wav'
        reg, p_score, t_score, mfcc_score, l_score = total_infer(video_name, file_path, target_name, target_dir, video_index, source_audio, target_audio, video_num,export_dir, api_sys, api_user, pro_path, sys_text=None, selected_dir=None)
        return render_template("sub-05.html")
    else:
        return    
@app.route("/start6")
def sub_06():
    return render_template("sub-06.html")

@app.route("/start6_1")
def sub_06_1():
    return render_template("sub-06_1.html")

@app.route("/start7")
def sub_07():
    return render_template("sub-07.html")

@app.route("/start8")
def sub_08():
    return render_template("sub-08.html")

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host='0.0.0.0', debug=True)