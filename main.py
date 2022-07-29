from utils import CLOVA
from utils import contents_select
from utils import Frontalize
from utils import Lip_motion
from utils import Pronounce
from utils import visualize
from utils import Wave
from utils import recommendar

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
warnings.filterwarnings('ignore')

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

if __name__ == '__main__':
    file_path = input('what is your file_path')
    video_name = input('which video do you want to stduy?')
    target_name = input('Record your own video')
    video_index = int(input('which number of video can be study do you want to stduy?'))
    video_num = int(input('which number of clip do you want to stduy?'))
    target_dir = input('what is your own video path')
    source_audio = input('what is your drama audio path')
    target_audio = input('what is your own audio path')
    export_dir = input('what is audio directory path')
    api_sys = input('what is your drama audio path for api')
    api_user = input('what is your own audio path for api')
    pro_path = input('audio path for pronunciation')

    reg, p_score, t_score, mfcc_score, l_score = total_infer(video_name, file_path, target_name, target_dir, video_index, source_audio, target_audio, video_num,export_dir, api_sys, api_user, pro_path, sys_text=None, selected_dir=None)
    
    print('total score:', reg)