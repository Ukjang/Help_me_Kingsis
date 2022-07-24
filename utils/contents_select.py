import json
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from utils import CLOVA
from utils import Lip_motion

def contents_select(filepath, video_name, exist=True):
    if exist==True:
        json_object = json.load(filepath + video_name + '.json')
        json_dialogue = json_object['dialogue_infos']

        st_time = {}
        for idx, dialog in enumerate(json_dialouge):
            st_time[idx] = dialog['start_time']
        sort_time = sorted(st_time.items(), key = lambda item: item[1])
        sort_time = [i[0] for i in sort_time]
        json_dialouge = [json_dialouge[i] for i in sort_time]

        script_lst = []
        script = ''

        for dialouge in json_dialouge:
            script = dialouge['utterance']
            script = re.sub(r'[^0-9a-zA-Zㄱ-ㅣ가-힣]', '', script)
            script_ls.append(script)
        
        script_len = len(script_lst)

    res = CLOVA.ClovaSpeechClient().req_upload(filepath+video_name + '.mp4', completion='sync')
    json_object = res.json()
    stt_lst = []
    for seg in json_object['segments']:
        stt = seg['text']
        stt = re.sub(r'[^0-9a-zA-Zㄱ-ㅣ가-힣]', '', stt)
        stt_lst.append(stt)

    i = -1
    validation_lst = []
    if exist==True:
        for sc, stt in zip(script_lst, stt_lst):
            i += 1
            if len(sc) >= 10:                                       # 글자수 10개 이상인 대본만 추출
                lst = [Counter(sc), Counter(stt)]
                df = pd.DataFrame(lst)
                acc = df.isna().sum(axis=1)[0] / len(Counter(sc))
                if acc < 5:                                         # 정확도 95% 이상인 대본만 추출
                    valid_video_lst.append(i)
    else:
        for stt in stt_lst:
            i += 1
            if len(stt) >= 10:
                valid_video_lst.append(i)
    if exist==True:
        return valid_lst, json_dialogue
    else:
        return valid_lst, json_object

# script = json_dialogue
def create_study_dir(video_name, lst, dialogue=None, object=None, exist=True):
    lets_study = []
    lets_study_lip_point_lst = []

    for i in lst:
        if exist==True:
            s = dialogue[i]['start_time']
            e = dialogue[i]['end_time']
            s = int(s[:2])*3600 + int(s[3:5])*60 + float(s[6:])
            e = int(e[:2])*3600 + int(e[3:5])*60 + float(e[6:])
        else:
            s = object['segments'][i]['start']
            e = object['segments'][i]['end']
            s = s/1000
            e = e/1000

        try:
            os.mkdir('./data/Study_Dir')
        except:
            pass
            print('Directory is already existed')

        try:
            os.mkdir('./data/Study_Dir/' + str(i) + 'th_Study_Dir')
        except:
            pass
            print('Directory is already existed')
        
        path = './data/Study_Dir/' + str(i) + 'th_Study_Dir/'
        ffmpeg_extract_subclip('./data/' + video_name + '.mp4', s, e, path + str(i) + 'th_video.mp4')

        video_cap = cv2.VideoCapture(path + str(i) + 'th_video.mp4')
        cnt = 0

        try:
            while video_cap.isOpened():
                ret, img = video_cap.read()
                cv2.imwrite(path + '%d.jpg' % cnt, img)
                cnt +=1
        except:
            video_cap.release()

        print(f'{i}번째 영상 확인을 시작합니다.')
        frame_lst = sorted(glob(path+'/*.jpg'), key=os.path.getctime)
        valid_frame_lst = []

        lip_index = list(range(48, 68))
        lip_point_lst = []
        lip_cnt = 0

        for fidx, frame in enumerate(frame_lst):
            img = cv2.imread(frame)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            face_detector = dlib.get_frontal_face_detector()
            faces = face_detector(img_gray)

            if len(faces) == 0:
                lip_point = []
                lip_point_lst.append(lip_point)
                lip_cnt += 1
                if lip_cnt > len(frame_lst) * 0.2:
                    print(f'{i}번 영상의 {frame}은 입술 좌표 추출이 불가합니다.')
                    break

            else:
                lm_point, lip_point = Lip_motion.draw_lipline_on_image(img, faces)
                valid_frame_lst.append(fidx)
                lip_point_lst.append(lip_point)
                if fidx + 1 == len(frame_lst):
                    lets_study.append(i)
                    lets_study_lip_point_lst.append(lip_point_lst)
                    print(f"{i}번 영상은 Let's study 학습 자료로 활용 가능합니다.")
                
        print('영상 확인 완료:', lets_study, '학습 가능')
        return lets_study, lets_study_lip_point_lst