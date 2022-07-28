from flask import Flask, render_template, request, redirect
from flask import current_app, make_response, url_for, flash
import cv2
import flask
import urllib3, json, base64
from utils import CLOVA
from utils import contents_select
from utils import Frontalize
from utils import koBERT
from utils import Lip_motion
from utils import Pronounce
from utils import recommendar
from utils import Spleeter
from utils import T5
from utils import Timesformer
from utils import visualize
from utils import Wave


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
        return render_template("sub-05.html")
    else:
        
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