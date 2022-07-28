from utils import CLOVA
from konlpy.tag import Okt
import librosa
import soundfile as sf
import urllib3
import json
import base64

def down_sample(input_wav, origin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    print("original wav sr: {}, original wav shape: {}, resample wav sr: {}, resmaple shape: {}".format(origin_sr, y.shape, resample_sr, resample.shape))
    sf.write('./data/Audio/' + 'pronoun.wav', resample, resample_sr, format='WAV', endian='LITTLE', subtype='PCM_16')


def text_recognition(source, target):
    # System Video to STT

    res = CLOVA.ClovaSpeechClient().req_upload(file=source, completion='sync')
    json_object = res.json()
    system_text = ''
    for seg in json_object['segments']:
        system_text += seg['text']

    # User Video to STT

    res = CLOVA.ClovaSpeechClient().req_upload(file=target, completion='sync')
    json_object = res.json()
    user_text = ''
    for seg in json_object['segments']:
        user_text += seg['text']
        
    # Comparison with System and User

    tagger = Okt()

    tag_user = tagger.pos(user_text)
    tag_system = tagger.morphs(system_text)

    total = max(len(tag_user), len(tag_system))

    penalty = ['Josa', 'Eomi', 'Punctuation']
    advance = ['Noun', 'Verb']

    cnt = 0 
    for morph, pos in tag_user:
        if (morph in tag_system) and (pos in advance):
            tag_system.remove(morph)
            cnt += 1.25
        elif (morph in tag_system) and (pos in penalty):
            tag_system.remove(morph)
            cnt += 0.75
        elif morph in tag_system:
            tag_system.remove(morph)
            cnt += 1
    score = cnt/total
    score = score * 120
    if (score) > 100:
        return score, system_text, user_text
    else:
        return score, system_text, user_text

def prounce_score(audiofile, system_text):
    data = './data/Audio/api_user_audio.wav'
    down_sample(data, 22050, 16000)

    openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/PronunciationKor"
    accessKey = "b2a87f6e-d634-4a4d-8a2a-4e166dd94560"
    languageCode = "korean"
    script = system_text

    file = open(audiofile, "rb")
    audioContents = base64.b64encode(file.read()).decode("utf8")
    file.close()


    requestJson = {
        "access_key": accessKey,
        "argument": {
            "language_code": languageCode,
            "script": script,
            "audio": audioContents
        }
    }
    
    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    
    score = str(response.data,"utf-8")
    score = round(float(score.split(":")[-1][:-3]),2)
    if score <= 1.0:
        score = 0
    elif (score > 1.0) and (score <= 4.0):
        score * 25
    elif score >= 4.0:
        score = 100

    return score