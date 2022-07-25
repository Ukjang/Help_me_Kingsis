import os
import moviepy.editor as mp
import matplotlib.pyplot
from pydub import AudioSegment, silence
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import librosa
from sklearn.preprocessing import minmax_scale
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def make_wave_file(videoname, directory):
    try:
        os.mkdir('./data/Audio')
    except:
        print('Directory is already existed')
        pass
    audiosegment = AudioSegment.from_file(directory + videoname+'.mp4')
    audiosegment.export('./data/Audio/' +videoname +'.wav', format='wav')

# sound is original video to wav file
def detect_leading_silence(sound, silence_threshold=-40.0, chunk_size=10):
    trim_ms = 0

    assert chunk_size > 0
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms

def match_target_amplitude(aChunk, target_dBFS):
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def DTW(A, B, window = sys.maxsize, d = lambda x,y: abs(x-y)):
    A, B = np.array(A), np.array(B)
    M, N = A.size, B.size
    cost = sys.maxsize * np.ones((M, N))

    cost[0, 0] = d(A[0], B[0])
    for i in range(1, M):
        cost[i, 0] = cost[i-1, 0] + d(A[i], B[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j-1] + d(A[0], B[j])

    for i in range(1, M):
        for j in range(max(1, i - window), min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
            cost[i, j] = min(choices) + d(A[i], B[j])

    n, m = N - 1, M - 1
    path = []

    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key = lambda x: cost[x[0], x[1]])
    
    path.append((0,0))
    return cost[-1, -1], path

def MFCC(source, target, export_dir):
    config = {'SR':44000,
        'N_MFCC':13}

    sample_rate = config['SR']
    n_mfcc = config['N_MFCC']

    videos = [source, target]
    names = ['system', 'user']
    for video, name in zip(videos, names):
        sound = AudioSegment.from_file(video, format='wav')

        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)
        trimmed_sound = sound[start_trim:duration-end_trim]

        dBFS = trimmed_sound.dBFS

        chunks = split_on_silence(trimmed_sound, min_silence_len=1000, silence_thresh=dBFS-16)

        for chunk in chunks:
            silence_chunk = AudioSegment.silent(duration=1000)
            audio_chunk = silence_chunk + chunk + silence_chunk
            normalized_chunk = match_target_amplitude(audio_chunk, -20.0)
            out_file = export_dir+ f'out_audio_{name}.wav'
            out_file_api = export_dir + f'api_{name}_audio.wav'
            print("exporting", out_file)
            chunk.export(out_file, format="wav")
            normalized_chunk.export(out_file_api, format="wav")

    y1, sr1 = librosa.load(export_dir + 'out_audio_system.wav', sample_rate)
    mfcc_system = librosa.feature.mfcc(y=y1, sr=16000, n_mfcc=n_mfcc)
    mfcc_system = minmax_scale(mfcc_system, axis=1)

    y2, sr2 = librosa.load(export_dir + 'out_audio_user.wav', sample_rate)
    mfcc_user = librosa.feature.mfcc(y=y2, sr=16000, n_mfcc=n_mfcc)
    mfcc_user = minmax_scale(mfcc_user, axis=1)

    total = 0
    arr = np.array([])

    for i in range(len(mfcc_system)):
        A, B = mfcc_system[i], mfcc_user[i]
        cost, path = DTW(A, B, window=999999)
        arr = np.append(arr, cost)
        total += cost

    max_cost = np.where(arr == max(arr))[0][0]
    mfcc_score = total/len(mfcc_system)
    offset = 5

    plt.plot(A)
    plt.plot(B + offset)
    for (x1, x2) in path:
        plt.plot([x1, x2], [A[x1], B[x2] + offset])
    plt.show()

    x = np.array(range(len(mfcc_system.mean(axis=0))))
    y = np.array(mfcc_system.mean(axis=0).tolist())

    x1 = np.array(range(len(mfcc_user.mean(axis=0))))
    y1 = np.array(mfcc_user.mean(axis=0).tolist())

    xnew = np.linspace(x.min(), x.max(), 200)
    xnew1 = np.linspace(x1.min(), x1.max(), 200)  

    spl = make_interp_spline(x, y, k=7)
    y_smooth = spl(xnew)
    spl1 = make_interp_spline(x1, y1, k=7)
    y_smooth1 = spl1(xnew1)

    plt.plot(xnew, y_smooth)
    plt.plot(xnew1, y_smooth1)
    plt.ylim([0,1])
    plt.show()

    return mfcc_score