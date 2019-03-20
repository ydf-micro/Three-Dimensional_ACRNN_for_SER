# -*- coding: utf-8 -*-

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import re

eps = 1e-5


def read_file(filename):
    with wave.open(filename, 'r') as file:
        params = file.getparams()
        nchannels, sampwidth, framerate, wav_length = params[:4]
        str_data = file.readframes(wav_length)
        wavedata = np.fromstring(str_data, dtype=np.short)
        time = np.arange(0, wav_length) * (1.0 / framerate)

    return wavedata, time, framerate


def read_CASIA():
    
    train_num = 2928
    filter_num = 40
    rootdir = '/home/ydf_micro/datasets/IEMOCAP_full_release'
    # horizontal axis denotes the number of Mel-filter bank
    # vertical axis denotes the time(frame) length
    traindata1 = np.empty((train_num * 300, filter_num), dtype=np.float32)
    traindata2 = np.empty((train_num * 300, filter_num), dtype=np.float32)
    traindata3 = np.empty((train_num * 300, filter_num), dtype=np.float32)
    train_num = 0

    for speaker in os.listdir(rootdir):
        if re.search('Session', speaker):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                if re.search('impro', sess):
                    emotdir = emoevl + '/' + sess + '.txt'
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if line[0] == '[':
                                t = line.split()
                                emot_map[t[3]] = t[4]

                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        wavename = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavename]
                        if emotion in ['hap', 'ang', 'neu', 'sad']:
                            data, time, rate = read_file(filename)
                            mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                            delta1 = ps.delta(mel_spec, 2)
                            delta2 = ps.delta(delta1, 2)
                             
                            time = mel_spec.shape[0]
                            if speaker in ['Session1', 'Session2', 'Session3', 'Session4']:
                                # training set
                                if time <= 300:
                                    part = mel_spec
                                    delta11 = delta1
                                    delta21 = delta2
                                    part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)),
                                                  'constant', constant_values=0)
                                    delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)),
                                                     'constant', constant_values=0)
                                    delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)),
                                                     'constant', constant_values=0)
                                    traindata1[train_num * 300:(train_num + 1) * 300] = part
                                    traindata2[train_num * 300:(train_num + 1) * 300] = delta11
                                    traindata3[train_num * 300:(train_num + 1) * 300] = delta21

                                    train_num += 1
                                else:

                                    if emotion in ['ang', 'neu', 'sad']:

                                        for i in range(2):
                                            if i == 0:
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                begin = time - 300
                                                end = time

                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]
                                            traindata1[train_num * 300:(train_num + 1) * 300] = part
                                            traindata2[train_num * 300:(train_num + 1) * 300] = delta11
                                            traindata3[train_num * 300:(train_num + 1) * 300] = delta21
                                            train_num += 1
                                    else:
                                        frames = divmod(time - 300, 100)[0] + 1
                                        for i in range(frames):
                                            begin = 100 * i
                                            end = begin + 300
                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]
                                            traindata1[train_num * 300:(train_num + 1) * 300] = part
                                            traindata2[train_num * 300:(train_num + 1) * 300] = delta11
                                            traindata3[train_num * 300:(train_num + 1) * 300] = delta21
                                            train_num += 1

                            else:
                                pass

                        else:
                            pass

        mean1 = np.mean(traindata1, axis=0)  # axis=0纵轴方向求均值
        std1 = np.std(traindata1, axis=0)
        mean2 = np.mean(traindata2, axis=0)  # axis=0纵轴方向求均值
        std2 = np.std(traindata2, axis=0)
        mean3 = np.mean(traindata3, axis=0)  # axis=0纵轴方向求均值
        std3 = np.std(traindata3, axis=0)
        output = '../data_extraction/zscore.pkl'
        with open(output, 'wb') as f:
            pickle.dump((mean1, std1, mean2, std2, mean3, std3), f)

    return


if __name__=='__main__':
    read_CASIA()
