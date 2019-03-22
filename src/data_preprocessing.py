# -*- coding: utf-8 -*-

import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import re
import time

eps = 1e-5


def read_file(filename):
    with wave.open(filename, 'r') as file:
        params = file.getparams()
        nchannels, sampwidth, framerate, wav_length = params[:4]
        str_data = file.readframes(wav_length)
        wavedata = np.fromstring(str_data, dtype=np.short)
        time = np.arange(0, wav_length) * (1.0 / framerate)

    return wavedata, time, framerate


def generate_label(emotion):
    label = -1
    if emotion == 'ang':
        label = 0
    elif emotion == 'sad':
        label = 1
    elif emotion == 'hap':
        label = 2
    elif emotion == 'neu':
        label = 3

    return label


def read_IEMOCAP():

    train_utter = 2280  # the number of train utterance
    test_utter = 259  # the number of test utterance
    valid_utter = 298  # the number of validation utterance

    train_num = 2928   # the number of train utterance segments of 3s
    test_num = 420  # the number of test utterance segments of 3s
    valid_num = 436  # the number of valid utterance segments of 3s

    filter_num = 40

    pernums_test = np.arange(test_utter)  # remember each utterance contain how many segments
    pernums_valid = np.arange(valid_utter)

    # # 2774
    angnum = 433  # 0
    sadnum = 799  # 1
    hapnum = 434  # 2
    neunum = 1262  # 3
    pernum = 300  # np.min([hapnum,angnum,sadnum,neunum])

    # label
    test_label = np.empty((test_utter, 1), dtype=np.int8)
    valid_label = np.empty((valid_utter, 1), dtype=np.int8)

    Train_label = np.empty((train_num, 1), dtype=np.int8)
    Test_label = np.empty((test_num, 1), dtype=np.int8)
    Valid_label = np.empty((valid_num, 1), dtype=np.int8)

    rootdir = '/home/ydf_micro/datasets/IEMOCAP_full_release'
    # horizontal axis denotes the number of Mel-filter bank
    # vertical axis denotes the time(frame) length
    train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)  # channels=3, static, deltas, delta-deltas
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)
    valid_data = np.empty((valid_num, 300, filter_num, 3), dtype=np.float32)

    test_utter = 0
    valid_utter = 0

    train_num = 0
    test_num = 0
    valid_num = 0

    train_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    test_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    valid_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}

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
                        if emotion in ['hap', 'ang', 'neu', 'sad']:  # only consider four emotions
                            data, _, rate = read_file(filename)
                            mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                            delta1 = ps.delta(mel_spec, 2)
                            delta2 = ps.delta(delta1, 2)

                            time = mel_spec.shape[0]
                            # eight speakers are selected as the training data and one speaker
                            # is selected as the validation data, while the remaining one speaker
                            # is used as the test data
                            if speaker in ['Session1', 'Session2', 'Session3', 'Session4']:
                                # training set
                                # split the speech signal into equal_length segments of 3s
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

                                    train_data[train_num, :, :, 0] = part
                                    train_data[train_num, :, :, 1] = delta11
                                    train_data[train_num, :, :, 2] = delta21

                                    em = generate_label(emotion)
                                    Train_label[train_num] = em
                                    train_emt[emotion] += 1
                                    train_num += 1
                                else:

                                    if emotion in ['ang', 'neu', 'sad']:
                                        # As time passes 300, the first 300 and the last 300 frames are intercepted
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

                                            train_data[train_num, :, :, 0] = part
                                            train_data[train_num, :, :, 1] = delta11
                                            train_data[train_num, :, :, 2] = delta21

                                            em = generate_label(emotion)
                                            Train_label[train_num] = em
                                            train_emt[emotion] += 1
                                            train_num += 1
                                    else:
                                        frames = divmod(time - 300, 100)[0] + 1
                                        # 100 frame sliding, size of 300 frame
                                        for i in range(frames):
                                            begin = 100 * i
                                            end = begin + 300
                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]

                                            train_data[train_num, :, :, 0] = part
                                            train_data[train_num, :, :, 1] = delta11
                                            train_data[train_num, :, :, 2] = delta21

                                            em = generate_label(emotion)
                                            Train_label[train_num] = em
                                            train_emt[emotion] += 1
                                            train_num += 1

                            else:   # Session5
                                em = generate_label(emotion)
                                if wavename[-4] == 'M':  # male
                                    # test set
                                    test_label[test_utter] = em
                                    if time <= 300:
                                        pernums_test[test_utter] = 1
                                        part = mel_spec
                                        delta11 = delta1
                                        delta21 = delta2
                                        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                                      constant_values=0)
                                        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant',
                                                         constant_values=0)
                                        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant',
                                                         constant_values=0)

                                        test_data[test_num, :, :, 0] = part
                                        test_data[test_num, :, :, 1] = delta11
                                        test_data[test_num, :, :, 2] = delta21

                                        test_emt[emotion] += 1
                                        Test_label[test_num] = em
                                        test_num += 1
                                        test_utter += 1
                                    else:
                                        pernums_test[test_utter] = 2
                                        test_utter += 1
                                        for i in range(2):
                                            if i == 0:
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                end = time
                                                begin = time - 300
                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]

                                            test_data[test_num, :, :, 0] = part
                                            test_data[test_num, :, :, 1] = delta11
                                            test_data[test_num, :, :, 2] = delta21

                                            test_emt[emotion] += 1
                                            Test_label[test_num] = em
                                            test_num += 1
                                else:
                                    # validation set
                                    em = generate_label(emotion)
                                    valid_label[valid_utter] = em
                                    if time <= 300:
                                        pernums_valid[valid_utter] = 1
                                        part = mel_spec
                                        delta11 = delta1
                                        delta21 = delta2

                                        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                                      constant_values=0)
                                        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant',
                                                         constant_values=0)
                                        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant',
                                                         constant_values=0)

                                        valid_data[valid_num, :, :, 0] = part
                                        valid_data[valid_num, :, :, 1] = delta11
                                        valid_data[valid_num, :, :, 2] = delta21

                                        valid_emt[emotion] += 1
                                        Valid_label[valid_num] = em
                                        valid_num += 1
                                        valid_utter += 1
                                    else:
                                        pernums_valid[valid_utter] = 2
                                        valid_utter += 1
                                        for i in range(2):
                                            if i == 0:
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                end = time
                                                begin = time - 300
                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]

                                            valid_data[valid_num, :, :, 0] = part
                                            valid_data[valid_num, :, :, 1] = delta11
                                            valid_data[valid_num, :, :, 2] = delta21

                                            valid_emt[emotion] += 1
                                            Valid_label[valid_num] = em
                                            valid_num += 1
                        else:  # another emotions
                            pass


    # normalization
    train_data = train_data.reshape((-1, filter_num, 3))
    test_data = test_data.reshape((-1, filter_num, 3))
    valid_data = valid_data.reshape((-1, filter_num, 3))
    data_mean = {}
    data_std = {}
    data_mean['static'] = np.mean(train_data[:, :, 0], axis=0)
    data_std['static'] = np.std(train_data[:, :, 0], axis=0)
    data_mean['deltas'] = np.mean(train_data[:, :, 1], axis=0)
    data_std['deltas'] = np.std(train_data[:, :, 1], axis=0)
    data_mean['delta-deltas'] = np.mean(train_data[:, :, 2], axis=0)
    data_std['delta-deltas'] = np.std(train_data[:, :, 2], axis=0)

    # zero mean and unit variance
    for i, key in enumerate(data_mean):
        train_data[:, :, i] = (train_data[:, :, i] - data_mean[key]) / (data_std[key] + eps)
        test_data[:, :, i] = (test_data[:, :, i] - data_mean[key]) / (data_std[key] + eps)
        valid_data[:, :, i] = (valid_data[:, :, i] - data_mean[key]) / (data_std[key] + eps)

    train_data = train_data.reshape((train_num, 300, filter_num, 3))
    test_data = test_data.reshape((test_num, 300, filter_num, 3))
    valid_data = valid_data.reshape((valid_num, 300, filter_num, 3))

    # test_data = test_data.reshape((-1, filter_num, 3))
    # test_mean = {}
    # test_std = {}
    # test_mean['static'] = np.mean(test_data[:, :, 0], axis=0)
    # test_std['static'] = np.std(test_data[:, :, 0], axis=0)
    # test_mean['deltas'] = np.mean(test_data[:, :, 1], axis=0)
    # test_std['deltas'] = np.std(test_data[:, :, 1], axis=0)
    # test_mean['delta-deltas'] = np.mean(test_data[:, :, 2], axis=0)
    # test_std['delta-deltas'] = np.std(test_data[:, :, 2], axis=0)
    #
    # for i, key in enumerate(test_mean):
    #     test_data[:, :, i] = (test_data[:, :, i] - test_mean[key]) / (test_std[key] + eps)
    #
    # test_data = test_data.reshape((test_num, 300, filter_num, 3))
    #
    #
    # valid_data = valid_data.reshape((-1, filter_num, 3))
    # valid_mean = {}
    # valid_std = {}
    # valid_mean['static'] = np.mean(valid_data[:, :, 0], axis=0)
    # valid_std['static'] = np.mean(valid_data[:, :, 0], axis=0)
    # valid_mean['deltas'] = np.mean(valid_data[:, :, 1], axis=0)
    # valid_std['deltas'] = np.mean(valid_data[:, :, 1], axis=0)
    # valid_mean['delta-deltas'] = np.mean(valid_data[:, :, 2], axis=0)
    # valid_std['delta-deltas'] = np.mean(valid_data[:, :, 2], axis=0)
    #
    # for i, key in enumerate(valid_mean):
    #     valid_data[:, :, i] = (valid_data[:, :, i] - valid_mean[key]) / (valid_std[key] + eps)
    #
    # valid_data = valid_data.reshape((valid_num, 300, filter_num, 3))

    # train_seq = np.arange(train_num)
    # test_seq = np.arange(test_num)
    # valid_seq = np.arange(valid_num)
    #
    # np.random.shuffle(train_seq)
    # np.random.shuffle(test_seq)
    # np.random.shuffle(valid_seq)

    # print(train_data)
    # print(test_data)
    # print(valid_data)

    hap_index = np.arange(hapnum)
    neu_index = np.arange(neunum)
    sad_index = np.arange(sadnum)
    ang_index = np.arange(angnum)

    a0, s1, h2, n3 = 0, 0, 0, 0

    for l in range(train_num):
        if Train_label[l] == 0:
            ang_index[a0] = l
            a0 = a0 + 1
        elif Train_label[l] == 1:
            sad_index[s1] = l
            s1 = s1 + 1
        elif Train_label[l] == 2:
            hap_index[h2] = l
            h2 = h2 + 1
        else:
            neu_index[n3] = l
            n3 = n3 + 1

    # shuffle the sequence
    np.random.shuffle(neu_index)
    np.random.shuffle(hap_index)
    np.random.shuffle(sad_index)
    np.random.shuffle(ang_index)

    hap_data = train_data[hap_index[0:pernum]].copy()
    hap_label = Train_label[hap_index[0:pernum]].copy()

    ang_data = train_data[ang_index[0:pernum]].copy()
    ang_label = Train_label[ang_index[0:pernum]].copy()

    sad_data = train_data[sad_index[0:pernum]].copy()
    sad_label = Train_label[sad_index[0:pernum]].copy()

    neu_data = train_data[neu_index[0:pernum]].copy()
    neu_label = Train_label[neu_index[0:pernum]].copy()

    train_num = 4 * pernum

    Train_label = np.empty((train_num, 1), dtype=np.int8)
    Train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)
    Train_data[0:pernum] = hap_data
    Train_label[0:pernum] = hap_label
    Train_data[pernum:2 * pernum] = sad_data
    Train_label[pernum:2 * pernum] = sad_label
    Train_data[2 * pernum:3 * pernum] = neu_data
    Train_label[2 * pernum:3 * pernum] = neu_label
    Train_data[3 * pernum:4 * pernum] = ang_data
    Train_label[3 * pernum:4 * pernum] = ang_label

    arr = np.arange(train_num)
    np.random.shuffle(arr)
    Train_data = Train_data[arr[0:]]
    Train_label = Train_label[arr[0:]]

    # print(Train_label.shape)
    # print(train_emt)
    # print(test_emt)
    # print(valid_emt)

    # print(train_data.shape, Train_label.shape)
    # print(test_data.shape, test_label.shape)
    # print(valid_data.shape, valid_label.shape)
    # print(Valid_label.shape, Test_label.shape)


    output = '../data/IEMOCAP.pkl'
    with open(output, 'wb') as f:
        pickle.dump((Train_data, Train_label, test_data, test_label, valid_data, valid_label,
                     Valid_label, Test_label, pernums_test, pernums_valid), f)

    return


if __name__ == '__main__':
    start = time.time()
    read_IEMOCAP()
    end = time.time()
    print('所用时间:{:.2f}min'.format((end-start)/60))
