import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import re

eps = 1e-5
emotion_list = ['hap', 'ang', 'sad', 'neu']     # happy, angry, neutral, sad


def wgn(x, snr):
    snr = 10 ** (snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr

    return np.random.randn(len(x)) * np.sqrt(npower)


def getlogspce(signal, samplerate=16000, winlen=0.02, winstep=0.01,
               nfilt=26, nfft=399, lowfreq=0, highfreq=None, preemph=0.097,
               winfunc=lambda x: np.ones((x,))):
    highfreq = highfreq or samplerate/2
    signal = ps.sigproc.preemphasis(signal, preemph)
    frames = ps.sigproc.framesig(signal, winlen*samplerate, winstep*samplerate, winfunc)
    pspec = ps.sigproc.logpowspec(frames, nfft)

    return pspec


def read_file(filename):
    file = wave.open(filename, 'r')

    # 读取格式信息, return tuple:
    # nchannels, sampwidth, framerate, nframes, comptype, compname
    # 声道数， 量化位数（byte单位）， 采样瓶铝， 采样点数，压缩类型，压缩类型的描述
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]

    # 读取波形数据 返回二进制数据
    str_data = file.readframes(wav_length)

    # 将波形数据转换成数组,因为声音文件是双声道的，因此它由左右声道交替构成，eg:LRLRLRLRLRLR
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0/framerate)

    file.close()

    return wavedata, time, framerate


def dense_to_one_hot(labels_dense, num_classes):
    '''Convert class labels from scalars to one-hot vectors.'''
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def zscore(data, mean, std):
    shape = np.array(data.shape, dtype=np.int32)
    for i in range(shape[0]):
        data[i, :, :, 0] = (data[i, :, :, 0] - mean) / std

    return data


def normalization(data):
    '''
    apply zscore
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    train_data = zscore(train_data, mean, std)
    test_data = zscore(test_data, mean, std)

    :param data:
    :return:
    '''

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std

    return data


def mapminmax(data):
    shape = np.array(data.shape, dtype=np.int32)
    for i in range(shape[0]):
        min = np.min(data[i, :, :, 0])
        max = np.max(data[i, :, :, 0])
        data[i, :, :, 0] = (data[i, :, :, 0] - min) / ((max - min) + eps)

    return data


def generate_label(emotion, classnum):
    label = -1
    if emotion == 'ang':
        label = 0
    elif emotion == 'sad':
        label = 1
    elif emotion == 'hap':
        label = 2
    elif emotion == 'neu':
        label = 3
    elif emotion == 'fear':
        label = 4
    else:
        label = 5

    return label


def load_data():
    with open('../zscore/zscore40.pkl', 'rb') as f:
        mean1, std1, mean2, std2, mean3, std3 = pickle.load(f)

    return mean1, std1, mean2, std2, mean3, std3

def read_IEMOCAP():
    tnum = 259  # the number of test utterance
    vnum = 298
    test_num = 420  # the number of test 2s segments
    valid_num = 436
    train_num = 2928
    filter_num = 40
    pernums_test = np.arange(tnum)  # remerber each utterance contain how many segments
    pernums_valid = np.arange(vnum)
    rootdir = '/home/ydf_micro/datasets/IEMOCAP_full_release'

    mean1, std1, mean2, std2, mean3, std3 = load_data()

    hapnum = 434  # 2
    angnum = 433  # 0
    neunum = 1262  # 3
    sadnum = 799  # 1
    pernum = 300  # np.min([hapnum,angnum,sadnum,neunum])

    # valid_num = divmod((train_num),10)[0]
    train_label = np.empty((train_num, 1), dtype=np.int8)
    test_label = np.empty((tnum, 1), dtype=np.int8)
    valid_label = np.empty((vnum, 1), dtype=np.int8)
    Test_label = np.empty((test_num, 1), dtype=np.int8)
    Valid_label = np.empty((valid_num, 1), dtype=np.int8)

    train_data = np.empty((train_num, 300, filter_num, 3), dtype=np.float32)
    test_data = np.empty((test_num, 300, filter_num, 3), dtype=np.float32)
    valid_data = np.empty((valid_num, 300, filter_num, 3), dtype=np.float32)

    tnum = 0
    vnum = 0
    train_num = 0
    test_num = 0
    valid_num = 0
    train_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    test_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}
    valid_emt = {'hap': 0, 'ang': 0, 'neu': 0, 'sad': 0}

    for sec in os.listdir(rootdir):
        if sec.startswith('Session'):
            sub_dir = os.path.join(rootdir, sec, 'sentences/wav')
            emoevl = os.path.join(rootdir, sec, 'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                if re.search('impro', sess):
                    emotdir = emoevl + '/' + sess + '.txt'
                    emot_map = {}
                    with open(emotdir, 'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if line.startswith('['):
                                t = line.split()
                                emot_map[t[3]] = t[4]

                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)   # Traverse all files that match the *.wav format, return list
                    for filename in files:
                        wavename = filename.split('/')[-1][:-4]
                        emotion = emot_map[wavename]
                        if emotion in emotion_list:
                            data, time, rate = read_file(filename)
                            mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
                            delta1 = ps.delta(mel_spec, 2)
                            delta2 = ps.delta(delta1, 2)

                            time = mel_spec.shape[0]

                            if sec in ['Session1', 'Session2', 'Session3', 'Session4']:
                                # training set
                                if time <= 300:
                                    part = mel_spec
                                    delta11 = delta1
                                    delta21 = delta2
                                    part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)),
                                                  'constant',  constant_values=0)
                                    delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)),
                                                     'constant', constant_values=0)
                                    delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)),
                                                     'constant', constant_values=0)

                                    train_data[train_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                    train_data[train_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                    train_data[train_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)

                                    em = generate_label(emotion, 6)
                                    train_label[train_num] = em
                                    train_emt[emotion] = train_emt[emotion] + 1
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

                                            train_data[train_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                            train_data[train_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                            train_data[train_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)

                                            em = generate_label(emotion, 6)
                                            train_label[train_num] = em
                                            train_emt[emotion] = train_emt[emotion] + 1
                                            train_num += 1
                                    else:
                                        frames = divmod(time-300, 100)[0] + 1
                                        for i in range(frames):
                                            begin = 100 * i
                                            end = begin + 300

                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]

                                            train_data[train_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                            train_data[train_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                            train_data[train_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)

                                            em = generate_label(emotion, 6)
                                            train_label[train_num] = em
                                            train_emt[emotion] = train_emt[emotion] + 1
                                            train_num += 1
                            else:
                                em = generate_label(emotion, 6)
                                if wavename[-4] == 'M':  #male
                                    test_label[tnum] = em
                                    if time <= 300:
                                        pernums_valid[tnum] = 1
                                        part = mel_spec
                                        delta11 = delta1
                                        delta21 = delta2
                                        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)),
                                                      'constant',  constant_values=0)
                                        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)),
                                                         'constant', constant_values=0)
                                        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)),
                                                         'constant', constant_values=0)

                                        train_data[train_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                        train_data[train_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                        train_data[train_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)

                                        test_emt[emotion] = test_emt[emotion] + 1
                                        Test_label[test_num] = em
                                        test_num += 1
                                        tnum += 1

                                    else:
                                        pernums_test[tnum] = 2
                                        tnum = tnum + 1
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
                                            test_data[test_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                            test_data[test_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                            test_data[test_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)

                                            test_emt[emotion] = test_emt[emotion] + 1
                                            Test_label[test_num] = em
                                            test_num += 1

                                else:
                                    # valid_set
                                    em = generate_label(emotion, 6)
                                    valid_label[vnum] = em
                                    if (time <= 300):
                                        pernums_valid[vnum] = 1
                                        part = mel_spec
                                        delta11 = delta1
                                        delta21 = delta2
                                        part = np.pad(part, ((0, 300 - part.shape[0]), (0, 0)), 'constant',
                                                      constant_values=0)
                                        delta11 = np.pad(delta11, ((0, 300 - delta11.shape[0]), (0, 0)), 'constant',
                                                         constant_values=0)
                                        delta21 = np.pad(delta21, ((0, 300 - delta21.shape[0]), (0, 0)), 'constant',
                                                         constant_values=0)
                                        valid_data[valid_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                        valid_data[valid_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                        valid_data[valid_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)
                                        valid_emt[emotion] = valid_emt[emotion] + 1
                                        Valid_label[valid_num] = em
                                        valid_num = valid_num + 1
                                        vnum = vnum + 1
                                    else:
                                        pernums_valid[vnum] = 2
                                        vnum = vnum + 1
                                        for i in range(2):
                                            if (i == 0):
                                                begin = 0
                                                end = begin + 300
                                            else:
                                                end = time
                                                begin = time - 300
                                            part = mel_spec[begin:end, :]
                                            delta11 = delta1[begin:end, :]
                                            delta21 = delta2[begin:end, :]
                                            valid_data[valid_num, :, :, 0] = (part - mean1) / (std1 + eps)
                                            valid_data[valid_num, :, :, 1] = (delta11 - mean2) / (std2 + eps)
                                            valid_data[valid_num, :, :, 2] = (delta21 - mean3) / (std3 + eps)
                                            valid_emt[emotion] = valid_emt[emotion] + 1
                                            Valid_label[valid_num] = em
                                            valid_num += 1

                        else:
                            pass

    hap_index = np.arange(hapnum)
    neu_index = np.arange(neunum)
    sad_index = np.arange(sadnum)
    ang_index = np.arange(angnum)

    h2 = 0
    a0 = 0
    n3 = 0
    s1 = 0

    for l in range(train_num):
        if train_label[l] == 0:
            ang_index[a0] = l
            a0 += 1
        elif train_label[l] == 1:
            sad_index[s1] = l
            s1 += 1
        elif train_label[l] == 2:
            hap_index[h2] = l
            h2 += 1
        else:
            neu_index[n3] = l
            n3 += 1

    np.random.shuffle(neu_index)
    np.random.shuffle(hap_index)
    np.random.shuffle(sad_index)
    np.random.shuffle(ang_index)
    # define emotional array
    hap_label = np.empty((pernum, 1), dtype=np.int8)
    ang_label = np.empty((pernum, 1), dtype=np.int8)
    sad_label = np.empty((pernum, 1), dtype=np.int8)
    neu_label = np.empty((pernum, 1), dtype=np.int8)
    hap_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)
    neu_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)
    sad_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)
    ang_data = np.empty((pernum, 300, filter_num, 3), dtype=np.float32)

    hap_data = train_data[hap_index[0:pernum]].copy()
    hap_label = train_label[hap_index[0:pernum]].copy()
    ang_data = train_data[ang_index[0:pernum]].copy()
    ang_label = train_label[ang_index[0:pernum]].copy()
    sad_data = train_data[sad_index[0:pernum]].copy()
    sad_label = train_label[sad_index[0:pernum]].copy()
    neu_data = train_data[neu_index[0:pernum]].copy()
    neu_label = train_label[neu_index[0:pernum]].copy()
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

    print(train_label.shape)
    print(train_emt)
    print(test_emt)
    print(valid_emt)

    output = '../data_extraction/IEMOCAP.pkl'
    with open(output, 'wb') as f:
        pickle.dump((Train_data, Train_label, test_data, test_label, valid_data, valid_label,
                    Valid_label, Test_label, pernums_test, pernums_valid), f)

    return


if __name__ == '__main__':
    read_IEMOCAP()