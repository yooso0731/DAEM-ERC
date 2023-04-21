# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 12:39:12 2023

@author: user
"""

import random
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

import librosa
import librosa.display
import matplotlib.pyplot as plt

import itertools


# audio data transform functions
class AudioUtill():
    # Load an audio file
    @staticmethod
    def open(audio_file):
        sig, sr = librosa.load(audio_file, sr=16000) # (1, sr*seconds)
        sig = torch.Tensor(sig)
        sig = torch.unsqueeze(sig, 0)
        return (sig, sr)
        
    # data augmentation (row audio)
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random()*shift_limit*sig_len)
        return (sig.roll(shift_amt), sr)
    
    @staticmethod
    def add_noise(aud, noise_factor):
        sig, sr = aud
        _, sig_len = sig.shape
        noise = torch.randn(sig_len)
        sig = sig + noise_factor * noise
        return (sig, sr)
    
    @staticmethod
    def change_pitch(aud, pitch_factor):
        sig, sr = aud
        sig = sig.numpy()
        sig = librosa.effects.pitch_shift(y=sig, sr=sr, n_steps=pitch_factor)
        sig = torch.Tensor(sig)
        return (sig, sr)
    
    @staticmethod
    def streching(aud, str_factor=1.1):
        sig, sr = aud
        sig = sig.numpy().astype('float')
        sig = librosa.effects.time_stretch(sig, rate=str_factor)
        sig = torch.Tensor(sig)
        return (sig, sr)
    
    # resize (length: 16000*8)
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        _, sig_len = sig.shape
        max_len = sr * max_ms 
        
        if (sig_len > max_len):
            sig = sig[0, :max_len]
            sig = torch.unsqueeze(sig, 0)
        elif (sig_len < max_len):
            pad_begin_len = random.randint(0, max_len-sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len
            
            pad_begin = torch.zeros((1, pad_begin_len))
            pad_end = torch.zeros((1, pad_end_len))
            
            sig = torch.cat((pad_begin, sig, pad_end), 1)
            
        return (sig, sr)
    
    # to MFCC
    @staticmethod
    def mfcc(aud, n_mfcc=40, n_fft=1024, hop_len=512):
        sig, sr = aud
        sig = sig.numpy()
        mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_len)
        mfcc = torch.Tensor(mfcc)
        return mfcc 
    
    # to Mel-spectrogram
    @staticmethod
    def mel_spectrogram(aud, n_mels=40, n_fft=1024, hop_len=512):
        sig, sr = aud
        sig = sig.numpy()
        mel = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_len)
        mel = librosa.amplitude_to_db(mel, ref=np.max)
        mel = torch.Tensor(mel)
        return mel
    
    @staticmethod
    def mfccAndMel(aud, n_mfcc=40, n_mels=40, n_fft=1024, hop_len=512):
        sig, sr = aud
        sig = sig.numpy()
        #mfcc
        mfcc = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_lenth=hop_len)
        mfcc = torch.Tensor(mfcc)
        #mel-spectrogram
        mel = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_len)
        mel = librosa.amplitude_to_db(mel, ref=np.max)
        mel = torch.Tensor(mel)
        cat = torch.cat((mfcc, mel), 0)
        return cat
    

# get data
class data_load():
    def __init__(self, df, data_path):
        ##################################
        # df: annotation data file
        # data_path: path to save dataset
        ##################################
        
        self.df = df
        self.data_path = str(data_path)
        self.duration = 8
        self.sr = 16000
        #self.shift_pct = 6 
        #self.noise_factor = np.random.uniform(0, 0.01)
        #self.pitch_factor = np.random.randint(-5, 5)
        self.mu = 0 
        self.std = 0
        
    def augmentation(self, aud, size, transformType):
        ########################################
        # aud: audio file, shape: (signal, sampling rate)
        # size: # of augmentation
        # transformType: {'mfcc', 'mel', 'both'}
        ## Return:: augmentation signal(sig, sr) list
        ########################################
        
        aug_type = ['add_noise', 'time_shift', 'change_pitch', 'streching']
        sampling_type_ = np.random.choice(range(len(aug_type)), size=size, replace=True)
        sampling_type = [aug_type[i] for i in sampling_type_]
        aug_list = []
        for sp_type in sampling_type:
            if sp_type == 'add_noise':
                factor = np.random.uniform(0, 0.01)
                tran_sig = AudioUtill.add_noise(aud, factor)
            elif sp_type == 'time_shift':
                tran_sig = AudioUtill.time_shift(aud, 6)
            elif sp_type == 'change_pitch':
                factor = np.random.randint(-5, 5)
                tran_sig = AudioUtill.change_pitch(aud, factor)
            elif sp_type == 'streching':
                factor = np.random.choice([0.75, 0.8, 0.85, 0.9, 1.1, 1.15, 1.2, 1.25])
                tran_sig = AudioUtill.change_pitch(aud, factor)
            tran_sig = AudioUtill.pad_trunc(tran_sig, self.duration)
            
            if transformType == 'mfcc':
                trans = AudioUtill.mfcc(tran_sig)
            elif transformType == 'mel':
                trans = AudioUtill.mel_spectrogram(tran_sig)
            elif transformType == 'both':
                trans = AudioUtill.mfccAndMel(tran_sig)
            aug_list.append(trans)
        return aug_list
    
    def get_data(self, method, aug=False, aug_dict=None): 
        ####################################
        # method: {'mfcc', 'mel', 'both'}
        # aug: {True, False}
        # aug_dict: {emotion_id: augmentation size} dict
        ## Return: final dataset (x, y)
        ####################################
        
        get_df = self.df
        emotion_id = get_df.emotion_id.tolist()
        
        x = []
        y = []
        for i in tqdm(range(len(get_df))):
            audio_file = self.data_path + '/Session'+get_df.iloc[i, 0][4:6] + '/' + get_df.iloc[i, 0]+'.wav'
            emotionId = get_df.emotion_id.tolist()[i]
            
            
            aud = AudioUtill.open(audio_file)
            if aug:  # data augmentation (wav file)
                aug_size = aug_dict[emotionId]
                sig_list = self.augmentation(aud=aud, size=aug_size, transformType=method)
                x = x + sig_list
                y = y + list(itertools.repeat(emotionId, aug_size))
            else: 
                dur_aud = AudioUtill.pad_trunc(aud, self.duration) # resize and zero padding (random)
                if method == 'mfcc':
                    trans_x = AudioUtill.mfcc(dur_aud)
                    x.append(trans_x)
                    y.append(emotionId)
                elif method == 'mel':
                    trans_x = AudioUtill.mel_spectrogram(dur_aud)
                    x.append(trans_x)
                    y.append(emotionId)
                elif method == 'both':
                    trans_x = AudioUtill.mfccAndMel(dur_aud)
                    x.append(trans_x)
                    y.append(emotionId)

        x = torch.stack(x)
        y = torch.LongTensor(y)
        return x, y
    
    def z_score_norm(self, data, fold):
        #################################
        # data: data to normalize
        # fold: {'train', 'test'} 
        #       if fold==train; mean, std save. else load
        ## Return: finished dataset
        #################################
        if fold=='train':
            self.mu = data.mean()
            self.std = data.std()
            data = (data - self.mu) / self.std
        elif fold=='test':
            if (self.mu==0) & (self.std==0):
                raise Exception('ERROR: Mean and std values of train data settings are required')
            else:
                data = (data - self.mu) / self.std
        return data