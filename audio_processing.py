#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音声感情推測ゲーム「はぁっていうゲーム」
音声処理モジュール

このモジュールは以下の機能を提供します：
1. リアルタイム音声入力（PyAudio）
2. 音声特徴量抽出（Librosa）
"""

import os
import numpy as np
import pyaudio
import wave
import time
import threading
import librosa
import librosa.display
import matplotlib.pyplot as plt
from queue import Queue

class AudioInputModule:
    """音声入力を処理するクラス"""
    
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                 format_type=pyaudio.paInt16, record_seconds=3):
        """
        音声入力モジュールの初期化
        
        Parameters:
        -----------
        sample_rate : int
            サンプリングレート（デフォルト: 16000Hz）
        chunk_size : int
            バッファサイズ（デフォルト: 1024サンプル）
        channels : int
            チャンネル数（デフォルト: 1=モノラル）
        format_type : int
            音声フォーマット（デフォルト: 16-bit PCM）
        record_seconds : int
            録音時間（デフォルト: 3秒）
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format_type
        self.record_seconds = record_seconds
        
        self.p = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.audio_data = None
        self.recording_thread = None
        self.audio_queue = Queue()
        
    def initialize(self):
        """PyAudioの初期化"""
        try:
            self.p = pyaudio.PyAudio()
            return True
        except Exception as e:
            print(f"音声入力の初期化に失敗しました: {e}")
            return False
    
    def list_devices(self):
        """利用可能な音声入力デバイスを一覧表示"""
        if not self.p:
            self.initialize()
            
        info = []
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # 入力デバイスのみ
                info.append({
                    'index': i,
                    'name': device_info['name'],
                    'channels': device_info['maxInputChannels'],
                    'sample_rate': int(device_info['defaultSampleRate'])
                })
        return info
    
    def start_recording(self, device_index=None):
        """
        録音を開始
        
        Parameters:
        -----------
        device_index : int or None
            使用する入力デバイスのインデックス（Noneの場合はデフォルトデバイス）
        """
        if self.is_recording:
            print("すでに録音中です")
            return False
            
        if not self.p:
            self.initialize()
        
        try:
            self.stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            self.frames = []
            
            # 別スレッドで録音を実行
            self.recording_thread = threading.Thread(target=self._record)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            return True
            
        except Exception as e:
            print(f"録音の開始に失敗しました: {e}")
            return False
    
    def _record(self):
        """録音処理の内部実装（スレッド実行用）"""
        try:
            # 録音開始
            print("録音を開始しました...")
            
            # チャンク数の計算
            chunk_count = int(self.sample_rate / self.chunk_size * self.record_seconds)
            
            for _ in range(chunk_count):
                if not self.is_recording:
                    break
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
                
                # リアルタイム処理用にキューにデータを追加
                audio_array = np.frombuffer(data, dtype=np.int16)
                self.audio_queue.put(audio_array)
            
            # 自動停止
            if self.is_recording:
                self.stop_recording()
                
        except Exception as e:
            print(f"録音中にエラーが発生しました: {e}")
            self.is_recording = False
    
    def stop_recording(self):
        """録音を停止"""
        if not self.is_recording:
            print("録音が開始されていません")
            return False
        
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # 録音データをnumpy配列に変換
        if self.frames:
            audio_bytes = b''.join(self.frames)
            self.audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            # float32に正規化 (-1.0 to 1.0)
            self.audio_data = self.audio_data.astype(np.float32) / 32768.0
        
        print("録音を停止しました")
        return True
    
    def save_audio(self, filename="recorded_audio.wav"):
        """
        録音した音声をWAVファイルとして保存
        
        Parameters:
        -----------
        filename : str
            保存するファイル名
        """
        if not self.frames:
            print("保存する録音データがありません")
            return False
        
        try:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            print(f"音声を {filename} に保存しました")
            return True
        except Exception as e:
            print(f"音声の保存に失敗しました: {e}")
            return False
    
    def get_audio_data(self):
        """録音した音声データを取得"""
        return self.audio_data
    
    def is_audio_valid(self):
        """音声品質チェック（音量レベルなど）"""
        if self.audio_data is None or len(self.audio_data) == 0:
            return False
        
        # 音量レベルチェック
        rms = np.sqrt(np.mean(np.square(self.audio_data)))
        if rms < 0.01:  # 閾値は調整可能
            print("音声レベルが低すぎます")
            return False
            
        return True
    
    def cleanup(self):
        """リソースの解放"""
        if self.stream:
            self.stream.close()
        
        if self.p:
            self.p.terminate()
            self.p = None


class FeatureExtractor:
    """音声特徴量を抽出するクラス"""
    
    def __init__(self, sample_rate=16000):
        """
        特徴量抽出モジュールの初期化
        
        Parameters:
        -----------
        sample_rate : int
            サンプリングレート（デフォルト: 16000Hz）
        """
        self.sample_rate = sample_rate
        
    def extract_features(self, audio_data):
        """
        音声データから特徴量を抽出
        
        Parameters:
        -----------
        audio_data : numpy.ndarray
            音声データ（float32形式、-1.0〜1.0の範囲）
            
        Returns:
        --------
        dict
            抽出された特徴量の辞書
        """
        if audio_data is None or len(audio_data) == 0:
            print("特徴量抽出用の音声データがありません")
            return None
        
        features = {}
        
        # MFCC特徴量
        mfcc_features = self.extract_mfcc(audio_data)
        if mfcc_features is not None:
            features['mfcc'] = mfcc_features
        
        # 韻律特徴量
        prosodic_features = self.extract_prosodic_features(audio_data)
        if prosodic_features is not None:
            features.update(prosodic_features)
        
        # 特徴量の正規化
        features = self.normalize_features(features)
        
        return features
    
    def extract_mfcc(self, audio_data, n_mfcc=13):
        """
        MFCC特徴量の抽出
        
        Parameters:
        -----------
        audio_data : numpy.ndarray
            音声データ
        n_mfcc : int
            MFCCの次元数（デフォルト: 13）
            
        Returns:
        --------
        numpy.ndarray
            MFCC特徴量
        """
        try:
            # MFCCの計算
            mfcc = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=n_mfcc,
                hop_length=int(self.sample_rate * 0.010),  # 10msフレームシフト
                n_fft=int(self.sample_rate * 0.025)  # 25msフレーム長
            )
            
            # デルタMFCCの計算
            delta_mfcc = librosa.feature.delta(mfcc)
            
            # デルタデルタMFCCの計算
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # 統計量の計算（平均と標準偏差）
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            delta_mfcc_mean = np.mean(delta_mfcc, axis=1)
            delta_mfcc_std = np.std(delta_mfcc, axis=1)
            delta2_mfcc_mean = np.mean(delta2_mfcc, axis=1)
            delta2_mfcc_std = np.std(delta2_mfcc, axis=1)
            
            # 特徴量ベクトルの結合
            mfcc_features = np.concatenate([
                mfcc_mean, mfcc_std, 
                delta_mfcc_mean, delta_mfcc_std,
                delta2_mfcc_mean, delta2_mfcc_std
            ])
            
            return mfcc_features
            
        except Exception as e:
            print(f"MFCC特徴量の抽出に失敗しました: {e}")
            return None
    
    def extract_prosodic_features(self, audio_data):
        """
        韻律特徴量の抽出
        
        Parameters:
        -----------
        audio_data : numpy.ndarray
            音声データ
            
        Returns:
        --------
        dict
            韻律特徴量の辞書
        """
        try:
            features = {}
            
            # 基本周波数（F0）の抽出
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            
            # F0の統計量
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features['f0_mean'] = np.mean(f0_valid)
                features['f0_std'] = np.std(f0_valid)
                features['f0_min'] = np.min(f0_valid)
                features['f0_max'] = np.max(f0_valid)
                features['f0_range'] = features['f0_max'] - features['f0_min']
            else:
                # 有効なF0がない場合はゼロで埋める
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['f0_min'] = 0
                features['f0_max'] = 0
                features['f0_range'] = 0
            
            # エネルギー（RMS）の抽出
            rms = librosa.feature.rms(y=audio_data)[0]
            features['energy_mean'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            features['energy_max'] = np.max(rms)
            
            # ゼロ交差率（音声の周波数に関連）
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # スペクトル重心（音色の明るさに関連）
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            # スペクトルフラックス（音色の変化に関連）
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
            features['spectral_flux_mean'] = np.mean(onset_env)
            features['spectral_flux_std'] = np.std(onset_env)
            
            return features
            
        except Exception as e:
            print(f"韻律特徴量の抽出に失敗しました: {e}")
            return None
    
    def normalize_features(self, features):
        """
        特徴量の正規化
        
        Parameters:
        -----------
        features : dict
            特徴量の辞書
            
        Returns:
        --------
        dict
            正規化された特徴量の辞書
        """
        if features is None:
            return None
        
        normalized_features = {}
        
        # MFCC特徴量の正規化
        if 'mfcc' in features:
            mfcc = features['mfcc']
            # 平均0、標準偏差1に正規化
            mfcc_mean = np.mean(mfcc)
            mfcc_std = np.std(mfcc)
            if mfcc_std > 0:
                normalized_features['mfcc'] = (mfcc - mfcc_mean) / mfcc_std
            else:
                normalized_features['mfcc'] = mfcc
        
        # 韻律特徴量の正規化
        prosodic_keys = [k for k in features.keys() if k != 'mfcc']
        if prosodic_keys:
            prosodic_values = np.array([features[k] for k in prosodic_keys])
            # 平均0、標準偏差1に正規化
            p_mean = np.mean(prosodic_values)
            p_std = np.std(prosodic_values)
            if p_std > 0:
                normalized_values = (prosodic_values - p_mean) / p_std
                for i, key in enumerate(prosodic_keys):
                    normalized_features[key] = normalized_values[i]
            else:
                for i, key in enumerate(prosodic_keys):
                    normalized_features[key] = prosodic_values[i]
        
        return normalized_features
    
    def visualize_features(self, audio_data, features=None, filename=None):
        """
        特徴量の可視化
        
        Parameters:
        -----------
        audio_data : numpy.ndarray
            音声データ
        features : dict or None
            抽出された特徴量（Noneの場合は内部で抽出）
        filename : str or None
            保存するファイル名（Noneの場合は表示のみ）
        """
        if features is None:
            features = self.extract_features(audio_data)
            
        if features is None:
            print("可視化する特徴量がありません")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 波形
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(audio_data, sr=self.sample_rate)
        plt.title('波形')
        
        # MFCCスペクトログラム
        if 'mfcc' in features:
            plt.subplot(3, 1, 2)
            mfcc = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=13
            )
            librosa.display.specshow(
                mfcc, 
                sr=self.sample_rate, 
                x_axis='time'
            )
            plt.colorbar()
            plt.title('MFCC')
        
        # 韻律特徴量
        plt.subplot(3, 1, 3)
        prosodic_keys = [k for k in features.keys() if k != 'mfcc']
        if prosodic_keys:
            prosodic_values = [features[k] for k in prosodic_keys]
            plt.bar(prosodic_keys, prosodic_values)
            plt.xticks(rotation=45, ha='right')
            plt.title('韻律特徴量')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"特徴量の可視化を {filename} に保存しました")
        else:
            plt.show()


def test_audio_input():
    """音声入力モジュールのテスト"""
    audio_input = AudioInputModule(record_seconds=3)
    
    if not audio_input.initialize():
        print("音声入力の初期化に失敗しました")
        return
    
    # 利用可能なデバイスの表示
    devices = audio_input.list_devices()
    print("利用可能な音声入力デバイス:")
    for device in devices:
        print(f"  インデックス: {device['index']}, 名前: {device['name']}")
    
    # 録音開始
    print("3秒間の録音を開始します...")
    audio_input.start_recording()
    
    # 録音が終了するまで待機
    while audio_input.is_recording:
        time.sleep(0.1)
    
    # 録音データの保存
    audio_input.save_audio("test_recording.wav")
    
    # 音声品質チェック
    if audio_input.is_audio_valid():
        print("録音品質は良好です")
    else:
        print("録音品質に問題があります")
    
    # リソースの解放
    audio_input.cleanup()


def test_feature_extraction():
    """特徴量抽出モジュールのテスト"""
    # 音声ファイルの読み込み
    try:
        audio_file = "test_recording.wav"
        if not os.path.exists(audio_file):
            print(f"ファイル {audio_file} が見つかりません")
            return
            
        audio_data, sample_rate = librosa.load(audio_file, sr=16000)
        
        # 特徴量抽出
        extractor = FeatureExtractor(sample_rate=sample_rate)
        features = extractor.extract_features(audio_data)
        
        # 特徴量の表示
        print("抽出された特徴量:")
        for key, value in features.items():
            if key == 'mfcc':
                print(f"  {key}: shape={value.shape}")
            else:
                print(f"  {key}: {value}")
        
        # 特徴量の可視化
        extractor.visualize_features(audio_data, features, "feature_visualization.png")
        
    except Exception as e:
        print(f"特徴量抽出テストに失敗しました: {e}")


if __name__ == "__main__":
    print("音声処理モジュールのテスト")
    print("-----------------------")
    
    # 音声入力のテスト
    test_audio_input()
    
    # 特徴量抽出のテスト
    test_feature_extraction()
