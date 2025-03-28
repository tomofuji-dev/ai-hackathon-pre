#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音声感情推測ゲーム「はぁっていうゲーム」
感情認識モデルモジュール

このモジュールは以下の機能を提供します：
1. CNN-LSTMハイブリッドモデルの構築（TensorFlow/Keras）
2. 感情認識モデルの学習と評価
3. 感情推論と結果の解釈
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class EmotionRecognizer:
    """感情認識モデルを構築・学習・評価するクラス"""
    
    def __init__(self, model_path=None, emotion_labels=None):
        """
        感情認識モデルの初期化
        
        Parameters:
        -----------
        model_path : str or None
            学習済みモデルのパス（Noneの場合は新規モデル）
        emotion_labels : list or None
            感情ラベルのリスト
        """
        # 感情ラベル（デフォルト）
        self.emotion_labels = emotion_labels or [
            "喜び", "怒り", "悲しみ", "驚き", "不安", "嫌悪", 
            "困惑", "照れ", "呆れ", "疑い", "感動", "軽蔑", "中立"
        ]
        self.num_classes = len(self.emotion_labels)
        
        # ラベルエンコーダー
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.emotion_labels)
        
        # 特徴量スケーラー
        self.feature_scaler = StandardScaler()
        
        # モデル
        self.model = None
        self.svm_model = None  # 代替モデル（SVM）
        
        # 学習済みモデルのロード
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # 最新の予測結果
        self.latest_prediction = None
        self.confidence_scores = None
    
    def build_model(self, input_shape, model_type='hybrid'):
        """
        感情認識モデルの構築
        
        Parameters:
        -----------
        input_shape : tuple
            入力特徴量の形状
        model_type : str
            モデルタイプ ('hybrid', 'cnn', 'lstm', 'dense')
            
        Returns:
        --------
        tensorflow.keras.models.Model
            構築されたモデル
        """
        if model_type == 'hybrid':
            # CNN-LSTMハイブリッドモデル
            
            # 入力層
            inputs = Input(shape=input_shape)
            
            # CNN部分
            x_cnn = Conv1D(64, 3, activation='relu', padding='same')(inputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Conv1D(128, 3, activation='relu', padding='same')(x_cnn)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Conv1D(256, 3, activation='relu', padding='same')(x_cnn)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
            
            # LSTM部分
            x_lstm = LSTM(128, return_sequences=True)(inputs)
            x_lstm = LSTM(64)(x_lstm)
            
            # 特徴の結合
            x = Concatenate()([x_cnn, x_lstm])
            
            # 全結合層
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            
            # 出力層
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            # モデル構築
            model = Model(inputs=inputs, outputs=outputs)
            
        elif model_type == 'cnn':
            # CNNモデル
            model = Sequential([
                Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                Conv1D(128, 3, activation='relu', padding='same'),
                MaxPooling1D(pool_size=2),
                Conv1D(256, 3, activation='relu', padding='same'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
            
        elif model_type == 'lstm':
            # LSTMモデル
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=input_shape),
                LSTM(64),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
            
        else:  # 'dense'
            # 単純なDNNモデル
            model = Sequential([
                Flatten(input_shape=input_shape),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
        
        # モデルのコンパイル
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_svm_model(self):
        """SVMモデルの構築（代替モデル）"""
        self.svm_model = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True
        )
        return self.svm_model
    
    def prepare_data(self, features, labels, test_size=0.2, validation_split=0.2):
        """
        学習データの準備
        
        Parameters:
        -----------
        features : numpy.ndarray
            特徴量データ
        labels : numpy.ndarray or list
            ラベルデータ
        test_size : float
            テストデータの割合
        validation_split : float
            検証データの割合（学習データから分割）
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # ラベルのエンコード
        y_encoded = self.label_encoder.transform(labels)
        y_categorical = to_categorical(y_encoded, num_classes=self.num_classes)
        
        # 学習・テストデータの分割
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, y_categorical, test_size=test_size, stratify=y_categorical, random_state=42
        )
        
        # 学習・検証データの分割
        val_size = validation_split / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
        )
        
        # 特徴量のスケーリング
        self.feature_scaler.fit(X_train.reshape(X_train.shape[0], -1))
        
        X_train_scaled = self.feature_scaler.transform(X_train.reshape(X_train.shape[0], -1))
        X_val_scaled = self.feature_scaler.transform(X_val.reshape(X_val.shape[0], -1))
        X_test_scaled = self.feature_scaler.transform(X_test.reshape(X_test.shape[0], -1))
        
        # 元の形状に戻す
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, model_path='emotion_model.h5'):
        """
        モデルの学習
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            学習データの特徴量
        y_train : numpy.ndarray
            学習データのラベル
        X_val : numpy.ndarray
            検証データの特徴量
        y_val : numpy.ndarray
            検証データのラベル
        epochs : int
            エポック数
        batch_size : int
            バッチサイズ
        model_path : str
            モデル保存パス
            
        Returns:
        --------
        tensorflow.keras.callbacks.History
            学習履歴
        """
        if self.model is None:
            print("モデルが構築されていません。build_model()を先に呼び出してください。")
            return None
        
        # コールバック
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # モデルの学習
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def train_svm(self, X_train, y_train):
        """
        SVMモデルの学習（代替モデル）
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            学習データの特徴量
        y_train : numpy.ndarray
            学習データのラベル
            
        Returns:
        --------
        sklearn.svm.SVC
            学習済みSVMモデル
        """
        if self.svm_model is None:
            self.build_svm_model()
        
        # 特徴量の平坦化
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # ラベルの変換（one-hotからインデックスへ）
        y_train_indices = np.argmax(y_train, axis=1)
        
        # モデルの学習
        self.svm_model.fit(X_train_flat, y_train_indices)
        
        return self.svm_model
    
    def evaluate(self, X_test, y_test):
        """
        モデルの評価
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            テストデータの特徴量
        y_test : numpy.ndarray
            テストデータのラベル
            
        Returns:
        --------
        dict
            評価結果
        """
        if self.model is None:
            print("モデルが構築されていません。")
            return None
        
        # モデル評価
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # 予測
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        
        # 分類レポート
        class_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # 結果をまとめる
        results = {
            'accuracy': accuracy,
            'loss': loss,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        return results
    
    def evaluate_svm(self, X_test, y_test):
        """
        SVMモデルの評価（代替モデル）
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            テストデータの特徴量
        y_test : numpy.ndarray
            テストデータのラベル
            
        Returns:
        --------
        dict
            評価結果
        """
        if self.svm_model is None:
            print("SVMモデルが構築されていません。")
            return None
        
        # 特徴量の平坦化
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # ラベルの変換（one-hotからインデックスへ）
        y_true = np.argmax(y_test, axis=1)
        
        # 予測
        y_pred = self.svm_model.predict(X_test_flat)
        y_pred_proba = self.svm_model.predict_proba(X_test_flat)
        
        # 精度
        accuracy = np.mean(y_pred == y_true)
        
        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        
        # 分類レポート
        class_names = self.label_encoder.classes_
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # 結果をまとめる
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'probabilities': y_pred_proba
        }
        
        return results
    
    def predict(self, features):
        """
        感情予測
        
        Parameters:
        -----------
        features : numpy.ndarray
            特徴量データ
            
        Returns:
        --------
        str
            予測された感情ラベル
        """
        if self.model is None:
            print("モデルが構築されていません。")
            return None
        
        # 特徴量の前処理
        features_reshaped = features.reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features_reshaped)
        
        # 予測用に形状を調整
        if len(self.model.input_shape) > 2:
            time_steps = self.model.input_shape[1]
            n_features = self.model.input_shape[2]
            features_scaled = features_scaled.reshape(1, time_steps, n_features)
        
        # 予測
        prediction = self.model.predict(features_scaled)
        self.confidence_scores = prediction[0]
        
        # 最も確率の高いクラスのインデックス
        predicted_index = np.argmax(self.confidence_scores)
        
        # インデックスから感情ラベルへ変換
        predicted_emotion = self.emotion_labels[predicted_index]
        
        self.latest_prediction = predicted_emotion
        return predicted_emotion
    
    def predict_svm(self, features):
        """
        SVMモデルによる感情予測（代替モデル）
        
        Parameters:
        -----------
        features : numpy.ndarray
            特徴量データ
            
        Returns:
        --------
        str
            予測された感情ラベル
        """
        if self.svm_model is None:
            print("SVMモデルが構築されていません。")
            return None
        
        # 特徴量の前処理
        features_reshaped = features.reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features_reshaped)
        
        # 予測
        predicted_index = self.svm_model.predict(features_scaled)[0]
        self.confidence_scores = self.svm_model.predict_proba(features_scaled)[0]
        
        # インデックスから感情ラベルへ変換
        predicted_emotion = self.emotion_labels[predicted_index]
        
        self.latest_prediction = predicted_emotion
        return predicted_emotion
    
    def get_confidence_scores(self):
        """
        確信度スコアの取得
        
        Returns:
        --------
        dict
            感情ラベルと確信度のマッピング
        """
        if self.confidence_scores is None:
            print("予測が行われていません。")
            return None
        
        # 感情ラベルと確信度のマッピング
        confidence_dict = {}
        for i, label in enumerate(self.emotion_labels):
            confidence_dict[label] = float(self.confidence_scores[i])
        
        # 確信度の降順でソート
        sorted_confidence = {
            k: v for k, v in sorted(
                confidence_dict.items(), 
                key=lambda item: item[1], 
                reverse=True
            )
        }
        
        return sorted_confidence
    
    def get_explanation(self):
        """
        予測根拠の生成
        
        Returns:
        --------
        str
            予測根拠の説明文
        """
        if self.latest_prediction is None or self.confidence_scores is None:
            return "まだ予測が行われていません。"
        
        # 確信度スコアの取得
        confidence_dict = self.get_confidence_scores()
        top_emotion = self.latest_prediction
        top_confidence = confidence_dict[top_emotion]
        
        # 上位3つの感情
        top3_emotions = list(confidence_dict.keys())[:3]
        top3_confidences = [confidence_dict[e] for e in top3_emotions]
        
        # 確信度の強さに基づくメッセージ
        if top_confidence > 0.8:
            confidence_msg = f"非常に高い確信度（{top_confidence:.1%}）で"
        elif top_confidence > 0.6:
            confidence_msg = f"高い確信度（{top_confidence:.1%}）で"
        elif top_confidence > 0.4:
            confidence_msg = f"中程度の確信度（{top_confidence:.1%}）で"
        else:
            confidence_msg = f"低い確信度（{top_confidence:.1%}）で"
        
        # 2位との差に基づくメッセージ
        if len(top3_emotions) > 1:
            diff = top_confidence - top3_confidences[1]
            if diff > 0.5:
                diff_msg = f"他の感情（{top3_emotions[1]}：{top3_confidences[1]:.1%}）と比べて大きな差があります。"
            elif diff > 0.2:
                diff_msg = f"他の感情（{top3_emotions[1]}：{top3_confidences[1]:.1%}）と比べて明確な差があります。"
            elif diff > 0.1:
                diff_msg = f"他の感情（{top3_emotions[1]}：{top3_confidences[1]:.1%}）との差は小さめです。"
            else:
                diff_msg = f"他の感情（{top3_emotions[1]}：{top3_confidences[1]:.1%}）との区別が難しい微妙な表現です。"
        else:
            diff_msg = ""
        
        # 総合的な説明文
        explanation = f"あなたの声は「{top_emotion}」の感情と{confidence_msg}判断されました。{diff_msg}"
        
        # 上位3つの感情を追加
        if len(top3_emotions) > 1:
            explanation += f"\n\n検出された感情（上位3つ）:\n"
            for i, (emotion, conf) in enumerate(zip(top3_emotions, top3_confidences)):
                explanation += f"{i+1}. {emotion}: {conf:.1%}\n"
        
        return explanation
    
    def visualize_results(self, results, filename=None):
        """
        評価結果の可視化
        
        Parameters:
        -----------
        results : dict
            evaluate()またはevaluate_svm()の結果
        filename : str or None
            保存するファイル名（Noneの場合は表示のみ）
        """
        if results is None:
            print("可視化する結果がありません。")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 混同行列
        plt.subplot(2, 1, 1)
        cm = results['confusion_matrix']
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels
        )
        plt.xlabel('予測')
        plt.ylabel('実際')
        plt.title('混同行列')
        
        # 分類レポート
        plt.subplot(2, 1, 2)
        report = results['classification_report']
        
        # 各クラスのF1スコア
        classes = list(self.emotion_labels)
        f1_scores = [report[c]['f1-score'] for c in classes]
        
        sns.barplot(x=classes, y=f1_scores)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('感情')
        plt.ylabel('F1スコア')
        plt.title(f"感情別F1スコア（全体精度: {results['accuracy']:.2%}）")
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"評価結果の可視化を {filename} に保存しました")
        else:
            plt.show()
    
    def save_model(self, model_path='emotion_model.h5', scaler_path='feature_scaler.pkl'):
        """
        モデルの保存
        
        Parameters:
        -----------
        model_path : str
            モデル保存パス
        scaler_path : str
            特徴量スケーラー保存パス
        """
        if self.model is None:
            print("保存するモデルがありません。")
            return False
        
        try:
            # モデルの保存
            self.model.save(model_path)
            print(f"モデルを {model_path} に保存しました")
            
            # 特徴量スケーラーの保存
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            print(f"特徴量スケーラーを {scaler_path} に保存しました")
            
            return True
        except Exception as e:
            print(f"モデルの保存に失敗しました: {e}")
            return False
    
    def save_svm_model(self, model_path='svm_model.pkl', scaler_path='feature_scaler.pkl'):
        """
        SVMモデルの保存（代替モデル）
        
        Parameters:
        -----------
        model_path : str
            モデル保存パス
        scaler_path : str
            特徴量スケーラー保存パス
        """
        if self.svm_model is None:
            print("保存するSVMモデルがありません。")
            return False
        
        try:
            # SVMモデルの保存
            with open(model_path, 'wb') as f:
                pickle.dump(self.svm_model, f)
            print(f"SVMモデルを {model_path} に保存しました")
            
            # 特徴量スケーラーの保存
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            print(f"特徴量スケーラーを {scaler_path} に保存しました")
            
            return True
        except Exception as e:
            print(f"SVMモデルの保存に失敗しました: {e}")
            return False
    
    def load_model(self, model_path='emotion_model.h5', scaler_path='feature_scaler.pkl'):
        """
        モデルの読み込み
        
        Parameters:
        -----------
        model_path : str
            モデル保存パス
        scaler_path : str
            特徴量スケーラー保存パス
            
        Returns:
        --------
        bool
            読み込み成功かどうか
        """
        try:
            # モデルの読み込み
            self.model = load_model(model_path)
            print(f"モデルを {model_path} から読み込みました")
            
            # 特徴量スケーラーの読み込み
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"特徴量スケーラーを {scaler_path} から読み込みました")
            
            return True
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            return False
    
    def load_svm_model(self, model_path='svm_model.pkl', scaler_path='feature_scaler.pkl'):
        """
        SVMモデルの読み込み（代替モデル）
        
        Parameters:
        -----------
        model_path : str
            モデル保存パス
        scaler_path : str
            特徴量スケーラー保存パス
            
        Returns:
        --------
        bool
            読み込み成功かどうか
        """
        try:
            # SVMモデルの読み込み
            with open(model_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            print(f"SVMモデルを {model_path} から読み込みました")
            
            # 特徴量スケーラーの読み込み
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"特徴量スケーラーを {scaler_path} から読み込みました")
            
            return True
        except Exception as e:
            print(f"SVMモデルの読み込みに失敗しました: {e}")
            return False


def create_dummy_data(num_samples=100, feature_dim=78, num_classes=13):
    """
    テスト用のダミーデータを生成
    
    Parameters:
    -----------
    num_samples : int
        サンプル数
    feature_dim : int
        特徴量の次元数
    num_classes : int
        クラス数
        
    Returns:
    --------
    tuple
        (features, labels)
    """
    # 感情ラベル
    emotion_labels = [
        "喜び", "怒り", "悲しみ", "驚き", "不安", "嫌悪", 
        "困惑", "照れ", "呆れ", "疑い", "感動", "軽蔑", "中立"
    ]
    
    # 特徴量の生成
    features = np.random.randn(num_samples, feature_dim)
    
    # ラベルの生成
    label_indices = np.random.randint(0, num_classes, num_samples)
    labels = [emotion_labels[i] for i in label_indices]
    
    return features, labels


def test_emotion_recognizer():
    """感情認識モデルのテスト"""
    print("感情認識モデルのテスト")
    print("-----------------")
    
    # ダミーデータの生成
    print("ダミーデータの生成...")
    features, labels = create_dummy_data(num_samples=500)
    
    # 感情認識モデルの初期化
    recognizer = EmotionRecognizer()
    
    # データの準備
    print("データの準備...")
    X_train, X_val, X_test, y_train, y_val, y_test = recognizer.prepare_data(
        features, labels, test_size=0.2, validation_split=0.2
    )
    
    # モデルの構築
    print("モデルの構築...")
    input_shape = (X_train.shape[1],)  # 1次元特徴量の場合
    recognizer.build_model(input_shape, model_type='dense')  # シンプルなDNNモデル
    
    # モデルの学習
    print("モデルの学習...")
    history = recognizer.train(
        X_train, y_train, X_val, y_val,
        epochs=10,  # テスト用に少ないエポック数
        batch_size=32,
        model_path='test_emotion_model.h5'
    )
    
    # モデルの評価
    print("モデルの評価...")
    results = recognizer.evaluate(X_test, y_test)
    
    # 評価結果の表示
    print(f"テスト精度: {results['accuracy']:.2%}")
    
    # 評価結果の可視化
    recognizer.visualize_results(results, filename='test_evaluation_results.png')
    
    # 予測テスト
    print("予測テスト...")
    test_features = X_test[0]  # テスト用の特徴量
    predicted_emotion = recognizer.predict(test_features)
    print(f"予測された感情: {predicted_emotion}")
    
    # 確信度スコアの取得
    confidence_scores = recognizer.get_confidence_scores()
    print("確信度スコア:")
    for emotion, score in list(confidence_scores.items())[:3]:  # 上位3つ
        print(f"  {emotion}: {score:.2%}")
    
    # 予測根拠の生成
    explanation = recognizer.get_explanation()
    print("予測根拠:")
    print(explanation)
    
    # モデルの保存
    recognizer.save_model(
        model_path='test_emotion_model.h5',
        scaler_path='test_feature_scaler.pkl'
    )
    
    # SVMモデルのテスト
    print("\nSVMモデルのテスト...")
    recognizer.build_svm_model()
    recognizer.train_svm(X_train, y_train)
    
    # SVMモデルの評価
    svm_results = recognizer.evaluate_svm(X_test, y_test)
    print(f"SVM精度: {svm_results['accuracy']:.2%}")
    
    # SVMモデルの保存
    recognizer.save_svm_model(
        model_path='test_svm_model.pkl',
        scaler_path='test_feature_scaler.pkl'
    )


if __name__ == "__main__":
    test_emotion_recognizer()
