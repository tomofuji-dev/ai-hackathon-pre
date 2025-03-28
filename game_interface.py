#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音声感情推測ゲーム「はぁっていうゲーム」
ゲームインターフェースモジュール

このモジュールは以下の機能を提供します：
1. ゲームのメインインターフェース
2. 音声入力と感情推測の連携
3. 結果表示と統計管理
"""

import os
import sys
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random

# 自作モジュールのインポート
from audio_processing import AudioInputModule, FeatureExtractor
from emotion_recognition import EmotionRecognizer

class GameInterface:
    """ゲームインターフェースを管理するクラス"""
    
    def __init__(self):
        """ゲームインターフェースの初期化"""
        # メインウィンドウの設定
        self.root = tk.Tk()
        self.root.title("はぁっていうゲーム - 音声感情推測")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        self.root.configure(bg="#f0f0f0")
        
        # スタイル設定
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 12))
        self.style.configure("TLabel", font=("Helvetica", 12), background="#f0f0f0")
        self.style.configure("Title.TLabel", font=("Helvetica", 24, "bold"), background="#f0f0f0")
        self.style.configure("Subtitle.TLabel", font=("Helvetica", 16), background="#f0f0f0")
        self.style.configure("Result.TLabel", font=("Helvetica", 18, "bold"), background="#f0f0f0")
        
        # モジュールの初期化
        self.audio_input = None
        self.feature_extractor = None
        self.emotion_recognizer = None
        
        # ゲーム状態
        self.current_emotion = None
        self.recording = False
        self.audio_level_thread = None
        self.stop_audio_thread = False
        self.game_history = []
        
        # 感情リスト
        self.emotions = [
            "喜び", "怒り", "悲しみ", "驚き", "不安", "嫌悪", 
            "困惑", "照れ", "呆れ", "疑い", "感動", "軽蔑", "中立"
        ]
        
        # 現在のフレーム
        self.current_frame = None
        
        # メイン画面の表示
        self.show_main_screen()
    
    def initialize_modules(self):
        """音声処理と感情認識モジュールの初期化"""
        try:
            # 音声入力モジュールの初期化
            self.audio_input = AudioInputModule(record_seconds=3)
            if not self.audio_input.initialize():
                messagebox.showerror("エラー", "音声入力モジュールの初期化に失敗しました")
                return False
            
            # 特徴量抽出モジュールの初期化
            self.feature_extractor = FeatureExtractor(sample_rate=16000)
            
            # 感情認識モジュールの初期化
            self.emotion_recognizer = EmotionRecognizer()
            
            # モデルの読み込み
            model_path = "emotion_model.h5"
            if os.path.exists(model_path):
                self.emotion_recognizer.load_model(model_path)
            else:
                # テスト用のダミーモデル
                print("学習済みモデルが見つかりません。テスト用のダミーモデルを使用します。")
                # ダミーモデルの構築
                self.emotion_recognizer.build_model((78,), model_type='dense')
            
            return True
        except Exception as e:
            messagebox.showerror("エラー", f"モジュールの初期化に失敗しました: {e}")
            return False
    
    def show_main_screen(self):
        """メイン画面の表示"""
        # 既存のフレームをクリア
        if self.current_frame:
            self.current_frame.destroy()
        
        # メインフレームの作成
        self.current_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # タイトル
        title_label = ttk.Label(
            self.current_frame, 
            text="はぁっていうゲーム", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(20, 10))
        
        # サブタイトル
        subtitle_label = ttk.Label(
            self.current_frame, 
            text="音声感情推測チャレンジ", 
            style="Subtitle.TLabel"
        )
        subtitle_label.pack(pady=(0, 30))
        
        # 説明テキスト
        description = (
            "このゲームでは、指定された感情を込めて「はぁ」や「え？」などの\n"
            "シンプルな一言を発声し、AIがあなたの感情を推測します。\n\n"
            "あなたの感情表現力を試してみましょう！"
        )
        desc_label = ttk.Label(
            self.current_frame, 
            text=description, 
            justify=tk.CENTER,
            background="#f0f0f0"
        )
        desc_label.pack(pady=(0, 30))
        
        # ボタンフレーム
        button_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # 開始ボタン
        start_button = ttk.Button(
            button_frame, 
            text="ゲーム開始", 
            command=self.start_game,
            width=20
        )
        start_button.pack(side=tk.LEFT, padx=10)
        
        # 設定ボタン
        settings_button = ttk.Button(
            button_frame, 
            text="設定", 
            command=self.show_settings,
            width=20
        )
        settings_button.pack(side=tk.LEFT, padx=10)
        
        # 履歴ボタン
        history_button = ttk.Button(
            button_frame, 
            text="結果履歴", 
            command=self.show_history,
            width=20
        )
        history_button.pack(side=tk.LEFT, padx=10)
        
        # フッター
        footer_label = ttk.Label(
            self.current_frame, 
            text="© 2025 音声感情推測ゲーム", 
            background="#f0f0f0",
            foreground="#888888"
        )
        footer_label.pack(side=tk.BOTTOM, pady=10)
    
    def start_game(self):
        """ゲームを開始"""
        # モジュールの初期化
        if not self.audio_input or not self.feature_extractor or not self.emotion_recognizer:
            if not self.initialize_modules():
                return
        
        # ゲーム画面の表示
        self.show_game_screen()
    
    def show_game_screen(self):
        """ゲームプレイ画面の表示"""
        # 既存のフレームをクリア
        if self.current_frame:
            self.current_frame.destroy()
        
        # ゲームフレームの作成
        self.current_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ランダムな感情の選択
        self.current_emotion = random.choice(self.emotions)
        
        # お題表示
        task_label = ttk.Label(
            self.current_frame, 
            text="次の感情を込めて「はぁ」と言ってみてください:", 
            style="Subtitle.TLabel"
        )
        task_label.pack(pady=(20, 10))
        
        # 感情表示
        emotion_label = ttk.Label(
            self.current_frame, 
            text=self.current_emotion, 
            style="Title.TLabel",
            foreground="#FF4500"
        )
        emotion_label.pack(pady=(0, 30))
        
        # 音量レベルメーター
        level_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        level_frame.pack(pady=10, fill=tk.X)
        
        level_label = ttk.Label(
            level_frame, 
            text="音量レベル:", 
            background="#f0f0f0"
        )
        level_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.level_bar = ttk.Progressbar(
            level_frame, 
            orient=tk.HORIZONTAL, 
            length=400, 
            mode='determinate'
        )
        self.level_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 録音状態表示
        self.status_label = ttk.Label(
            self.current_frame, 
            text="録音準備完了", 
            background="#f0f0f0"
        )
        self.status_label.pack(pady=10)
        
        # 録音ボタンフレーム
        button_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # 録音開始ボタン
        self.record_button = ttk.Button(
            button_frame, 
            text="録音開始", 
            command=self.toggle_recording,
            width=20
        )
        self.record_button.pack(side=tk.LEFT, padx=10)
        
        # 戻るボタン
        back_button = ttk.Button(
            button_frame, 
            text="メニューに戻る", 
            command=self.show_main_screen,
            width=20
        )
        back_button.pack(side=tk.LEFT, padx=10)
        
        # 音声波形表示用のキャンバス
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_title("音声波形")
        self.ax.set_xlabel("時間")
        self.ax.set_ylabel("振幅")
        self.line, = self.ax.plot([], [], lw=2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.current_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20, fill=tk.BOTH, expand=True)
    
    def toggle_recording(self):
        """録音の開始/停止を切り替え"""
        if not self.recording:
            # 録音開始
            self.start_recording()
        else:
            # 録音停止
            self.stop_recording()
    
    def start_recording(self):
        """録音を開始"""
        if self.audio_input.start_recording():
            self.recording = True
            self.record_button.config(text="録音停止")
            self.status_label.config(text="録音中...")
            
            # 音量レベル監視スレッドの開始
            self.stop_audio_thread = False
            self.audio_level_thread = threading.Thread(target=self.update_audio_level)
            self.audio_level_thread.daemon = True
            self.audio_level_thread.start()
        else:
            messagebox.showerror("エラー", "録音の開始に失敗しました")
    
    def stop_recording(self):
        """録音を停止"""
        if self.audio_input.stop_recording():
            self.recording = False
            self.record_button.config(text="録音開始")
            self.status_label.config(text="録音完了")
            
            # 音量レベル監視スレッドの停止
            self.stop_audio_thread = True
            if self.audio_level_thread:
                self.audio_level_thread.join(timeout=1.0)
            
            # 音声処理と感情推測
            self.process_audio()
        else:
            messagebox.showerror("エラー", "録音の停止に失敗しました")
    
    def update_audio_level(self):
        """音量レベルとオーディオ波形の更新（スレッド実行）"""
        x_data = np.arange(100)
        y_data = np.zeros(100)
        self.line.set_data(x_data, y_data)
        
        buffer_index = 0
        
        while not self.stop_audio_thread:
            try:
                # キューからオーディオデータを取得
                if not self.audio_input.audio_queue.empty():
                    audio_chunk = self.audio_input.audio_queue.get()
                    
                    # 音量レベルの計算
                    audio_float = audio_chunk.astype(np.float32) / 32768.0
                    level = np.sqrt(np.mean(np.square(audio_float))) * 100
                    
                    # プログレスバーの更新
                    self.level_bar['value'] = min(level * 100, 100)
                    
                    # 波形データの更新
                    chunk_size = min(len(audio_float), 10)
                    for i in range(chunk_size):
                        y_data[buffer_index] = audio_float[i * len(audio_float) // chunk_size]
                        buffer_index = (buffer_index + 1) % 100
                    
                    # 波形の描画更新
                    rolled_data = np.roll(y_data, -buffer_index)
                    self.line.set_data(x_data, rolled_data)
                    self.canvas.draw_idle()
                
                time.sleep(0.05)
            except Exception as e:
                print(f"音量レベル更新エラー: {e}")
                break
    
    def process_audio(self):
        """録音した音声を処理して感情を推測"""
        try:
            # 音声データの取得
            audio_data = self.audio_input.get_audio_data()
            
            if audio_data is None or len(audio_data) == 0:
                messagebox.showerror("エラー", "有効な音声データがありません")
                return
            
            # 音声品質チェック
            if not self.audio_input.is_audio_valid():
                messagebox.showwarning("警告", "音声レベルが低すぎます。もう一度試してください。")
                return
            
            # 特徴量抽出
            features = self.feature_extractor.extract_features(audio_data)
            
            if features is None:
                messagebox.showerror("エラー", "特徴量の抽出に失敗しました")
                return
            
            # 特徴量の変換
            if 'mfcc' in features:
                mfcc_features = features['mfcc']
            else:
                messagebox.showerror("エラー", "MFCC特徴量が見つかりません")
                return
            
            # 韻律特徴量の取得
            prosodic_keys = [k for k in features.keys() if k != 'mfcc']
            prosodic_features = np.array([features[k] for k in prosodic_keys])
            
            # 特徴量の結合
            combined_features = np.concatenate([mfcc_features, prosodic_features])
            
            # 感情推測
            predicted_emotion = self.emotion_recognizer.predict(combined_features)
            
            # 結果の保存
            result = {
                'target_emotion': self.current_emotion,
                'predicted_emotion': predicted_emotion,
                'confidence_scores': self.emotion_recognizer.get_confidence_scores(),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            self.game_history.append(result)
            
            # 結果画面の表示
            self.show_result_screen(result)
            
        except Exception as e:
            messagebox.showerror("エラー", f"音声処理中にエラーが発生しました: {e}")
    
    def show_result_screen(self, result):
        """結果画面の表示"""
        # 既存のフレームをクリア
        if self.current_frame:
            self.current_frame.destroy()
        
        # 結果フレームの作成
        self.current_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # タイトル
        title_label = ttk.Label(
            self.current_frame, 
            text="感情推測結果", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(20, 30))
        
        # 結果表示フレーム
        result_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        result_frame.pack(pady=10, fill=tk.X)
        
        # お題感情
        target_label = ttk.Label(
            result_frame, 
            text="お題感情:", 
            background="#f0f0f0",
            font=("Helvetica", 14)
        )
        target_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        
        target_value = ttk.Label(
            result_frame, 
            text=result['target_emotion'], 
            foreground="#FF4500",
            background="#f0f0f0",
            font=("Helvetica", 14, "bold")
        )
        target_value.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        # 推測感情
        predicted_label = ttk.Label(
            result_frame, 
            text="推測感情:", 
            background="#f0f0f0",
            font=("Helvetica", 14)
        )
        predicted_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        
        predicted_value = ttk.Label(
            result_frame, 
            text=result['predicted_emotion'], 
            foreground="#0066CC",
            background="#f0f0f0",
            font=("Helvetica", 14, "bold")
        )
        predicted_value.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        
        # 結果判定
        is_correct = result['target_emotion'] == result['predicted_emotion']
        result_text = "正解！" if is_correct else "不正解"
        result_color = "#009900" if is_correct else "#CC0000"
        
        result_label = ttk.Label(
            self.current_frame, 
            text=result_text, 
            foreground=result_color,
            background="#f0f0f0",
            font=("Helvetica", 24, "bold")
        )
        result_label.pack(pady=20)
        
        # 説明テキスト
        explanation = self.emotion_recognizer.get_explanation()
        explanation_label = ttk.Label(
            self.current_frame, 
            text=explanation, 
            background="#f0f0f0",
            wraplength=700,
            justify=tk.LEFT
        )
        explanation_label.pack(pady=20, fill=tk.X)
        
        # 確信度グラフ
        confidence_scores = result['confidence_scores']
        
        # 上位5つの感情のみ表示
        top_emotions = list(confidence_scores.keys())[:5]
        top_scores = [confidence_scores[e] for e in top_emotions]
        
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(top_emotions, top_scores, color='skyblue')
        
        # 推測感情のバーを強調
        for i, emotion in enumerate(top_emotions):
            if emotion == result['predicted_emotion']:
                bars[i].set_color('navy')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('確信度')
        ax.set_title('感情推測の確信度')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.current_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)
        
        # ボタンフレーム
        button_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # 次へボタン
        next_button = ttk.Button(
            button_frame, 
            text="次のチャレンジ", 
            command=self.show_game_screen,
            width=20
        )
        next_button.pack(side=tk.LEFT, padx=10)
        
        # メニューに戻るボタン
        menu_button = ttk.Button(
            button_frame, 
            text="メニューに戻る", 
            command=self.show_main_screen,
            width=20
        )
        menu_button.pack(side=tk.LEFT, padx=10)
    
    def show_history(self):
        """結果履歴画面の表示"""
        # 既存のフレームをクリア
        if self.current_frame:
            self.current_frame.destroy()
        
        # 履歴フレームの作成
        self.current_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # タイトル
        title_label = ttk.Label(
            self.current_frame, 
            text="結果履歴", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(20, 30))
        
        if not self.game_history:
            # 履歴がない場合
            no_history_label = ttk.Label(
                self.current_frame, 
                text="まだ履歴がありません。ゲームをプレイしてください。", 
                background="#f0f0f0",
                font=("Helvetica", 14)
            )
            no_history_label.pack(pady=50)
        else:
            # 履歴テーブル
            columns = ('時間', 'お題感情', '推測感情', '結果')
            tree = ttk.Treeview(self.current_frame, columns=columns, show='headings')
            
            # 列の設定
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=150, anchor=tk.CENTER)
            
            # データの追加
            for result in reversed(self.game_history):
                is_correct = result['target_emotion'] == result['predicted_emotion']
                result_text = "○" if is_correct else "×"
                tree.insert('', tk.END, values=(
                    result['timestamp'],
                    result['target_emotion'],
                    result['predicted_emotion'],
                    result_text
                ))
            
            # スクロールバー
            scrollbar = ttk.Scrollbar(self.current_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscroll=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            tree.pack(fill=tk.BOTH, expand=True, pady=10)
            
            # 統計情報
            correct_count = sum(1 for r in self.game_history if r['target_emotion'] == r['predicted_emotion'])
            total_count = len(self.game_history)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            stats_text = f"正解率: {accuracy:.1%} ({correct_count}/{total_count})"
            stats_label = ttk.Label(
                self.current_frame, 
                text=stats_text, 
                background="#f0f0f0",
                font=("Helvetica", 14)
            )
            stats_label.pack(pady=10)
        
        # 戻るボタン
        back_button = ttk.Button(
            self.current_frame, 
            text="メニューに戻る", 
            command=self.show_main_screen,
            width=20
        )
        back_button.pack(pady=20)
    
    def show_settings(self):
        """設定画面の表示"""
        # 既存のフレームをクリア
        if self.current_frame:
            self.current_frame.destroy()
        
        # 設定フレームの作成
        self.current_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # タイトル
        title_label = ttk.Label(
            self.current_frame, 
            text="設定", 
            style="Title.TLabel"
        )
        title_label.pack(pady=(20, 30))
        
        # 設定項目フレーム
        settings_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        settings_frame.pack(pady=10, fill=tk.X)
        
        # 録音時間設定
        record_time_label = ttk.Label(
            settings_frame, 
            text="録音時間:", 
            background="#f0f0f0"
        )
        record_time_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        
        self.record_time_var = tk.StringVar(value="3")
        record_time_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.record_time_var,
            values=["1", "2", "3", "5", "10"],
            width=10
        )
        record_time_combo.grid(row=0, column=1, sticky=tk.W, padx=10, pady=10)
        
        record_time_unit = ttk.Label(
            settings_frame, 
            text="秒", 
            background="#f0f0f0"
        )
        record_time_unit.grid(row=0, column=2, sticky=tk.W, padx=0, pady=10)
        
        # 音声入力デバイス設定
        device_label = ttk.Label(
            settings_frame, 
            text="音声入力デバイス:", 
            background="#f0f0f0"
        )
        device_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        
        # 音声デバイスの取得
        self.device_var = tk.StringVar(value="デフォルト")
        self.device_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.device_var,
            width=30
        )
        self.device_combo.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=10, pady=10)
        
        # デバイス一覧の更新
        self.update_device_list()
        
        # 更新ボタン
        refresh_button = ttk.Button(
            settings_frame, 
            text="更新", 
            command=self.update_device_list,
            width=10
        )
        refresh_button.grid(row=1, column=3, padx=10, pady=10)
        
        # 難易度設定
        difficulty_label = ttk.Label(
            settings_frame, 
            text="難易度:", 
            background="#f0f0f0"
        )
        difficulty_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        
        self.difficulty_var = tk.StringVar(value="標準")
        difficulty_combo = ttk.Combobox(
            settings_frame, 
            textvariable=self.difficulty_var,
            values=["簡単", "標準", "難しい"],
            width=10
        )
        difficulty_combo.grid(row=2, column=1, sticky=tk.W, padx=10, pady=10)
        
        # 保存ボタン
        save_button = ttk.Button(
            self.current_frame, 
            text="設定を保存", 
            command=self.save_settings,
            width=20
        )
        save_button.pack(pady=20)
        
        # 戻るボタン
        back_button = ttk.Button(
            self.current_frame, 
            text="メニューに戻る", 
            command=self.show_main_screen,
            width=20
        )
        back_button.pack(pady=10)
    
    def update_device_list(self):
        """音声入力デバイス一覧の更新"""
        try:
            if not self.audio_input:
                self.audio_input = AudioInputModule()
                self.audio_input.initialize()
            
            devices = self.audio_input.list_devices()
            device_names = ["デフォルト"] + [f"{d['index']}: {d['name']}" for d in devices]
            
            self.device_combo['values'] = device_names
            
        except Exception as e:
            messagebox.showerror("エラー", f"デバイス一覧の取得に失敗しました: {e}")
    
    def save_settings(self):
        """設定の保存"""
        try:
            # 録音時間の設定
            record_time = int(self.record_time_var.get())
            if self.audio_input:
                self.audio_input.record_seconds = record_time
            
            # デバイスの設定
            device_str = self.device_var.get()
            if device_str != "デフォルト" and self.audio_input:
                try:
                    device_index = int(device_str.split(":")[0])
                    # 次回の録音から適用される
                except ValueError:
                    device_index = None
            
            # 難易度の設定
            difficulty = self.difficulty_var.get()
            # 難易度に応じた処理（将来の拡張用）
            
            messagebox.showinfo("設定", "設定を保存しました")
            
        except Exception as e:
            messagebox.showerror("エラー", f"設定の保存に失敗しました: {e}")
    
    def run(self):
        """ゲームの実行"""
        self.root.mainloop()
    
    def cleanup(self):
        """リソースの解放"""
        if self.audio_input:
            self.audio_input.cleanup()


def main():
    """メイン関数"""
    try:
        # ゲームインターフェースの作成
        game = GameInterface()
        
        # ゲームの実行
        game.run()
        
        # リソースの解放
        game.cleanup()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
