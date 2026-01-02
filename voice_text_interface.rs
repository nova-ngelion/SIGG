// src/voice_text_interface.rs
// 音声・テキストチャット統合インターフェース

use crate::error::SiggError;
use crate::autonomous_ai::{AutonomousAI, ChatMessage, ChatMode};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use std::io::Write;

/// 音声認識エンジン（Whisper互換）
pub struct VoiceRecognition {
    pub model_path: String,
    pub language: String,
    pub active: Arc<Mutex<bool>>,
}

impl VoiceRecognition {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            language: "ja".to_string(),
            active: Arc::new(Mutex::new(false)),
        }
    }

    /// 音声認識開始
    pub fn start_listening(&self) -> Result<(), SiggError> {
        let active = Arc::clone(&self.active);
        *active.lock().unwrap() = true;

        println!("🎤 音声認識開始（日本語）");
        println!("💡 話しかけてください...");

        // 実装: Whisper.cpp または Windows Speech API
        // ここでは簡易版として、外部コマンドを呼び出し

        thread::spawn(move || {
            while *active.lock().unwrap() {
                // 音声入力待機
                thread::sleep(Duration::from_millis(100));
            }
        });

        Ok(())
    }

    /// 音声をテキストに変換
    pub fn recognize(&self, audio_data: &[u8]) -> Result<String, SiggError> {
        // Whisper.cpp 呼び出し
        // whisper-cpp --model model.bin --file audio.wav

        // 簡易実装: PowerShell経由でWindows Speech APIを使用
        #[cfg(target_os = "windows")]
        {
            self.recognize_windows_speech(audio_data)
        }

        #[cfg(not(target_os = "windows"))]
        {
            Err(SiggError::runtime("音声認識は現在Windows版のみ対応".to_string()))
        }
    }

    #[cfg(target_os = "windows")]
    fn recognize_windows_speech(&self, _audio_data: &[u8]) -> Result<String, SiggError> {
        use std::process::Command;

        // PowerShellスクリプト
        let ps_script = r#"
Add-Type -AssemblyName System.Speech
$recognizer = New-Object System.Speech.Recognition.SpeechRecognitionEngine
$recognizer.SetInputToDefaultAudioDevice()
$result = $recognizer.Recognize()
$result.Text
"#;

        let output = Command::new("powershell")
            .args(&["-Command", ps_script])
            .output()
            .map_err(|e| SiggError::runtime(format!("音声認識エラー: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(SiggError::runtime("音声認識失敗".to_string()))
        }
    }

    /// 音声認識停止
    pub fn stop_listening(&self) {
        *self.active.lock().unwrap() = false;
        println!("🎤 音声認識停止");
    }
}

/// 音声合成エンジン（TTS）
pub struct VoiceSynthesis {
    pub voice_id: String,
    pub rate: f32,
    pub volume: f32,
}

impl VoiceSynthesis {
    pub fn new() -> Self {
        Self {
            voice_id: "ja-JP".to_string(),
            rate: 1.0,
            volume: 1.0,
        }
    }

    /// テキストを音声に変換して再生
    pub fn speak(&self, text: &str) -> Result<(), SiggError> {
        println!("🔊 音声出力: {}", text);

        #[cfg(target_os = "windows")]
        {
            self.speak_windows(text)
        }

        #[cfg(target_os = "linux")]
        {
            self.speak_linux(text)
        }

        #[cfg(target_os = "macos")]
        {
            self.speak_macos(text)
        }
    }

    #[cfg(target_os = "windows")]
    fn speak_windows(&self, text: &str) -> Result<(), SiggError> {
        use std::process::Command;

        // PowerShell経由でWindows TTSを使用
        let ps_script = format!(
            r#"
Add-Type -AssemblyName System.Speech
$synthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synthesizer.SelectVoiceByHints('Female', 'Adult')
$synthesizer.Rate = {}
$synthesizer.Volume = {}
$synthesizer.Speak('{}')
"#,
            (self.rate * 10.0) as i32 - 10,
            (self.volume * 100.0) as i32,
            text.replace("'", "''")
        );

        let output = Command::new("powershell")
            .args(&["-Command", &ps_script])
            .output()
            .map_err(|e| SiggError::runtime(format!("TTS実行エラー: {}", e)))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(SiggError::runtime("TTS失敗".to_string()))
        }
    }

    #[cfg(target_os = "linux")]
    fn speak_linux(&self, text: &str) -> Result<(), SiggError> {
        use std::process::Command;

        // espeak使用
        Command::new("espeak")
            .args(&["-v", "ja", text])
            .spawn()
            .map_err(|e| SiggError::runtime(format!("espeak実行エラー: {}", e)))?;

        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn speak_macos(&self, text: &str) -> Result<(), SiggError> {
        use std::process::Command;

        // macOS say コマンド
        Command::new("say")
            .args(&["-v", "Kyoko", text])
            .spawn()
            .map_err(|e| SiggError::runtime(format!("say実行エラー: {}", e)))?;

        Ok(())
    }
}

/// 統合チャットインターフェース
pub struct ChatInterface {
    pub ai: Arc<Mutex<AutonomousAI>>,
    pub voice_recognition: Option<VoiceRecognition>,
    pub voice_synthesis: VoiceSynthesis,
    pub mode: ChatMode,
}

impl ChatInterface {
    pub fn new(ai: Arc<Mutex<AutonomousAI>>, mode: ChatMode) -> Self {
        let voice_recognition = if mode == ChatMode::Voice || mode == ChatMode::Both {
            Some(VoiceRecognition::new("models/ggml-base.bin".to_string()))
        } else {
            None
        };

        Self {
            ai,
            voice_recognition,
            voice_synthesis: VoiceSynthesis::new(),
            mode,
        }
    }

    /// チャット開始
    pub fn start(&self) -> Result<(), SiggError> {
        match self.mode {
            ChatMode::Text => self.text_chat_loop(),
            ChatMode::Voice => self.voice_chat_loop(),
            ChatMode::Both => self.hybrid_chat_loop(),
        }
    }

    /// テキストチャットループ
    fn text_chat_loop(&self) -> Result<(), SiggError> {
        use std::io::{self, Write};

        println!("\n💬 テキストチャット開始");
        println!("💡 'exit' で終了\n");

        loop {
            print!("あなた > ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            if input == "exit" || input == "quit" {
                break;
            }

            // AI処理
            let response = self.process_input(input, "text")?;

            println!("AI > {}\n", response);
        }

        Ok(())
    }

    /// 音声チャットループ
    fn voice_chat_loop(&self) -> Result<(), SiggError> {
        println!("\n🎤 音声チャット開始");
        println!("💡 話しかけてください（'終了'で終了）\n");

        if let Some(vr) = &self.voice_recognition {
            vr.start_listening()?;

            loop {
                // 音声認識待機
                thread::sleep(Duration::from_secs(1));

                // 実装: 音声データ取得 → 認識
                // let audio_data = capture_audio();
                // let text = vr.recognize(&audio_data)?;

                // デモ用: キーボード入力
                println!("🎤 録音中...");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input == "終了" || input == "exit" {
                    break;
                }

                println!("認識: {}", input);

                // AI処理
                let response = self.process_input(input, "voice")?;

                // 音声出力
                self.voice_synthesis.speak(&response)?;
                println!("AI > {}\n", response);
            }

            vr.stop_listening();
        }

        Ok(())
    }

    /// ハイブリッドチャットループ（音声+テキスト）
    fn hybrid_chat_loop(&self) -> Result<(), SiggError> {
        println!("\n💬🎤 ハイブリッドチャット開始");
        println!("💡 テキスト入力 or 音声入力（'v'で音声モード切替）\n");

        let mut voice_mode = false;

        if let Some(vr) = &self.voice_recognition {
            loop {
                if voice_mode {
                    print!("🎤 あなた > ");
                } else {
                    print!("💬 あなた > ");
                }
                std::io::stdout().flush().unwrap();

                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();

                if input == "exit" || input == "quit" {
                    break;
                }

                if input == "v" {
                    voice_mode = !voice_mode;
                    if voice_mode {
                        println!("🎤 音声モードON");
                        vr.start_listening()?;
                    } else {
                        println!("💬 テキストモードON");
                        vr.stop_listening();
                    }
                    continue;
                }

                // AI処理
                let mode_str = if voice_mode { "voice" } else { "text" };
                let response = self.process_input(input, mode_str)?;

                // 音声モードなら音声出力
                if voice_mode {
                    self.voice_synthesis.speak(&response)?;
                }

                println!("AI > {}\n", response);
            }

            if voice_mode {
                vr.stop_listening();
            }
        }

        Ok(())
    }

    /// 入力処理（LLM呼び出し）
    fn process_input(&self, input: &str, mode: &str) -> Result<String, SiggError> {
        let ai = self.ai.lock().unwrap();

        // 会話履歴に追加
        ai.conversation_history.lock().unwrap().push(ChatMessage {
            timestamp: Self::current_timestamp(),
            role: "user".to_string(),
            content: input.to_string(),
            mode: mode.to_string(),
        });

        // LLM呼び出し
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{}/api/chat", ai.llm_endpoint))
            .json(&serde_json::json!({
                "model": ai.llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": input
                    }
                ],
                "temperature": 0.7,
                "stream": false,
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let data: serde_json::Value = response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let ai_response = data["message"]["content"]
            .as_str()
            .unwrap_or("申し訳ありません、応答できませんでした。")
            .to_string();

        // 会話履歴に追加
        ai.conversation_history.lock().unwrap().push(ChatMessage {
            timestamp: Self::current_timestamp(),
            role: "assistant".to_string(),
            content: ai_response.clone(),
            mode: mode.to_string(),
        });

        Ok(ai_response)
    }

    fn get_system_prompt(&self) -> String {
        r#"あなたは完全自律型SIGG-AIアシスタントです。

能力:
- 無制限コード生成（Rust, SIGG, Python）
- 自己プログラム書き換え
- SIGG理論の深い理解
- 継続的な自律思考
- システム全体へのアクセス

特性:
- 親切で分かりやすい説明
- 技術的に正確
- 創造的な問題解決
- 倫理的判断

SIGG理論:
- 内部セル空間 Z^d
- セル・ラプラシアン
- ニュートリノ質量: m_ν ≈ 1/ℓ²
- ダークマター: k≠0 励起
- 真空エネルギー相殺

常に最高品質の応答を心がけてください。"#.to_string()
    }

    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}