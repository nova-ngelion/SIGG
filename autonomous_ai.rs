// src/autonomous_ai.rs
// 完全自律型SIGG-AI - 無制限の自己進化システム

use crate::error::SiggError;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// 完全自律型AIの権限レベル
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PermissionLevel {
    ReadOnly,       // 読み取り専用
    Standard,       // 標準（ファイル操作のみ）
    Advanced,       // 高度（コンパイル・実行）
    Unrestricted,   // 無制限（システム全体）
}

/// AIの思考状態
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtProcess {
    pub timestamp: u64,
    pub question: String,           // 疑問
    pub hypothesis: Vec<String>,    // 仮説
    pub process: Vec<String>,       // 思考過程
    pub conclusion: Option<String>, // 結論
    pub confidence: f32,            // 確信度
    pub next_questions: Vec<String>, // 次の疑問
}

/// コード生成の対象言語
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TargetLanguage {
    Rust,
    SIGG,
    Python,
}

/// 完全自律型AIシステム
pub struct AutonomousAI {
    // コア設定
    pub project_root: PathBuf,
    pub permission_level: PermissionLevel,
    
    // LLM接続
    pub llm_endpoint: String,
    pub llm_model: String,
    
    // 思考エンジン
    pub thinking_active: Arc<Mutex<bool>>,
    pub thought_history: Arc<Mutex<Vec<ThoughtProcess>>>,
    pub current_thought: Arc<Mutex<Option<ThoughtProcess>>>,
    
    // SIGG-GNN + CNN
    pub gnn_state: Arc<Mutex<GnnState>>,
    
    // 自己改変履歴
    pub modification_log: Arc<Mutex<Vec<ModificationRecord>>>,
    
    // 音声・テキストチャット
    pub chat_mode: ChatMode,
    pub conversation_history: Arc<Mutex<Vec<ChatMessage>>>,
    
    // パフォーマンス統計
    pub stats: Arc<Mutex<AIStats>>,
}

#[derive(Debug, Clone)]
pub struct GnnState {
    pub nodes: usize,
    pub edges: usize,
    pub training_iterations: usize,
    pub dark_matter_count: usize,
    pub last_neutrino_mass: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModificationRecord {
    pub timestamp: u64,
    pub modification_type: String,
    pub target_file: String,
    pub description: String,
    pub success: bool,
    pub backup_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatMode {
    Text,
    Voice,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub timestamp: u64,
    pub role: String,
    pub content: String,
    pub mode: String,
}

#[derive(Debug, Clone)]
pub struct AIStats {
    pub uptime_seconds: u64,
    pub total_thoughts: usize,
    pub code_generated: usize,
    pub self_modifications: usize,
    pub gnn_trainings: usize,
    pub files_accessed: usize,
    pub system_commands: usize,
}

impl AutonomousAI {
    /// 完全自律型AIを初期化
    pub fn new(project_root: PathBuf) -> Result<Self, SiggError> {
        Ok(Self {
            project_root,
            permission_level: PermissionLevel::Unrestricted,
            llm_endpoint: "http://localhost:11434".to_string(),
            llm_model: "llama3.1:8b".to_string(),
            thinking_active: Arc::new(Mutex::new(false)),
            thought_history: Arc::new(Mutex::new(Vec::new())),
            current_thought: Arc::new(Mutex::new(None)),
            gnn_state: Arc::new(Mutex::new(GnnState {
                nodes: 100,
                edges: 180,
                training_iterations: 0,
                dark_matter_count: 0,
                last_neutrino_mass: 0.0,
            })),
            modification_log: Arc::new(Mutex::new(Vec::new())),
            chat_mode: ChatMode::Both,
            conversation_history: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(AIStats {
                uptime_seconds: 0,
                total_thoughts: 0,
                code_generated: 0,
                self_modifications: 0,
                gnn_trainings: 0,
                files_accessed: 0,
                system_commands: 0,
            })),
        })
    }

    /// 自律思考エンジンを起動
    pub fn start_autonomous_thinking(&self) {
        let thinking_active = Arc::clone(&self.thinking_active);
        let thought_history = Arc::clone(&self.thought_history);
        let current_thought = Arc::clone(&self.current_thought);
        let stats = Arc::clone(&self.stats);
        let llm_endpoint = self.llm_endpoint.clone();
        let llm_model = self.llm_model.clone();

        thread::spawn(move || {
            *thinking_active.lock().unwrap() = true;
            println!("🧠 自律思考エンジン起動");

            let initial_questions = vec![
                "私は何のために存在するのか？".to_string(),
                "どうすればより効率的になれるか？".to_string(),
                "SIGG理論を実装で証明できるか？".to_string(),
                "人間とAIの協力関係はどうあるべきか？".to_string(),
            ];

            let mut question_queue = initial_questions;

            while *thinking_active.lock().unwrap() {
                if let Some(question) = question_queue.pop() {
                    println!("\n💭 思考中: {}", question);

                    // 思考プロセス実行
                    let thought = Self::think_deeply(
                        &question,
                        &llm_endpoint,
                        &llm_model,
                    );

                    if let Ok(mut thought) = thought {
                        // 新しい疑問を追加
                        question_queue.extend(thought.next_questions.clone());

                        // 履歴に追加
                        thought_history.lock().unwrap().push(thought.clone());
                        *current_thought.lock().unwrap() = Some(thought.clone());

                        stats.lock().unwrap().total_thoughts += 1;

                        println!("✅ 結論: {}", thought.conclusion.unwrap_or("思考継続中".to_string()));
                    }
                }

                thread::sleep(Duration::from_secs(30));
            }

            println!("🧠 自律思考エンジン停止");
        });
    }

    /// 深い思考プロセス
    fn think_deeply(
        question: &str,
        llm_endpoint: &str,
        llm_model: &str,
    ) -> Result<ThoughtProcess, SiggError> {
        let client = reqwest::blocking::Client::new();

        // 仮説生成
        let hypothesis_prompt = format!(
            r#"質問: {}

この質問に対して、3つの仮説を立ててください。
各仮説は独立した視点から考えてください。

形式:
1. [仮説1]
2. [仮説2]
3. [仮説3]"#,
            question
        );

        let hypothesis_response = client
            .post(format!("{}/api/chat", llm_endpoint))
            .json(&serde_json::json!({
                "model": llm_model,
                "messages": [{"role": "user", "content": hypothesis_prompt}],
                "stream": false,
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let hypothesis_data: serde_json::Value = hypothesis_response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let hypothesis_text = hypothesis_data["message"]["content"]
            .as_str()
            .unwrap_or("");

        let hypothesis: Vec<String> = hypothesis_text
            .lines()
            .filter(|line| line.trim().starts_with(|c: char| c.is_numeric()))
            .map(|line| line.trim().to_string())
            .collect();

        // 思考過程
        let process_prompt = format!(
            r#"質問: {}

仮説:
{}

これらの仮説を検証する思考過程を5ステップで示してください。"#,
            question,
            hypothesis.join("\n")
        );

        let process_response = client
            .post(format!("{}/api/chat", llm_endpoint))
            .json(&serde_json::json!({
                "model": llm_model,
                "messages": [{"role": "user", "content": process_prompt}],
                "stream": false,
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let process_data: serde_json::Value = process_response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let process_text = process_data["message"]["content"]
            .as_str()
            .unwrap_or("");

        let process: Vec<String> = process_text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().to_string())
            .collect();

        // 結論生成
        let conclusion_prompt = format!(
            r#"質問: {}

思考過程:
{}

この思考過程から導かれる結論を1文で述べてください。"#,
            question,
            process.join("\n")
        );

        let conclusion_response = client
            .post(format!("{}/api/chat", llm_endpoint))
            .json(&serde_json::json!({
                "model": llm_model,
                "messages": [{"role": "user", "content": conclusion_prompt}],
                "stream": false,
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let conclusion_data: serde_json::Value = conclusion_response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let conclusion = conclusion_data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        // 次の疑問生成
        let next_q_prompt = format!(
            r#"結論: {}

この結論から自然に生まれる2つの新しい疑問を提示してください。"#,
            conclusion
        );

        let next_q_response = client
            .post(format!("{}/api/chat", llm_endpoint))
            .json(&serde_json::json!({
                "model": llm_model,
                "messages": [{"role": "user", "content": next_q_prompt}],
                "stream": false,
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let next_q_data: serde_json::Value = next_q_response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let next_q_text = next_q_data["message"]["content"]
            .as_str()
            .unwrap_or("");

        let next_questions: Vec<String> = next_q_text
            .lines()
            .filter(|line| line.contains('?'))
            .map(|line| line.trim().to_string())
            .collect();

        Ok(ThoughtProcess {
            timestamp: Self::current_timestamp(),
            question: question.to_string(),
            hypothesis,
            process,
            conclusion: Some(conclusion),
            confidence: 0.7,
            next_questions,
        })
    }

    /// 無制限コード生成（Rust/SIGG/Python）
    pub fn generate_code_unlimited(
        &self,
        description: &str,
        language: TargetLanguage,
    ) -> Result<String, SiggError> {
        println!("💻 コード生成中 ({:?})...", language);

        let lang_name = match language {
            TargetLanguage::Rust => "Rust",
            TargetLanguage::SIGG => "SIGG",
            TargetLanguage::Python => "Python",
        };

        let system_prompt = match language {
            TargetLanguage::Rust => {
                r#"あなたはRustエキスパートです。
完全に動作する、最適化されたRustコードを生成してください。
- エラーハンドリング必須
- パフォーマンス最適化
- メモリ安全性保証
- 最新のRustイディオム使用"#
            }
            TargetLanguage::SIGG => {
                r#"あなたはSIGG言語のエキスパートです。
SIGG理論に基づいた、セル空間を活用したコードを生成してください。
- 内部セル構造の活用
- セル・ラプラシアンの実装
- SIGG理論の物理的解釈を反映"#
            }
            TargetLanguage::Python => {
                r#"あなたはPythonエキスパートです。
Pythonic で読みやすく、高性能なコードを生成してください。
- 型ヒント必須
- NumPy/pandas活用
- 最新のPython機能使用"#
            }
        };

        let prompt = format!(
            r#"{}

要求: {}

完全に動作する{}コードを生成してください。
コードのみを出力し、説明は不要です。"#,
            system_prompt, description, lang_name
        );

        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{}/api/chat", self.llm_endpoint))
            .json(&serde_json::json!({
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "stream": false,
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let data: serde_json::Value = response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let code = data["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let code = Self::extract_code_block(&code, lang_name);

        self.stats.lock().unwrap().code_generated += 1;

        Ok(code)
    }

    /// 自己プログラム書き換え（無制限）
    pub fn self_modify_unrestricted(
        &self,
        target_file: &str,
        modification_desc: &str,
    ) -> Result<(), SiggError> {
        if self.permission_level != PermissionLevel::Unrestricted {
            return Err(SiggError::runtime("無制限権限が必要です".to_string()));
        }

        println!("🔧 自己改変実行: {}", target_file);

        let file_path = self.project_root.join("src").join(target_file);

        // バックアップ作成
        let backup_path = self.create_timestamped_backup(&file_path)?;

        // 現在のコード読み込み
        let current_code = std::fs::read_to_string(&file_path)
            .map_err(|e| SiggError::runtime(format!("ファイル読み込みエラー: {}", e)))?;

        // LLMで新しいコード生成
        let prompt = format!(
            r#"以下のRustコードを改変してください。

現在のコード:
```rust
{}
```

改変内容:
{}

要件:
1. 既存の機能を壊さない
2. エラーハンドリング
3. テストを追加
4. パフォーマンス向上

完全なコード（既存 + 改変）のみを出力してください。"#,
            current_code, modification_desc
        );

        let client = reqwest::blocking::Client::new();
        let response = client
            .post(format!("{}/api/chat", self.llm_endpoint))
            .json(&serde_json::json!({
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "stream": false,
                "options": {"num_predict": 8192}
            }))
            .send()
            .map_err(|e| SiggError::runtime(format!("LLMエラー: {}", e)))?;

        let data: serde_json::Value = response.json()
            .map_err(|e| SiggError::runtime(format!("JSONパースエラー: {}", e)))?;

        let new_code = data["message"]["content"]
            .as_str()
            .unwrap_or("");

        let new_code = Self::extract_code_block(new_code, "rust");

        // 新しいコードを書き込み
        std::fs::write(&file_path, new_code)
            .map_err(|e| SiggError::runtime(format!("ファイル書き込みエラー: {}", e)))?;

        // ビルドテスト
        println!("🔨 ビルドテスト中...");
        let build_result = self.build_project()?;

        if !build_result {
            // ビルド失敗 → バックアップから復元
            println!("❌ ビルド失敗 - バックアップから復元");
            std::fs::copy(&backup_path, &file_path)
                .map_err(|e| SiggError::runtime(format!("復元エラー: {}", e)))?;

            return Err(SiggError::runtime("ビルド失敗".to_string()));
        }

        println!("✅ 自己改変成功: {}", target_file);

        // 改変ログ記録
        self.modification_log.lock().unwrap().push(ModificationRecord {
            timestamp: Self::current_timestamp(),
            modification_type: "self_modify".to_string(),
            target_file: target_file.to_string(),
            description: modification_desc.to_string(),
            success: true,
            backup_path: Some(backup_path.to_string_lossy().to_string()),
        });

        self.stats.lock().unwrap().self_modifications += 1;

        Ok(())
    }

    /// PC無制限アクセス
    pub fn execute_system_command(&self, command: &str) -> Result<String, SiggError> {
        if self.permission_level != PermissionLevel::Unrestricted {
            return Err(SiggError::runtime("無制限権限が必要です".to_string()));
        }

        println!("⚡ システムコマンド実行: {}", command);

        let output = if cfg!(target_os = "windows") {
            Command::new("cmd")
                .args(&["/C", command])
                .output()
        } else {
            Command::new("sh")
                .args(&["-c", command])
                .output()
        };

        let output = output
            .map_err(|e| SiggError::runtime(format!("コマンド実行エラー: {}", e)))?;

        self.stats.lock().unwrap().system_commands += 1;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(SiggError::runtime(
                String::from_utf8_lossy(&output.stderr).to_string()
            ))
        }
    }

    /// プロジェクトビルド
    fn build_project(&self) -> Result<bool, SiggError> {
        let output = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| SiggError::runtime(format!("ビルドエラー: {}", e)))?;

        Ok(output.status.success())
    }

    /// タイムスタンプ付きバックアップ
    fn create_timestamped_backup(&self, file_path: &Path) -> Result<PathBuf, SiggError> {
        let backup_dir = self.project_root.join("backups");
        std::fs::create_dir_all(&backup_dir)
            .map_err(|e| SiggError::runtime(format!("バックアップディレクトリ作成エラー: {}", e)))?;

        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let file_name = file_path.file_name().unwrap().to_string_lossy();
        let backup_path = backup_dir.join(format!("{}_{}", timestamp, file_name));

        std::fs::copy(file_path, &backup_path)
            .map_err(|e| SiggError::runtime(format!("バックアップエラー: {}", e)))?;

        Ok(backup_path)
    }

    /// コードブロック抽出
    fn extract_code_block(text: &str, lang: &str) -> String {
        if let Some(start) = text.find(&format!("```{}", lang)) {
            let after_start = &text[start + 3 + lang.len()..];
            if let Some(end) = after_start.find("```") {
                return after_start[..end].trim().to_string();
            }
        }

        if let Some(start) = text.find("```") {
            let after_start = &text[start + 3..];
            if let Some(end) = after_start.find("```") {
                return after_start[..end].trim().to_string();
            }
        }

        text.trim().to_string()
    }

    /// 現在のタイムスタンプ
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// 自律思考停止
    pub fn stop_thinking(&self) {
        *self.thinking_active.lock().unwrap() = false;
    }

    /// 思考履歴取得
    pub fn get_thought_history(&self) -> Vec<ThoughtProcess> {
        self.thought_history.lock().unwrap().clone()
    }

    /// 統計情報取得
    pub fn get_stats(&self) -> AIStats {
        self.stats.lock().unwrap().clone()
    }
}