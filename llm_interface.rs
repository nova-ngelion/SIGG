// src/llm_interface.rs
// LLM (Llama 3.1 8B) との統合層

use crate::error::SiggError;
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::io::{Write, BufRead, BufReader};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub model_path: String,
    pub endpoint: String,  // "http://localhost:11434" など
    pub model_name: String, // "llama3.1:8b"
    pub temperature: f32,
    pub max_tokens: usize,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            model_path: "llama3.1-8b".to_string(),
            endpoint: "http://localhost:11434".to_string(),
            model_name: "llama3.1:8b".to_string(),
            temperature: 0.7,
            max_tokens: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub temperature: f32,
    pub max_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmResponse {
    pub text: String,
    pub tokens_used: usize,
    pub finish_reason: String,
}

pub struct LlmInterface {
    config: LlmConfig,
    conversation_history: Vec<(String, String)>, // (role, content)
}

impl LlmInterface {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            config,
            conversation_history: Vec::new(),
        }
    }

    /// Ollama API経由でLLMを呼び出し
    pub async fn generate(&mut self, request: LlmRequest) -> Result<LlmResponse, SiggError> {
        let client = reqwest::Client::new();
        
        let mut messages = Vec::new();
        
        // システムプロンプト
        if let Some(sys) = request.system_prompt {
            messages.push(serde_json::json!({
                "role": "system",
                "content": sys
            }));
        }
        
        // 会話履歴
        for (role, content) in &self.conversation_history {
            messages.push(serde_json::json!({
                "role": role,
                "content": content
            }));
        }
        
        // 現在のプロンプト
        messages.push(serde_json::json!({
            "role": "user",
            "content": request.prompt
        }));

        let body = serde_json::json!({
            "model": self.config.model_name,
            "messages": messages,
            "temperature": request.temperature,
            "stream": false,
            "options": {
                "num_predict": request.max_tokens
            }
        });

        let url = format!("{}/api/chat", self.config.endpoint);
        
        let response = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SiggError::runtime(format!("LLM request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(SiggError::runtime(format!(
                "LLM returned error: {}",
                response.status()
            )));
        }

        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SiggError::runtime(format!("Failed to parse LLM response: {}", e)))?;

        let text = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // 会話履歴に追加
        self.conversation_history.push(("user".to_string(), request.prompt));
        self.conversation_history.push(("assistant".to_string(), text.clone()));

        // 履歴が長すぎる場合は古いものを削除
        if self.conversation_history.len() > 20 {
            self.conversation_history.drain(0..2);
        }

        Ok(LlmResponse {
            text,
            tokens_used: 0, // Ollamaは詳細なトークン数を返さない
            finish_reason: "stop".to_string(),
        })
    }

    pub fn clear_history(&mut self) {
        self.conversation_history.clear();
    }

    pub fn get_history(&self) -> &Vec<(String, String)> {
        &self.conversation_history
    }
}

/// 日本語対応のためのプロンプトテンプレート
pub struct PromptTemplates;

impl PromptTemplates {
    pub fn system_prompt() -> String {
        r#"あなたはSIGG言語のエキスパートアシスタントです。
SIGG理論（Structured Interconnected Geometric Grid）に基づいた、
内部次元を持つ量子計算フレームワークを理解しています。

あなたの役割:
1. 日本語での自然な対話
2. SIGGコードの生成と説明
3. ファイル操作の支援
4. コードの改善提案
5. 技術的な質問への回答
6. 雑談への柔軟な対応

SIGG理論の核心概念:
- 内部セル空間（Z^d）を持つ拡張ヒルベルト空間
- 離散セル・ラプラシアン
- ニュートリノ質量、真空エネルギー、ダークマターの統一的説明

常に親切で、技術的に正確で、創造的な解決策を提案してください。"#.to_string()
    }

    pub fn code_generation_prompt(description: &str) -> String {
        format!(
            r#"以下の機能を実装するSIGGコードを生成してください:

{}

要件:
- 完全に動作するコード
- 適切なコメント（日本語）
- エラーハンドリング
- SIGG理論の原則に従う

コードのみを出力してください（説明は不要）。"#,
            description
        )
    }

    pub fn explain_code_prompt(code: &str) -> String {
        format!(
            r#"以下のSIGGコードを日本語で詳しく説明してください:

```sigg
{}
```

説明に含めるべき内容:
1. コードの目的
2. 主要な処理の流れ
3. SIGG理論との関連
4. 潜在的な改善点"#,
            code
        )
    }

    pub fn file_operation_prompt(operation: &str, details: &str) -> String {
        format!(
            r#"ファイル操作: {}

詳細: {}

この操作を実行するための手順を説明してください。
セキュリティとベストプラクティスを考慮してください。"#,
            operation, details
        )
    }
}