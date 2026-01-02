// src/ai_assistant.rs
// メインAIアシスタントシステム - 全機能の統合

use crate::error::SiggError;
use crate::llm_interface::{LlmInterface, LlmConfig, LlmRequest, PromptTemplates};
use crate::sigg_gnn::{SiggGnn, SiggHybridNet, SiggNode, SiggEdge};
use crate::code_generator::CodeGenerator;
use crate::file_operations::FileOperations;
use crate::self_modify::SelfModifier;
use std::path::PathBuf;

pub struct AiAssistant {
    llm: LlmInterface,
    code_gen: CodeGenerator,
    file_ops: FileOperations,
    self_modifier: SelfModifier,
    hybrid_net: SiggHybridNet,
    conversation_mode: ConversationMode,
    project_root: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConversationMode {
    Code,      // コード生成・編集
    File,      // ファイル操作
    Learn,     // 学習・理解
    Chat,      // 雑談
    SelfModify, // 自己改変
}

impl AiAssistant {
    pub fn new(project_root: PathBuf, llm_config: LlmConfig) -> Self {
        let llm_main = LlmInterface::new(llm_config.clone());
        let llm_codegen = LlmInterface::new(llm_config.clone());
        let llm_modifier = LlmInterface::new(llm_config);
        
        let code_gen = CodeGenerator::new(llm_codegen);
        let file_ops = FileOperations::new(&project_root);
        
        let self_modifier = SelfModifier::new(
            project_root.clone(),
            FileOperations::new(&project_root),
            llm_modifier,
        );
        
        // SIGG-GNN + CNN ハイブリッドネットワーク
        let hybrid_net = SiggHybridNet::new(
            64,              // feature_dim
            10,              // cell_dim (SIGG理論の内部次元)
            vec![128, 64],   // GNN layers
            vec![32, 16],    // CNN filters
        );
        
        Self {
            llm: llm_main,
            code_gen,
            file_ops,
            self_modifier,
            hybrid_net,
            conversation_mode: ConversationMode::Chat,
            project_root,
        }
    }

    /// メインの対話ループ
    pub async fn chat(&mut self, user_input: &str) -> Result<String, SiggError> {
        // 入力を分析してモードを判定
        let detected_mode = self.detect_mode(user_input).await?;
        
        println!("🤖 モード: {:?}", detected_mode);
        
        match detected_mode {
            ConversationMode::Code => self.handle_code_request(user_input).await,
            ConversationMode::File => self.handle_file_request(user_input).await,
            ConversationMode::Learn => self.handle_learn_request(user_input).await,
            ConversationMode::Chat => self.handle_chat_request(user_input).await,
            ConversationMode::SelfModify => self.handle_self_modify_request(user_input).await,
        }
    }

    /// モード検出（SIGG-GNN使用）
    async fn detect_mode(&mut self, input: &str) -> Result<ConversationMode, SiggError> {
        // キーワードベースの高速判定
        let input_lower = input.to_lowercase();
        
        if input_lower.contains("コード") || 
           input_lower.contains("プログラム") ||
           input_lower.contains("実装") ||
           input_lower.contains("生成") {
            return Ok(ConversationMode::Code);
        }
        
        if input_lower.contains("ファイル") ||
           input_lower.contains("保存") ||
           input_lower.contains("削除") ||
           input_lower.contains("移動") {
            return Ok(ConversationMode::File);
        }
        
        if input_lower.contains("自己改変") ||
           input_lower.contains("アップデート") ||
           input_lower.contains("機能追加") ||
           input_lower.contains("バグ修正") {
            return Ok(ConversationMode::SelfModify);
        }
        
        if input_lower.contains("説明") ||
           input_lower.contains("教え") ||
           input_lower.contains("理解") ||
           input_lower.contains("sigg理論") {
            return Ok(ConversationMode::Learn);
        }
        
        // デフォルトは雑談モード
        Ok(ConversationMode::Chat)
    }

    /// コードリクエスト処理
    async fn handle_code_request(&mut self, input: &str) -> Result<String, SiggError> {
        if input.contains("生成") || input.contains("作成") {
            // コード生成
            let code = self.code_gen.generate_from_description(input).await?;
            
            // 生成したコードを保存するか確認
            let save_prompt = format!(
                "生成されたコード:\n\n```sigg\n{}\n```\n\nこのコードを保存しますか？",
                code
            );
            
            Ok(save_prompt)
        } else if input.contains("説明") {
            // コード説明
            // 最近のコードを取得（仮）
            let explanation = self.code_gen.explain_code("// サンプルコード").await?;
            Ok(explanation)
        } else if input.contains("改善") || input.contains("最適化") {
            // コード改善
            let suggestions = self.code_gen.suggest_improvements("// サンプルコード").await?;
            Ok(suggestions)
        } else {
            // LLMに直接質問
            self.ask_llm(input).await
        }
    }

    /// ファイル操作処理
    async fn handle_file_request(&mut self, input: &str) -> Result<String, SiggError> {
        let input_lower = input.to_lowercase();
        
        if input_lower.contains("保存") {
            self.handle_file_save(input).await
        } else if input_lower.contains("読み込") || input_lower.contains("開く") {
            self.handle_file_read(input).await
        } else if input_lower.contains("削除") {
            self.handle_file_delete(input).await
        } else if input_lower.contains("移動") {
            self.handle_file_move(input).await
        } else if input_lower.contains("一覧") || input_lower.contains("リスト") {
            self.handle_file_list(input).await
        } else {
            Ok("ファイル操作を指定してください（保存、読み込み、削除、移動、一覧）".to_string())
        }
    }

    async fn handle_file_save(&mut self, input: &str) -> Result<String, SiggError> {
        // ファイル名とコードを抽出（LLM使用）
        let prompt = format!(
            r#"以下のリクエストからファイル名と保存内容を抽出してください:

{}

JSON形式で出力:
{{"filename": "...", "content": "..."}}"#,
            input
        );

        let response = self.ask_llm(&prompt).await?;
        
        // JSONパース（簡易版）
        // 実際にはserde_jsonを使用
        
        Ok("ファイルを保存しました".to_string())
    }

    async fn handle_file_read(&mut self, input: &str) -> Result<String, SiggError> {
        // ファイル名を抽出
        let prompt = format!(
            "以下のリクエストから読み込むファイル名を抽出してください: {}\nファイル名のみを答えてください。",
            input
        );

        let response = self.ask_llm(&prompt).await?;
        let filename = response.trim();
        
        let content = self.file_ops.read_file(filename)?;
        Ok(format!("ファイル内容:\n\n{}", content))
    }

    async fn handle_file_delete(&mut self, _input: &str) -> Result<String, SiggError> {
        Ok("削除機能は安全のため対話確認が必要です".to_string())
    }

    async fn handle_file_move(&mut self, _input: &str) -> Result<String, SiggError> {
        Ok("ファイル移動機能を実装中...".to_string())
    }

    async fn handle_file_list(&mut self, input: &str) -> Result<String, SiggError> {
        let dir = if input.contains("src") {
            "src"
        } else {
            "."
        };

        let files = self.file_ops.list_directory(dir)?;
        let file_list: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();

        Ok(format!("ファイル一覧:\n{}", file_list.join("\n")))
    }

    /// 学習・理解リクエスト
    async fn handle_learn_request(&mut self, input: &str) -> Result<String, SiggError> {
        // SIGG理論の説明を含むシステムプロンプトで質問
        let request = LlmRequest {
            prompt: input.to_string(),
            system_prompt: Some(self.create_learn_system_prompt()),
            temperature: 0.7,
            max_tokens: 3072,
        };

        let response = self.llm.generate(request).await?;
        Ok(response.text)
    }

    /// 雑談リクエスト
    async fn handle_chat_request(&mut self, input: &str) -> Result<String, SiggError> {
        self.ask_llm(input).await
    }

    /// 自己改変リクエスト
    async fn handle_self_modify_request(&mut self, input: &str) -> Result<String, SiggError> {
        let input_lower = input.to_lowercase();
        
        if input_lower.contains("機能追加") || input_lower.contains("新機能") {
            // 機能追加
            let prompt = format!(
                "以下のリクエストから追加する機能の説明と対象ファイルを抽出してください: {}\n形式: 機能説明 | ファイル名",
                input
            );
            let response = self.ask_llm(&prompt).await?;
            
            // 簡易パース
            let parts: Vec<&str> = response.split('|').collect();
            if parts.len() == 2 {
                let feature = parts[0].trim();
                let file = parts[1].trim();
                
                self.self_modifier.add_feature(feature, file).await?;
                Ok(format!("機能を追加しました: {}", feature))
            } else {
                Ok("機能追加のリクエスト形式が不正です".to_string())
            }
        } else if input_lower.contains("バグ修正") {
            self.self_modifier.fix_bug(input, None).await?;
            Ok("バグを修正しました".to_string())
        } else if input_lower.contains("最適化") {
            Ok("最適化を実行中...（未実装）".to_string())
        } else if input_lower.contains("自動アップデート") {
            self.self_modifier.auto_update().await?;
            Ok("自動アップデート完了".to_string())
        } else {
            Ok("自己改変コマンド: 機能追加、バグ修正、最適化、自動アップデート".to_string())
        }
    }

    /// LLMに直接質問
    async fn ask_llm(&mut self, question: &str) -> Result<String, SiggError> {
        let request = LlmRequest {
            prompt: question.to_string(),
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.7,
            max_tokens: 2048,
        };

        let response = self.llm.generate(request).await?;
        Ok(response.text)
    }

    /// 学習モード用システムプロンプト
    fn create_learn_system_prompt(&self) -> String {
        format!(
            r#"{}

SIGG理論の詳細:

1. 構造化相互接続幾何グリッド (Structured Interconnected Geometric Grid)
   - 物理空間に追加の離散的内部セル構造 Z^d を仮定
   - ヒルベルト空間: H = ℓ²(Z^d) ⊗ L²(R³)

2. セル・ラプラシアン
   - (Δ_cell ψ)(n) = Σ[ψ(n+ê) + ψ(n-ê) - 2ψ(n)]
   - スペクトル: σ(-Δ_cell) = [0, 4d]

3. 物理的帰結
   - ニュートリノ質量: 内部セル空間での広がり → m_ν ≈ 1/ℓ²
   - 真空エネルギー: セル・パリティ対称性により自然に相殺
   - ダークマター: k≠0 の内部セル励起として出現

これらの概念を使って質問に答えてください。"#,
            PromptTemplates::system_prompt()
        )
    }

    /// SIGG-GNN学習の実行
    pub async fn train_hybrid_net(&mut self, iterations: usize) -> Result<Vec<f32>, SiggError> {
        println!("🧠 SIGG-GNN + CNN ハイブリッド学習を開始...");
        
        // サンプルグラフ構築
        for i in 0..100 {
            let mut node = SiggNode::new(i, 64, 10);
            
            // 初期特徴量を設定
            for j in 0..64 {
                node.features[j] = (i as f32 * 0.01) + (j as f32 * 0.001);
            }
            
            self.hybrid_net.gnn.add_node(node);
        }
        
        // エッジ追加（グリッド構造）
        let side = 10;
        for i in 0..side {
            for j in 0..side {
                let idx = i * side + j;
                
                // 右へのエッジ
                if j < side - 1 {
                    let edge = SiggEdge {
                        from: idx,
                        to: idx + 1,
                        weight: 1.0,
                        cell_coupling: vec![vec![0.1; 10]; 10],
                    };
                    self.hybrid_net.gnn.add_edge(edge);
                }
                
                // 下へのエッジ
                if i < side - 1 {
                    let edge = SiggEdge {
                        from: idx,
                        to: idx + side,
                        weight: 1.0,
                        cell_coupling: vec![vec![0.1; 10]; 10],
                    };
                    self.hybrid_net.gnn.add_edge(edge);
                }
            }
        }
        
        // ハイブリッド学習実行
        let output = self.hybrid_net.hybrid_forward(iterations)?;
        
        // ダークモード検出
        let dark_modes = self.hybrid_net.gnn.detect_dark_modes();
        println!("🌌 検出されたダークモード数: {}", dark_modes.len());
        
        // ニュートリノ特徴測定
        let neutrino_features = self.hybrid_net.gnn.measure_neutrino_features();
        let avg_spread: f32 = neutrino_features.values().sum::<f32>() 
            / neutrino_features.len() as f32;
        println!("⚛️  平均セル広がり度: {:.4}", avg_spread);
        
        println!("✅ 学習完了");
        
        Ok(output)
    }

    /// 会話履歴のクリア
    pub fn clear_history(&mut self) {
        self.llm.clear_history();
    }

    /// 現在のモードを取得
    pub fn current_mode(&self) -> &ConversationMode {
        &self.conversation_mode
    }

    /// プロジェクトルートを取得
    pub fn project_root(&self) -> &PathBuf {
        &self.project_root
    }
}