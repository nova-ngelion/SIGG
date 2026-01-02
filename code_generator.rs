// src/code_generator.rs
// 対話からSIGGコードを生成

use crate::error::SiggError;
use crate::llm_interface::{LlmInterface, LlmRequest, PromptTemplates};
use std::path::Path;

pub struct CodeGenerator {
    llm: LlmInterface,
}

impl CodeGenerator {
    pub fn new(llm: LlmInterface) -> Self {
        Self { llm }
    }

    /// ユーザーの要求からSIGGコードを生成
    pub async fn generate_from_description(
        &mut self,
        description: &str,
    ) -> Result<String, SiggError> {
        let prompt = PromptTemplates::code_generation_prompt(description);
        
        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.3, // コード生成は低温度で
            max_tokens: 4096,
        };

        let response = self.llm.generate(request).await?;
        
        // コードブロックを抽出
        let code = self.extract_code_block(&response.text);
        
        Ok(code)
    }

    /// コードの説明を生成
    pub async fn explain_code(&mut self, code: &str) -> Result<String, SiggError> {
        let prompt = PromptTemplates::explain_code_prompt(code);
        
        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.7,
            max_tokens: 2048,
        };

        let response = self.llm.generate(request).await?;
        Ok(response.text)
    }

    /// コードの改善提案
    pub async fn suggest_improvements(&mut self, code: &str) -> Result<String, SiggError> {
        let prompt = format!(
            r#"以下のSIGGコードを分析し、改善提案を日本語で提供してください:

```sigg
{}
```

以下の観点から分析してください:
1. パフォーマンス最適化
2. コードの可読性
3. SIGG理論の原則への適合性
4. エラーハンドリング
5. セキュリティ"#,
            code
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.7,
            max_tokens: 2048,
        };

        let response = self.llm.generate(request).await?;
        Ok(response.text)
    }

    /// 機能の追加
    pub async fn add_feature(
        &mut self,
        existing_code: &str,
        feature_description: &str,
    ) -> Result<String, SiggError> {
        let prompt = format!(
            r#"以下の既存のSIGGコードに新しい機能を追加してください:

既存コード:
```sigg
{}
```

追加する機能:
{}

完全なコード（既存 + 新機能）を出力してください。"#,
            existing_code, feature_description
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.4,
            max_tokens: 4096,
        };

        let response = self.llm.generate(request).await?;
        let code = self.extract_code_block(&response.text);
        
        Ok(code)
    }

    /// バグ修正
    pub async fn fix_bug(
        &mut self,
        code: &str,
        error_message: &str,
    ) -> Result<String, SiggError> {
        let prompt = format!(
            r#"以下のSIGGコードにエラーがあります。修正してください:

コード:
```sigg
{}
```

エラーメッセージ:
{}

修正したコードを出力してください。"#,
            code, error_message
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.2, // バグ修正は決定論的に
            max_tokens: 4096,
        };

        let response = self.llm.generate(request).await?;
        let code = self.extract_code_block(&response.text);
        
        Ok(code)
    }

    /// コードブロック抽出ヘルパー
    fn extract_code_block(&self, text: &str) -> String {
        // ```sigg ... ``` または ``` ... ``` からコードを抽出
        if let Some(start) = text.find("```") {
            let after_start = &text[start + 3..];
            
            // 言語指定をスキップ
            let code_start = if after_start.starts_with("sigg") {
                after_start.find('\n').unwrap_or(0) + 1
            } else if after_start.starts_with('\n') {
                1
            } else {
                0
            };
            
            if let Some(end) = after_start[code_start..].find("```") {
                return after_start[code_start..code_start + end].trim().to_string();
            }
        }
        
        // コードブロックが見つからない場合は全体を返す
        text.trim().to_string()
    }

    /// テストケース生成
    pub async fn generate_tests(&mut self, code: &str) -> Result<String, SiggError> {
        let prompt = format!(
            r#"以下のSIGGコードに対するテストケースを生成してください:

```sigg
{}
```

包括的なテストケースを含むコードを出力してください。"#,
            code
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.5,
            max_tokens: 3072,
        };

        let response = self.llm.generate(request).await?;
        let code = self.extract_code_block(&response.text);
        
        Ok(code)
    }

    /// ドキュメント生成
    pub async fn generate_documentation(&mut self, code: &str) -> Result<String, SiggError> {
        let prompt = format!(
            r#"以下のSIGGコードの詳細なドキュメントを日本語で生成してください:

```sigg
{}
```

含めるべき内容:
1. 概要
2. 関数/メソッドの説明
3. 使用例
4. 注意事項
5. SIGG理論との関連"#,
            code
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.6,
            max_tokens: 3072,
        };

        let response = self.llm.generate(request).await?;
        Ok(response.text)
    }

    /// 最適化
    pub async fn optimize_code(&mut self, code: &str) -> Result<String, SiggError> {
        let prompt = format!(
            r#"以下のSIGGコードを最適化してください:

```sigg
{}
```

最適化の観点:
1. 計算効率
2. メモリ使用量
3. セル空間の効率的な利用
4. 並列化の可能性

最適化後のコードを出力してください。"#,
            code
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(PromptTemplates::system_prompt()),
            temperature: 0.3,
            max_tokens: 4096,
        };

        let response = self.llm.generate(request).await?;
        let code = self.extract_code_block(&response.text);
        
        Ok(code)
    }
}

/// コードテンプレート
pub struct CodeTemplates;

impl CodeTemplates {
    /// 基本的なSIGGプログラムテンプレート
    pub fn basic_program() -> &'static str {
        r#"// 基本的なSIGGプログラム

fn main() {
    // セル空間の初期化
    let cell_dim = 10;
    let space = create_cell_space(cell_dim);
    
    // 計算処理
    let result = compute(space);
    
    // 結果の出力
    print(result);
}

fn compute(space) {
    // ここに処理を記述
    space
}"#
    }

    /// SIGG-GNN利用テンプレート
    pub fn gnn_program() -> &'static str {
        r#"// SIGG-GNNを使用したプログラム

import sigg_gnn;

fn main() {
    // グラフの構築
    let gnn = sigg_gnn::new(
        feature_dim: 64,
        cell_dim: 10,
        layers: [128, 64, 32]
    );
    
    // ノードの追加
    for i in 0..100 {
        let node = create_node(i, 64, 10);
        gnn.add_node(node);
    }
    
    // エッジの追加
    add_edges(gnn);
    
    // 学習
    gnn.forward(iterations: 50);
    
    // ダークモード検出
    let dark_modes = gnn.detect_dark_modes();
    print("検出されたダークモード:", dark_modes);
}"#
    }

    /// ファイル操作テンプレート
    pub fn file_operations() -> &'static str {
        r#"// ファイル操作の例

import fs;

fn main() {
    // ファイル読み込み
    let content = fs::read("input.txt");
    
    // 処理
    let processed = process(content);
    
    // ファイル書き込み
    fs::write("output.txt", processed);
    
    // ディレクトリ操作
    fs::create_dir("results");
    fs::copy("output.txt", "results/output.txt");
}"#
    }
}