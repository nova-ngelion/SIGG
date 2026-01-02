// src/self_modify.rs
// 自己改変およびアップデートシステム

use crate::error::SiggError;
use crate::file_operations::FileOperations;
use crate::llm_interface::{LlmInterface, LlmRequest};
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct SelfModifier {
    file_ops: FileOperations,
    llm: LlmInterface,
    project_root: PathBuf,
    backup_dir: PathBuf,
}

impl SelfModifier {
    pub fn new(
        project_root: impl AsRef<Path>,
        file_ops: FileOperations,
        llm: LlmInterface,
    ) -> Self {
        let project_root = project_root.as_ref().to_path_buf();
        let backup_dir = project_root.join("backups");
        
        Self {
            file_ops,
            llm,
            project_root,
            backup_dir,
        }
    }

    /// 新機能を追加
    pub async fn add_feature(
        &mut self,
        feature_description: &str,
        target_file: &str,
    ) -> Result<(), SiggError> {
        println!("🔧 新機能を追加中: {}", feature_description);
        
        // 対象ファイル読み込み
        let file_path = self.project_root.join("src").join(target_file);
        let existing_code = self.file_ops.read_file(&file_path)?;
        
        // バックアップ作成
        self.create_version_backup(&file_path)?;
        
        // LLMで新しいコードを生成
        let prompt = format!(
            r#"以下のRustコードに新しい機能を追加してください。

既存コード:
```rust
{}
```

追加する機能:
{}

要件:
1. 既存の機能を壊さない
2. 適切なエラーハンドリング
3. コメントを日本語で
4. SIGG理論の原則に従う
5. 完全に動作するコード

完全なコード（既存 + 新機能）のみを出力してください。"#,
            existing_code, feature_description
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(self.system_prompt()),
            temperature: 0.3,
            max_tokens: 8192,
        };

        let response = self.llm.generate(request).await?;
        let new_code = self.extract_rust_code(&response.text);
        
        // 新しいコードを書き込み
        self.file_ops.write_file(&file_path, &new_code)?;
        
        println!("✅ コードを更新しました: {}", target_file);
        
        // ビルドテスト
        self.test_build().await?;
        
        Ok(())
    }

    /// バグ修正
    pub async fn fix_bug(
        &mut self,
        bug_description: &str,
        target_file: Option<&str>,
    ) -> Result<(), SiggError> {
        println!("🐛 バグを修正中: {}", bug_description);
        
        // 対象ファイルを特定
        let file_path = if let Some(file) = target_file {
            self.project_root.join("src").join(file)
        } else {
            // LLMに対象ファイルを推測させる
            self.identify_bug_location(bug_description).await?
        };
        
        let existing_code = self.file_ops.read_file(&file_path)?;
        self.create_version_backup(&file_path)?;
        
        let prompt = format!(
            r#"以下のRustコードのバグを修正してください。

コード:
```rust
{}
```

バグの説明:
{}

修正したコードを出力してください。変更箇所にコメントをつけてください。"#,
            existing_code, bug_description
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(self.system_prompt()),
            temperature: 0.2,
            max_tokens: 8192,
        };

        let response = self.llm.generate(request).await?;
        let fixed_code = self.extract_rust_code(&response.text);
        
        self.file_ops.write_file(&file_path, &fixed_code)?;
        
        println!("✅ バグを修正しました");
        
        self.test_build().await?;
        
        Ok(())
    }

    /// コード最適化
    pub async fn optimize_code(&mut self, target_file: &str) -> Result<(), SiggError> {
        println!("⚡ コードを最適化中: {}", target_file);
        
        let file_path = self.project_root.join("src").join(target_file);
        let existing_code = self.file_ops.read_file(&file_path)?;
        
        self.create_version_backup(&file_path)?;
        
        let prompt = format!(
            r#"以下のRustコードを最適化してください。

コード:
```rust
{}
```

最適化の観点:
1. パフォーマンス向上
2. メモリ効率
3. コードの可読性
4. 並列化の可能性
5. SIGG理論の効率的な実装

最適化後のコードを出力してください。"#,
            existing_code
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(self.system_prompt()),
            temperature: 0.3,
            max_tokens: 8192,
        };

        let response = self.llm.generate(request).await?;
        let optimized_code = self.extract_rust_code(&response.text);
        
        self.file_ops.write_file(&file_path, &optimized_code)?;
        
        println!("✅ コードを最適化しました");
        
        self.test_build().await?;
        
        Ok(())
    }

    /// 完全な自動アップデート
    pub async fn auto_update(&mut self) -> Result<(), SiggError> {
        println!("🔄 自動アップデートを開始します...");
        
        // 1. プロジェクト全体の分析
        let analysis = self.analyze_project().await?;
        
        // 2. 改善提案の取得
        let suggestions = self.get_improvement_suggestions(&analysis).await?;
        
        println!("📋 改善提案:");
        for (i, suggestion) in suggestions.iter().enumerate() {
            println!("  {}. {}", i + 1, suggestion);
        }
        
        // 3. 各改善を適用
        for suggestion in suggestions {
            if let Err(e) = self.apply_suggestion(&suggestion).await {
                eprintln!("⚠️  改善適用エラー: {}", e);
            }
        }
        
        // 4. 最終ビルドとテスト
        self.full_rebuild().await?;
        
        println!("✅ 自動アップデート完了!");
        
        Ok(())
    }

    /// プロジェクト分析
    async fn analyze_project(&mut self) -> Result<String, SiggError> {
        let src_dir = self.project_root.join("src");
        let files = self.file_ops.list_directory(&src_dir)?;
        
        let mut analysis = String::from("# プロジェクト分析\n\n");
        
        for file_path in files {
            if file_path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = self.file_ops.read_file(&file_path) {
                    let lines = content.lines().count();
                    analysis.push_str(&format!(
                        "- {}: {} 行\n",
                        file_path.file_name().unwrap().to_string_lossy(),
                        lines
                    ));
                }
            }
        }
        
        Ok(analysis)
    }

    /// 改善提案取得
    async fn get_improvement_suggestions(&mut self, analysis: &str) -> Result<Vec<String>, SiggError> {
        let prompt = format!(
            r#"以下のSIGG言語プロジェクトの分析結果から、改善提案を3-5個提示してください。

{}

各提案は具体的で実行可能なものにしてください。
形式: 「<ファイル名>: <改善内容>」"#,
            analysis
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(self.system_prompt()),
            temperature: 0.7,
            max_tokens: 2048,
        };

        let response = self.llm.generate(request).await?;
        
        // 提案を行ごとに分割
        let suggestions: Vec<String> = response
            .text
            .lines()
            .filter(|line| !line.trim().is_empty() && line.contains(':'))
            .map(|s| s.trim().to_string())
            .collect();
        
        Ok(suggestions)
    }

    /// 提案を適用
    async fn apply_suggestion(&mut self, suggestion: &str) -> Result<(), SiggError> {
        println!("🔨 適用中: {}", suggestion);
        
        // 提案からファイル名と内容を抽出
        let parts: Vec<&str> = suggestion.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(SiggError::runtime("無効な提案形式".to_string()));
        }
        
        let file_name = parts[0].trim();
        let improvement = parts[1].trim();
        
        self.add_feature(improvement, file_name).await?;
        
        Ok(())
    }

    /// バグの場所を特定
    async fn identify_bug_location(&mut self, bug_description: &str) -> Result<PathBuf, SiggError> {
        let prompt = format!(
            r#"以下のバグの説明から、最も関連するRustファイル名を1つだけ答えてください。

バグの説明: {}

プロジェクト構造:
- ai_runtime.rs
- llm_interface.rs
- sigg_gnn.rs
- code_generator.rs
- file_operations.rs
- self_modify.rs
- builtins.rs
- vm.rs

ファイル名のみを答えてください（説明不要）。"#,
            bug_description
        );

        let request = LlmRequest {
            prompt,
            system_prompt: Some(self.system_prompt()),
            temperature: 0.1,
            max_tokens: 64,
        };

        let response = self.llm.generate(request).await?;
        let file_name = response.text.trim();
        
        Ok(self.project_root.join("src").join(file_name))
    }

    /// ビルドテスト
    async fn test_build(&self) -> Result<(), SiggError> {
        println!("🔨 ビルド中...");
        
        let output = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| SiggError::runtime(format!("ビルドコマンド実行エラー: {}", e)))?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(SiggError::runtime(format!("ビルドエラー:\n{}", error)));
        }
        
        println!("✅ ビルド成功");
        Ok(())
    }

    /// 完全リビルド
    async fn full_rebuild(&self) -> Result<(), SiggError> {
        println!("🧹 クリーン中...");
        
        Command::new("cargo")
            .arg("clean")
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| SiggError::runtime(format!("クリーンエラー: {}", e)))?;
        
        self.test_build().await?;
        
        println!("📦 インストール中...");
        
        let output = Command::new("cargo")
            .args(&["install", "--path", "."])
            .current_dir(&self.project_root)
            .output()
            .map_err(|e| SiggError::runtime(format!("インストールエラー: {}", e)))?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(SiggError::runtime(format!("インストールエラー:\n{}", error)));
        }
        
        println!("✅ インストール完了");
        Ok(())
    }

    /// バージョンバックアップ
    fn create_version_backup(&self, file_path: &Path) -> Result<(), SiggError> {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let file_name = file_path.file_name().unwrap().to_string_lossy();
        let backup_name = format!("{}_{}", timestamp, file_name);
        let backup_path = self.backup_dir.join(backup_name);
        
        self.file_ops.create_directory(&self.backup_dir)?;
        self.file_ops.copy_file(file_path, backup_path)?;
        
        Ok(())
    }

    fn extract_rust_code(&self, text: &str) -> String {
        if let Some(start) = text.find("```rust") {
            let after_start = &text[start + 7..];
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

    fn system_prompt(&self) -> String {
        r#"あなたはSIGG言語とRustのエキスパートです。
コードの改善、バグ修正、最適化を行います。
常に以下を遵守してください:
1. 完全に動作するコード
2. 既存機能を壊さない
3. 適切なエラーハンドリング
4. 日本語コメント
5. SIGG理論の原則に従う"#.to_string()
    }
}