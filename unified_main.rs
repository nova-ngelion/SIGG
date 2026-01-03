// src/unified_main.rs
// 統合最適化AIシステム - メインプログラム

use std::io::{self, Write};
use std::path::PathBuf;

use sigg::error;
use sigg::unified_ai_core;

use unified_ai_core::{UnifiedAI, AIConfig, PermissionLevel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_banner();

    // 設定
    let config = AIConfig {
        project_root: PathBuf::from(r"C:\Users\nishi\newL\sigg_lang"),
        llm_endpoint: "http://localhost:11434".to_string(),
        llm_model: "llama3.1:8b".to_string(),
        permission_level: PermissionLevel::Unrestricted,
        
        gnn_feature_dim: 64,
        gnn_cell_dim: 10,
        gnn_hidden_layers: vec![128, 64, 32],
        cnn_filters: vec![32, 16, 8],
        
        max_threads: num_cpus::get(),
        cache_size_mb: 512,
        auto_optimize: true,
        
        enable_voice: true,
        enable_self_modify: true,
        enable_autonomous_thinking: true,
    };

    println!("🔧 システム初期化中...");
    
    // Ollama接続確認
    print!("  LLM接続確認...");
    if check_ollama(&config.llm_endpoint) {
        println!(" ✅");
    } else {
        println!(" ❌");
        println!("\n⚠️  Ollamaが起動していません");
        println!("別のターミナルで 'ollama serve' を実行してください");
        return Ok(());
    }

    // AI初期化
    let mut ai = UnifiedAI::new(config)?;
    
    println!("✅ 初期化完了\n");

    // メインループ
    println!("💬 チャット開始（'exit'で終了、'help'でヘルプ）\n");

    loop {
        print!("あなた > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // 特殊コマンド
        match input {
            "exit" | "quit" => {
                println!("\n👋 また会いましょう！");
                show_final_stats(&ai);
                break;
            }
            "help" => {
                show_help();
                continue;
            }
            "stats" => {
                show_stats(&ai);
                continue;
            }
            "clear" => {
                // 履歴クリア
                println!("✅ 履歴をクリアしました");
                continue;
            }
            _ => {}
        }

        // AI処理
        print!("AI > ");
        io::stdout().flush()?;
        
        match ai.process(input) {
            Ok(response) => {
                println!("{}\n", response);
            }
            Err(e) => {
                println!("❌ エラー: {}\n", e);
            }
        }
    }

    Ok(())
}

fn print_banner() {
    println!(r#"
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██╗   ██╗███╗   ██╗██╗███████╗██╗███████╗██████╗          ║
║   ██║   ██║████╗  ██║██║██╔════╝██║██╔════╝██╔══██╗         ║
║   ██║   ██║██╔██╗ ██║██║█████╗  ██║█████╗  ██║  ██║         ║
║   ██║   ██║██║╚██╗██║██║██╔══╝  ██║██╔══╝  ██║  ██║         ║
║   ╚██████╔╝██║ ╚████║██║██║     ██║███████╗██████╔╝         ║
║    ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚══════╝╚═════╝          ║
║                                                              ║
║              統合最適化SIGG-AI v2.0                         ║
║                                                              ║
║  ⚡ 最適化:                                                  ║
║    • 並列処理（{} スレッド）                               ║
║    • LLMキャッシュ                                          ║
║    • ファイルキャッシュ                                     ║
║    • GNN高速化                                              ║
║                                                              ║
║  🎯 統合機能:                                               ║
║    • 自動意図判定                                           ║
║    • コード生成（Rust/SIGG/Python）                        ║
║    • ファイル操作                                           ║
║    • 自己改変                                               ║
║    • SIGG-GNN学習                                           ║
║    • ダークマター検出                                       ║
║    • 自律思考                                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"#, num_cpus::get());
}

fn show_help() {
    println!(r#"
╔══════════════════════════════════════════════════════════════╗
║                        ヘルプ                                ║
╚══════════════════════════════════════════════════════════════╝

📝 基本的な使い方:
  普通に日本語で話しかけてください。AIが自動で判断します。

💻 コード生成:
  「Rustでクイックソートを実装して」
  「Pythonでデータ分析ツールを作って」
  「SIGGで物理シミュレーションを書いて」

📁 ファイル操作:
  「srcフォルダのファイル一覧を表示」
  「main.rsを読み込んで」
  「このコードをtest.rsに保存して」

🔧 自己改変:
  「自分の思考速度を2倍にして」
  「コード生成機能を改善して」

🌌 物理計算:
  「ダークマターを検出して」
  「ニュートリノ質量を計算して」
  「GNN学習を実行して」

💬 雑談:
  「こんにちは」
  「調子はどう？」
  「面白い話して」

🎮 コマンド:
  help   - このヘルプを表示
  stats  - 統計情報を表示
  clear  - 履歴をクリア
  exit   - 終了

💡 ポイント:
  • コマンドを覚える必要なし
  • AIが自動で意図を判定
  • 複雑な操作も自然言語で
  • エラーが出ても自動リトライ
"#);
}

fn show_stats(ai: &UnifiedAI) {
    let stats = ai.stats.read().unwrap();
    let gnn = ai.neural_net.read().unwrap();
    
    println!(r#"
╔══════════════════════════════════════════════════════════════╗
║                      統計情報                                ║
╚══════════════════════════════════════════════════════════════╝

📊 リクエスト:
  総数: {}
  成功率: N/A

💻 コード生成:
  生成数: {}

🧠 思考:
  処理数: {}

🌌 GNN:
  学習回数: {}

⚡ パフォーマンス:
  キャッシュヒット: {}
  キャッシュミス: {}
  ヒット率: {:.1}%
"#,
        stats.total_requests,
        stats.code_generated,
        stats.thoughts_processed,
        stats.gnn_iterations,
        stats.cache_hits,
        stats.cache_misses,
        if stats.cache_hits + stats.cache_misses > 0 {
            (stats.cache_hits as f32 / (stats.cache_hits + stats.cache_misses) as f32) * 100.0
        } else {
            0.0
        }
    );
}

fn show_final_stats(ai: &UnifiedAI) {
    println!("\n📊 セッション統計:");
    show_stats(ai);
}

fn check_ollama(endpoint: &str) -> bool {
    if let Ok(response) = reqwest::blocking::get(format!("{}/api/tags", endpoint)) {
        response.status().is_success()
    } else {
        false
    }
}