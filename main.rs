fn main() {
    if let Err(e) = sigg::cli::main() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

// use std::env;
// use std::io::{Read, Write};
// use std::net::{TcpListener, TcpStream};

// // 既存のモジュールを再利用
// use sigg::pocket::asm;
// use sigg::pocket::cpu::{self, CpuMem};
// use sigg::pocket::compute::ComputeSpace;

// fn main() {
//     let args: Vec<String> = env::args().collect();

//     if args.len() > 1 && args[1] == "bench-server" {
//         start_benchmark_server();
//     } else {
//         // 通常のCLI動作
//         if let Err(e) = sigg::cli::main() {
//             eprintln!("{e}");
//             std::process::exit(1);
//         }
//     }
// }

// fn start_benchmark_server() {
//     let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
//     println!("🚀 SIGG Benchmark Server (CPU-VM Mode) running on 8080");

//     for stream in listener.incoming() {
//         if let Ok(stream) = stream {
//             handle_client(stream);
//         }
//     }
// }

// fn handle_client(mut stream: TcpStream) {
//     let mut buffer = String::new();
//     if let Err(_) = stream.read_to_string(&mut buffer) { return; }

//     use sigg::pocket::tensor::{Tensor, Complex, OpType}; // OpTypeも追加
//     use std::sync::{Arc, RwLock};

//     let start = std::time::Instant::now();

//     // --- SIGG Tensor エンジン生テスト (1000回更新) ---
    
//     // 1. 重みの初期化 (1x1のテンソル)
//     // コンパイラの指示通り zeros を使用
//     let mut w_inner = Tensor::zeros(vec![1]);
//     w_inner.requires_grad = true;
//     let w = Arc::new(RwLock::new(w_inner));

//     // 2. 入力とターゲット (5x1のテンソル)
//     let input = Arc::new(RwLock::new(Tensor::zeros(vec![5])));
//     {
//         let mut inp = input.write().unwrap();
//         inp.data[2] = Complex::new(1.0, 0.0); // 中央に刺激
//     }

//     let lr = 0.01;

//     for _ in 0..1000 {
//         // --- Forward ---
//         // ラプラシアンを計算
//         let diff = {
//             let inp = input.read().unwrap();
//             inp.discrete_laplacian()
//         };
//         let diff_arc = Arc::new(RwLock::new(diff));

//         // 重みを掛ける (簡易的に w * diff)
//         let mut pred = {
//             let d = diff_arc.read().unwrap();
//             let weight = w.read().unwrap().data[0];
//             let mut p = Tensor::zeros(vec![5]);
//             for i in 0..5 {
//                 p.data[i] = d.data[i] * weight;
//             }
//             // 逆伝播のために親を登録 (本来はMul演算の中でやるが、ここでは手動)
//             p.parents = vec![w.clone()];
//             p.op = OpType::Mul;
//             p
//         };

//         // --- Backward ---
//         // 本来は Loss を計算してから backward ですが、
//         // エンジンの速度を見るため、pred に対して直接 backward を呼びます
//         pred.backward();

//         // --- Update ---
//         {
//             let mut w_lock = w.write().unwrap();
//             if let Some(ref g) = w_lock.grad {
//                 let current_w = w_lock.data[0];
//                 w_lock.data[0] = Complex::new(current_w.re - lr * g[0].re, 0.0);
//             }
//             w_lock.grad = None; // 勾配リセット
//         }
//     }

//     let duration = start.elapsed();
//     let response = format!("SIGG Tensor Engine (1000 Ticks) Time: {:?}\n", duration);
//     let _ = stream.write_all(response.as_bytes());
// }