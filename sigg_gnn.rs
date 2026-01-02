// src/sigg_gnn.rs
// SIGG理論に基づく多層グラフニューラルネットワーク
// 内部次元（セル空間）を持つGNN

use crate::error::SiggError;
use crate::pocket::tensor::Tensor;
use std::collections::HashMap;

/// SIGG-GNNノード: 通常の特徴量 + 内部セル次元
#[derive(Debug, Clone)]
pub struct SiggNode {
    pub id: usize,
    pub features: Vec<f32>,          // 通常の特徴量
    pub cell_state: Vec<Vec<f32>>,   // 内部セル次元の状態 [cell_idx][feature]
    pub cell_dim: usize,             // 内部セル次元数 d
}

impl SiggNode {
    pub fn new(id: usize, feature_dim: usize, cell_dim: usize) -> Self {
        Self {
            id,
            features: vec![0.0; feature_dim],
            cell_state: vec![vec![0.0; feature_dim]; cell_dim],
            cell_dim,
        }
    }

    /// セル・ラプラシアンを適用（SIGG理論の核心）
    pub fn apply_cell_laplacian(&self) -> Vec<f32> {
        let mut result = vec![0.0; self.features.len()];
        
        for i in 0..self.features.len() {
            let mut laplacian = 0.0;
            
            // 離散ラプラシアン: Σ[ψ(n+ê) + ψ(n-ê) - 2ψ(n)]
            for j in 0..self.cell_dim {
                let next = if j + 1 < self.cell_dim {
                    self.cell_state[j + 1][i]
                } else {
                    0.0
                };
                
                let prev = if j > 0 {
                    self.cell_state[j - 1][i]
                } else {
                    0.0
                };
                
                laplacian += next + prev - 2.0 * self.cell_state[j][i];
            }
            
            result[i] = laplacian;
        }
        
        result
    }

    /// セル空間スペクトル（固有値）
    pub fn cell_spectrum(&self) -> Vec<f32> {
        let mut spectrum = Vec::new();
        
        for k in 0..self.cell_dim {
            // λ(k) = 2d(1 - cos(2πk/d))
            let theta = 2.0 * std::f32::consts::PI * (k as f32) / (self.cell_dim as f32);
            let eigenvalue = 2.0 * (self.cell_dim as f32) * (1.0 - theta.cos());
            spectrum.push(eigenvalue);
        }
        
        spectrum
    }
}

/// エッジ: 物理空間での接続 + 内部セル間相互作用
#[derive(Debug, Clone)]
pub struct SiggEdge {
    pub from: usize,
    pub to: usize,
    pub weight: f32,
    pub cell_coupling: Vec<Vec<f32>>, // セル間結合 [from_cell][to_cell]
}

/// SIGG-GNN全体
pub struct SiggGnn {
    pub nodes: HashMap<usize, SiggNode>,
    pub edges: Vec<SiggEdge>,
    pub feature_dim: usize,
    pub cell_dim: usize,
    pub layers: Vec<GnnLayer>,
}

#[derive(Debug, Clone)]
pub struct GnnLayer {
    pub weights: Vec<Vec<f32>>,      // [out_dim][in_dim]
    pub cell_weights: Vec<Vec<f32>>, // セル空間用の重み
    pub bias: Vec<f32>,
}

impl GnnLayer {
    pub fn new(in_dim: usize, out_dim: usize, cell_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let weights: Vec<Vec<f32>> = (0..out_dim)
            .map(|_| {
                (0..in_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();
        
        let cell_weights: Vec<Vec<f32>> = (0..cell_dim)
            .map(|_| {
                (0..cell_dim)
                    .map(|_| rng.gen_range(-0.05..0.05))
                    .collect()
            })
            .collect();
        
        let bias = vec![0.0; out_dim];
        
        Self {
            weights,
            cell_weights,
            bias,
        }
    }
}

impl SiggGnn {
    pub fn new(feature_dim: usize, cell_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let mut layers = Vec::new();
        
        let mut prev_dim = feature_dim;
        for &dim in &hidden_dims {
            layers.push(GnnLayer::new(prev_dim, dim, cell_dim));
            prev_dim = dim;
        }
        
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            feature_dim,
            cell_dim,
            layers,
        }
    }

    pub fn add_node(&mut self, node: SiggNode) {
        self.nodes.insert(node.id, node);
    }

    pub fn add_edge(&mut self, edge: SiggEdge) {
        self.edges.push(edge);
    }

    /// メッセージパッシング（物理空間 + 内部セル空間）
    pub fn message_passing(&mut self) -> Result<(), SiggError> {
        let node_ids: Vec<usize> = self.nodes.keys().copied().collect();
        
        // 各ノードの新しい状態を計算
        let mut new_states: HashMap<usize, (Vec<f32>, Vec<Vec<f32>>)> = HashMap::new();
        
        for &node_id in &node_ids {
            let node = self.nodes.get(&node_id).unwrap();
            
            // 隣接ノードからのメッセージを集約
            let mut aggregated = vec![0.0; self.feature_dim];
            let mut aggregated_cells = vec![vec![0.0; self.feature_dim]; self.cell_dim];
            let mut neighbor_count = 0;
            
            for edge in &self.edges {
                if edge.to == node_id {
                    if let Some(neighbor) = self.nodes.get(&edge.from) {
                        // 物理空間のメッセージ
                        for i in 0..self.feature_dim {
                            aggregated[i] += neighbor.features[i] * edge.weight;
                        }
                        
                        // 内部セル空間のメッセージ
                        for c in 0..self.cell_dim {
                            for i in 0..self.feature_dim {
                                aggregated_cells[c][i] += neighbor.cell_state[c][i] * edge.weight;
                            }
                        }
                        
                        neighbor_count += 1;
                    }
                }
            }
            
            if neighbor_count > 0 {
                let count_f = neighbor_count as f32;
                for i in 0..self.feature_dim {
                    aggregated[i] /= count_f;
                }
                for c in 0..self.cell_dim {
                    for i in 0..self.feature_dim {
                        aggregated_cells[c][i] /= count_f;
                    }
                }
            }
            
            // セル・ラプラシアンの効果を追加
            let laplacian = node.apply_cell_laplacian();
            for i in 0..self.feature_dim {
                aggregated[i] += 0.1 * laplacian[i]; // スケーリング係数
            }
            
            // 自己状態と結合
            for i in 0..self.feature_dim {
                aggregated[i] = 0.5 * aggregated[i] + 0.5 * node.features[i];
            }
            
            new_states.insert(node_id, (aggregated, aggregated_cells));
        }
        
        // 状態を更新
        for (node_id, (new_features, new_cells)) in new_states {
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.features = new_features;
                node.cell_state = new_cells;
            }
        }
        
        Ok(())
    }

    /// レイヤーを通した順伝播
    pub fn forward(&mut self, iterations: usize) -> Result<(), SiggError> {
        for _ in 0..iterations {
            self.message_passing()?;
            
            // 活性化関数を適用（ReLU）
            for node in self.nodes.values_mut() {
                for i in 0..node.features.len() {
                    node.features[i] = node.features[i].max(0.0);
                }
                
                for c in 0..node.cell_dim {
                    for i in 0..node.features.len() {
                        node.cell_state[c][i] = node.cell_state[c][i].max(0.0);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// ダークマターモード検出（k≠0のセル励起）
    pub fn detect_dark_modes(&self) -> Vec<(usize, Vec<f32>)> {
        let mut dark_modes = Vec::new();
        
        for (node_id, node) in &self.nodes {
            // k≠0モードのエネルギーを計算
            let spectrum = node.cell_spectrum();
            let mut dark_energy = 0.0;
            
            for (k, eigenval) in spectrum.iter().enumerate() {
                if k > 0 { // k≠0
                    // このモードのエネルギー
                    let mut mode_energy = 0.0;
                    for i in 0..node.features.len() {
                        mode_energy += node.cell_state[k][i].powi(2);
                    }
                    dark_energy += eigenval * mode_energy;
                }
            }
            
            if dark_energy > 0.01 {
                dark_modes.push((*node_id, spectrum));
            }
        }
        
        dark_modes
    }

    /// ニュートリノ質量的特徴（広がったセルプロファイル）
    pub fn measure_neutrino_features(&self) -> HashMap<usize, f32> {
        let mut features = HashMap::new();
        
        for (node_id, node) in &self.nodes {
            // セル空間での広がり度を測定
            let mut spread = 0.0;
            for c in 0..node.cell_dim {
                for i in 0..node.features.len() {
                    spread += node.cell_state[c][i].abs();
                }
            }
            spread /= (node.cell_dim * node.features.len()) as f32;
            
            features.insert(*node_id, spread);
        }
        
        features
    }
}

/// SIGG-GNN + CNN ハイブリッドアーキテクチャ
pub struct SiggHybridNet {
    pub gnn: SiggGnn,
    pub cnn_kernels: Vec<Vec<Vec<f32>>>, // [layer][kernel][weights]
    pub feature_maps: Vec<Vec<Vec<f32>>>, // CNN特徴マップ
}

impl SiggHybridNet {
    pub fn new(
        feature_dim: usize,
        cell_dim: usize,
        gnn_layers: Vec<usize>,
        cnn_filters: Vec<usize>,
    ) -> Self {
        let gnn = SiggGnn::new(feature_dim, cell_dim, gnn_layers);
        
        // CNN カーネル初期化
        let mut cnn_kernels = Vec::new();
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for &filter_count in &cnn_filters {
            let mut layer_kernels = Vec::new();
            for _ in 0..filter_count {
                let kernel: Vec<f32> = (0..9)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect();
                layer_kernels.push(kernel);
            }
            cnn_kernels.push(layer_kernels);
        }
        
        Self {
            gnn,
            cnn_kernels,
            feature_maps: Vec::new(),
        }
    }

    /// ハイブリッド学習: GNN特徴 → CNN処理
    pub fn hybrid_forward(&mut self, gnn_iterations: usize) -> Result<Vec<f32>, SiggError> {
        // 1. GNN処理
        self.gnn.forward(gnn_iterations)?;
        
        // 2. GNN出力を2D特徴マップに変換
        let node_count = self.gnn.nodes.len();
        let side = (node_count as f32).sqrt().ceil() as usize;
        
        let mut feature_map = vec![vec![0.0; side]; side];
        for (idx, (_, node)) in self.gnn.nodes.iter().enumerate() {
            let row = idx / side;
            let col = idx % side;
            if row < side && col < side {
                // ノードの特徴量を平均化
                let avg: f32 = node.features.iter().sum::<f32>() / node.features.len() as f32;
                feature_map[row][col] = avg;
            }
        }
        
        // 3. CNN処理（簡易版）
        let mut output = Vec::new();
        for kernel_layer in &self.cnn_kernels {
            for kernel in kernel_layer {
                let mut activation = 0.0;
                // 3x3畳み込み
                for i in 0..(side - 2) {
                    for j in 0..(side - 2) {
                        let mut sum = 0.0;
                        for ki in 0..3 {
                            for kj in 0..3 {
                                sum += feature_map[i + ki][j + kj] * kernel[ki * 3 + kj];
                            }
                        }
                        activation += sum.max(0.0); // ReLU
                    }
                }
                output.push(activation);
            }
        }
        
        Ok(output)
    }
}