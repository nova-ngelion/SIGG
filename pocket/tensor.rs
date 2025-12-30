// src/pocket/tensor.rs

use std::ops::{Add, Sub, Mul};

/// SIGG理論における「内部セル空間」の次元や、場の値を保持する多次元テンソル
/// 論文の $\Psi(n, x)$ のうち、内部セル部分 $l^2(\mathbb{Z}^d)$ を表現します。
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<Complex>, // 基本は複素数体として扱います
}

/// 簡易的な複素数構造体（num_complexクレートを使わない場合）
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    
    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

// 複素数の演算実装 (簡略化)
impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}
impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}
impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Tensor {
    /// ゼロ初期化されたテンソルを作成
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            shape,
            data: vec![Complex::new(0.0, 0.0); size],
        }
    }

    /// フラットなインデックスを計算
    fn get_index(&self, coords: &[usize]) -> Option<usize> {
        if coords.len() != self.shape.len() {
            return None;
        }
        let mut index = 0;
        let mut stride = 1;
        for (i, &dim_size) in self.shape.iter().rev().enumerate() {
            let coord = coords[self.shape.len() - 1 - i];
            if coord >= dim_size {
                return None; // 境界外
            }
            index += coord * stride;
            stride *= dim_size;
        }
        Some(index)
    }

    /// 座標から値を取得
    pub fn get(&self, coords: &[usize]) -> Complex {
        if let Some(idx) = self.get_index(coords) {
            self.data[idx]
        } else {
            Complex::new(0.0, 0.0) // 境界外は0とみなす（ディリクレ境界条件的な扱い）
        }
    }

    /// 座標に値を設定
    pub fn set(&mut self, coords: &[usize], value: Complex) {
        if let Some(idx) = self.get_index(coords) {
            self.data[idx] = value;
        }
    }

    /// SIGG理論: 定義 2.3 離散セル・ラプラシアンの実装
    /// (Delta_cell psi)(n) = sum_{j=1}^{d} [psi(n + e_j) + psi(n - e_j) - 2psi(n)]
    /// 
    pub fn discrete_laplacian(&self) -> Tensor {
        let mut result = Tensor::zeros(self.shape.clone());
        let d = self.shape.len(); // 内部次元 d

        // 全要素に対してラプラシアンを計算
        // (注: 本来は再帰やイテレータでN次元ループを行いますが、ここでは簡略化のため概念を示します)
        // 実装時は `ndarray` クレートのようなイテレータを使用するか、
        // フラットなインデックスから座標を復元して計算します。
        
        for i in 0..self.data.len() {
            let coords = self.index_to_coords(i);
            let center_val = self.data[i];
            let mut sum_neighbors = Complex::new(0.0, 0.0);

            for axis in 0..d {
                // n + e_j (正方向の隣人)
                let mut neighbor_coords_pos = coords.clone();
                if neighbor_coords_pos[axis] + 1 < self.shape[axis] {
                    neighbor_coords_pos[axis] += 1;
                    sum_neighbors = sum_neighbors + self.get(&neighbor_coords_pos);
                }
                
                // n - e_j (負方向の隣人)
                let mut neighbor_coords_neg = coords.clone();
                if neighbor_coords_neg[axis] > 0 {
                    neighbor_coords_neg[axis] -= 1;
                    sum_neighbors = sum_neighbors + self.get(&neighbor_coords_neg);
                }
            }

            // ラプラシアンの計算: 近傍の和 - 2*次元数 * 自分
            // 定義式: psi(n+e) + psi(n-e) - 2psi(n) の総和
            // これは (近傍の総和) - (2 * d * psi(n)) と同値です。
            let laplacian_val = sum_neighbors - (center_val * (2.0 * d as f64));
            
            result.data[i] = laplacian_val;
        }

        result
    }

    /// インデックスから座標への変換ヘルパー
    fn index_to_coords(&self, index: usize) -> Vec<usize> {
        let mut coords = vec![0; self.shape.len()];
        let mut current_idx = index;
        for (i, &dim_size) in self.shape.iter().rev().enumerate() {
            coords[self.shape.len() - 1 - i] = current_idx % dim_size;
            current_idx /= dim_size;
        }
        coords
    }
}