use std::fmt;
use std::ops::{Add, Mul, Sub, Div};
use std::sync::{Arc, RwLock};

// ==========================================
// 複素数構造体 (変更なし)
// ==========================================
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

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        let denom = other.re * other.re + other.im * other.im;
        if denom == 0.0 {
            // ゼロ除算回避 (簡易)
            return Complex::new(0.0, 0.0);
        }
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.2}+{:.2}i", self.re, self.im)
    }
}

// ==========================================
// Autograd用 定義
// ==========================================

#[derive(Clone, Debug)]
pub enum OpType {
    Leaf,
    Add,
    Sub,
    Mul,
    Div, // 割り算
    Rem, // 剰余
    Neg, // 単項マイナス
    Exp,
    Laplacian,
}

pub struct Tensor {
    pub data: Vec<Complex>,
    pub shape: Vec<usize>,
    pub grad: Option<Vec<Complex>>,
    
    pub requires_grad: bool,
    pub op: OpType,
    pub parents: Vec<Arc<RwLock<Tensor>>>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Tensor {
            data: vec![Complex::new(0.0, 0.0); size],
            shape,
            grad: None,
            requires_grad: false,
            op: OpType::Leaf,
            parents: Vec::new(),
        }
    }

    pub fn set(&mut self, indices: &[usize], value: Complex) {
        let idx = self.get_index(indices);
        if idx < self.data.len() {
            self.data[idx] = value;
        }
    }

    pub fn get(&self, indices: &[usize]) -> Complex {
        let idx = self.get_index(indices);
        if idx < self.data.len() {
            self.data[idx]
        } else {
            Complex::new(0.0, 0.0)
        }
    }

    fn get_index(&self, indices: &[usize]) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        for (i, &dim_idx) in indices.iter().rev().zip(self.shape.iter().rev()) {
            idx += i * stride;
            stride *= dim_idx;
        }
        idx
    }

    // ラプラシアンフィルタ (以前追加したもの)
    pub fn discrete_laplacian(&self) -> Self {
        let mut out = Tensor::zeros(self.shape.clone());
        let ndim = self.shape.len();

        for i in 0..self.data.len() {
            let mut coords = vec![0; ndim];
            let mut temp_i = i;
            for d in (0..ndim).rev() {
                coords[d] = temp_i % self.shape[d];
                temp_i /= self.shape[d];
            }

            let center = self.data[i];
            let mut sum_neighbors = Complex::new(0.0, 0.0);
            let mut neighbor_count = 0;

            for d in 0..ndim {
                if coords[d] + 1 < self.shape[d] {
                    let mut neighbor_coords = coords.clone();
                    neighbor_coords[d] += 1;
                    let idx = self.get_index(&neighbor_coords);
                    sum_neighbors = sum_neighbors + self.data[idx];
                    neighbor_count += 1;
                }
                if coords[d] > 0 {
                    let mut neighbor_coords = coords.clone();
                    neighbor_coords[d] -= 1;
                    let idx = self.get_index(&neighbor_coords);
                    sum_neighbors = sum_neighbors + self.data[idx];
                    neighbor_count += 1;
                }
            }

            let val = sum_neighbors - center * Complex::new(neighbor_count as f64, 0.0);
            out.data[i] = val;
        }
        out
    }

    // --- Autograd Graph Operations ---

    pub fn add_graph(lhs: Arc<RwLock<Tensor>>, rhs: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (lhs_data, lhs_shape, lhs_req) = {
            let r = lhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };
        let (rhs_data, rhs_shape, rhs_req) = {
            let r = rhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };

        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in add: {:?} vs {:?}", lhs_shape, rhs_shape);
        }

        let new_data: Vec<Complex> = lhs_data.iter().zip(rhs_data.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape: lhs_shape,
            grad: None,
            requires_grad: lhs_req || rhs_req,
            op: OpType::Add,
            parents: vec![lhs, rhs],
        }))
    }

    pub fn sub_graph(lhs: Arc<RwLock<Tensor>>, rhs: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (lhs_data, lhs_shape, lhs_req) = {
            let r = lhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };
        let (rhs_data, rhs_shape, rhs_req) = {
            let r = rhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };

        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in sub: {:?} vs {:?}", lhs_shape, rhs_shape);
        }

        let new_data: Vec<Complex> = lhs_data.iter().zip(rhs_data.iter())
            .map(|(a, b)| *a - *b)
            .collect();

        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape: lhs_shape,
            grad: None,
            requires_grad: lhs_req || rhs_req,
            op: OpType::Sub,
            parents: vec![lhs, rhs],
        }))
    }

    pub fn mul_graph(lhs: Arc<RwLock<Tensor>>, rhs: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (lhs_data, lhs_shape, lhs_req) = {
            let r = lhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };
        let (rhs_data, rhs_shape, rhs_req) = {
            let r = rhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };

        // ★改良: スカラ放送 (Broadcasting) 対応
        let (new_data, new_shape) = if lhs_shape == rhs_shape {
            // サイズが同じ場合 (これまで通り)
            let data = lhs_data.iter().zip(rhs_data.iter()).map(|(a, b)| *a * *b).collect();
            (data, lhs_shape)
        } else if lhs_shape.len() == 1 && lhs_shape[0] == 1 {
            // 左がスカラ [1]、右がベクトル [N]
            let s = lhs_data[0];
            let data = rhs_data.iter().map(|b| s * *b).collect();
            (data, rhs_shape)
        } else if rhs_shape.len() == 1 && rhs_shape[0] == 1 {
            // 左がベクトル [N]、右がスカラ [1]
            let s = rhs_data[0];
            let data = lhs_data.iter().map(|a| *a * s).collect();
            (data, lhs_shape)
        } else {
            panic!("Shape mismatch in mul: {:?} vs {:?}", lhs_shape, rhs_shape);
        };

        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape: new_shape,
            grad: None,
            requires_grad: lhs_req || rhs_req,
            op: OpType::Mul,
            parents: vec![lhs, rhs],
        }))
    }

    pub fn div_graph(lhs: Arc<RwLock<Tensor>>, rhs: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (lhs_data, lhs_shape, lhs_req) = {
            let r = lhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };
        let (rhs_data, rhs_shape, rhs_req) = {
            let r = rhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };

        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in div: {:?} vs {:?}", lhs_shape, rhs_shape);
        }

        let new_data: Vec<Complex> = lhs_data.iter().zip(rhs_data.iter())
            .map(|(a, b)| *a / *b)
            .collect();

        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape: lhs_shape,
            grad: None,
            requires_grad: lhs_req || rhs_req,
            op: OpType::Div,
            parents: vec![lhs, rhs],
        }))
    }

    pub fn rem_graph(lhs: Arc<RwLock<Tensor>>, rhs: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (lhs_data, lhs_shape, _) = {
            let r = lhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };
        let (rhs_data, rhs_shape, _) = {
            let r = rhs.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };

        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in rem: {:?} vs {:?}", lhs_shape, rhs_shape);
        }

        let new_data: Vec<Complex> = lhs_data.iter().zip(rhs_data.iter())
            .map(|(a, b)| {
                // 実部のみ剰余をとる簡易実装
                if b.re == 0.0 { Complex::new(0.0, 0.0) } 
                else { Complex::new(a.re % b.re, 0.0) }
            })
            .collect();

        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape: lhs_shape,
            grad: None,
            requires_grad: false, // 勾配計算なし
            op: OpType::Rem,
            parents: Vec::new(),
        }))
    }

    pub fn neg_graph(arg: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (data, shape, req) = {
            let r = arg.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };
        let new_data: Vec<Complex> = data.iter().map(|a| Complex::new(-a.re, -a.im)).collect();
        
        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape,
            grad: None,
            requires_grad: req,
            op: OpType::Neg,
            parents: vec![arg],
        }))
    }
    pub fn exp_graph(arg: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (data, shape, req) = {
            let r = arg.read().unwrap();
            (r.data.clone(), r.shape.clone(), r.requires_grad)
        };

        // Complexのexp: e^(a+bi) = e^a * (cos(b) + i*sin(b))
        // RustのComplex型で計算させるため、一度Complex型として計算します
        let new_data: Vec<Complex> = data.iter().map(|z| {
             // 自作Complexにメソッドがない場合は手動計算、あるいはnum_complexを使う手もありますが
             // ここでは簡易的に実装します
             let r = z.re;
             let i = z.im;
             let exp_r = r.exp();
             Complex::new(exp_r * i.cos(), exp_r * i.sin())
        }).collect();

        Arc::new(RwLock::new(Tensor {
            data: new_data,
            shape,
            grad: None,
            requires_grad: req,
            op: OpType::Exp, // ★OpType::Exp
            parents: vec![arg],
        }))
    }
    pub fn laplacian_graph(arg: Arc<RwLock<Tensor>>) -> Arc<RwLock<Tensor>> {
        let (out_tensor, shape, req) = {
            let r = arg.read().unwrap();
            // 既存の discrete_laplacian を呼んでデータ計算
            let out = r.discrete_laplacian(); 
            (out, r.shape.clone(), r.requires_grad)
        };

        Arc::new(RwLock::new(Tensor {
            data: out_tensor.data,
            shape: shape,
            grad: None,
            requires_grad: req,
            op: OpType::Laplacian, // ★OpType::Laplacian
            parents: vec![arg],
        }))
    }

    // ==========================================
    // Backward Logic
    // ==========================================
    
    pub fn backward(&mut self) {
        if self.grad.is_none() {
            let size = self.data.len();
            self.grad = Some(vec![Complex::new(1.0, 0.0); size]);
        }
        self.propagate();
    }

    fn propagate(&mut self) {
        if self.grad.is_none() { return; }
        let my_grad = self.grad.as_ref().unwrap().clone();

        match self.op {
            OpType::Leaf | OpType::Rem => {},
            
            OpType::Add => {
                for parent_arc in &self.parents {
                    let mut parent = parent_arc.write().unwrap();
                    if parent.requires_grad {
                        parent.accumulate_grad(&my_grad);
                        parent.propagate();
                    }
                }
            }
            OpType::Sub => {
                if self.parents.len() == 2 {
                    // Left
                    {
                        let mut left = self.parents[0].write().unwrap();
                        if left.requires_grad {
                            left.accumulate_grad(&my_grad);
                            left.propagate();
                        }
                    }
                    // Right (-1)
                    {
                        let mut right = self.parents[1].write().unwrap();
                        if right.requires_grad {
                            let neg_grad: Vec<Complex> = my_grad.iter().map(|g| Complex::new(-g.re, -g.im)).collect();
                            right.accumulate_grad(&neg_grad);
                            right.propagate();
                        }
                    }
                }
            }
            OpType::Mul => {
                if self.parents.len() == 2 {
                    // 親のデータを取得（ロック時間を最小にするためクローン）
                    let left_data = self.parents[0].read().unwrap().data.clone();
                    let left_shape = self.parents[0].read().unwrap().shape.clone();
                    
                    let right_data = self.parents[1].read().unwrap().data.clone();
                    let right_shape = self.parents[1].read().unwrap().shape.clone();
                    
                    // --- Left (a) への逆伝播 ---
                    // z = a * b
                    // もし a, b が同サイズなら dz/da = b * grad
                    // もし a がスカラなら、 dz/da = sum(b * grad)
                    {
                        let mut left = self.parents[0].write().unwrap();
                        if left.requires_grad {
                            // 計算ロジック
                            let is_scalar = left_shape.len() == 1 && left_shape[0] == 1;
                            let partner_is_scalar = right_shape.len() == 1 && right_shape[0] == 1;

                            if !is_scalar && !partner_is_scalar {
                                // [N] * [N] -> 通常
                                let g: Vec<Complex> = my_grad.iter().zip(right_data.iter())
                                    .map(|(dy, r)| *dy * *r).collect();
                                left.accumulate_grad(&g);
                                left.propagate();
                            } else if is_scalar {
                                // [1] * [N] -> スカラ側への勾配は内積(合計)
                                // grad = sum( my_grad[i] * right_data[i] )
                                let mut sum = Complex::new(0.0, 0.0);
                                for (dy, r) in my_grad.iter().zip(right_data.iter()) {
                                    sum = sum + (*dy * *r);
                                }
                                left.accumulate_grad(&[sum]); // スカラとして加算
                                left.propagate();
                            } else {
                                // [N] * [1] -> ベクトル側への勾配
                                // grad[i] = my_grad[i] * right_scalar
                                let s = right_data[0];
                                let g: Vec<Complex> = my_grad.iter().map(|dy| *dy * s).collect();
                                left.accumulate_grad(&g);
                                left.propagate();
                            }
                        }
                    }
                    
                    // --- Right (b) への逆伝播 ---
                    {
                        let mut right = self.parents[1].write().unwrap();
                        if right.requires_grad {
                            let is_scalar = right_shape.len() == 1 && right_shape[0] == 1;
                            let partner_is_scalar = left_shape.len() == 1 && left_shape[0] == 1;

                            if !is_scalar && !partner_is_scalar {
                                // [N] * [N]
                                let g: Vec<Complex> = my_grad.iter().zip(left_data.iter())
                                    .map(|(dy, l)| *dy * *l).collect();
                                right.accumulate_grad(&g);
                                right.propagate();
                            } else if is_scalar {
                                // [N] * [1] -> スカラ側への勾配は内積
                                let mut sum = Complex::new(0.0, 0.0);
                                for (dy, l) in my_grad.iter().zip(left_data.iter()) {
                                    sum = sum + (*dy * *l);
                                }
                                right.accumulate_grad(&[sum]);
                                right.propagate();
                            } else {
                                // [1] * [N] -> ベクトル側
                                let s = left_data[0];
                                let g: Vec<Complex> = my_grad.iter().map(|dy| *dy * s).collect();
                                right.accumulate_grad(&g);
                                right.propagate();
                            }
                        }
                    }
                }
            }
            OpType::Div => {
                if self.parents.len() == 2 {
                    let a_data = self.parents[0].read().unwrap().data.clone();
                    let b_data = self.parents[1].read().unwrap().data.clone();

                    // Left (a): grad * 1/b
                    {
                        let mut left = self.parents[0].write().unwrap();
                        if left.requires_grad {
                            let g: Vec<Complex> = my_grad.iter().zip(b_data.iter())
                                .map(|(dy, b)| *dy / *b).collect();
                            left.accumulate_grad(&g);
                            left.propagate();
                        }
                    }

                    // Right (b): grad * -a/b^2
                    {
                        let mut right = self.parents[1].write().unwrap();
                        if right.requires_grad {
                            let g: Vec<Complex> = my_grad.iter().zip(a_data.iter()).zip(b_data.iter())
                                .map(|((dy, a), b)| {
                                    let b2 = *b * *b;
                                    let term = *a / b2;
                                    *dy * Complex::new(-term.re, -term.im)
                                }).collect();
                            right.accumulate_grad(&g);
                            right.propagate();
                        }
                    }
                }
            }
            OpType::Neg => {
                if self.parents.len() == 1 {
                    let mut parent = self.parents[0].write().unwrap();
                    if parent.requires_grad {
                        let g: Vec<Complex> = my_grad.iter().map(|y| Complex::new(-y.re, -y.im)).collect();
                        parent.accumulate_grad(&g);
                        parent.propagate();
                    }
                }
            }
            OpType::Exp => {
                if self.parents.len() == 1 {
                    let mut parent = self.parents[0].write().unwrap();
                    if parent.requires_grad {
                        // 微分は自分自身の値(y) * grad
                        let my_val = self.data.clone(); // y = e^x
                        let new_grad: Vec<Complex> = my_grad.iter().zip(my_val.iter())
                            .map(|(g, y)| *g * *y)
                            .collect();
                        
                        parent.accumulate_grad(&new_grad);
                        parent.propagate();
                    }
                }
            }
            OpType::Laplacian => {
                if self.parents.len() == 1 {
                    let mut parent = self.parents[0].write().unwrap();
                    if parent.requires_grad {
                        // 勾配の逆伝播:
                        // dE/dx = Laplacian(dE/dy)
                        // (ラプラシアン行列は対称なので転置しても同じ)
                        
                        // 1. my_grad を Tensor 化してラプラシアンを計算させる
                        //    (少し非効率ですが、discrete_laplacianの実装を再利用するため)
                        let grad_tensor = Tensor {
                            data: my_grad,
                            shape: parent.shape.clone(),
                            grad: None, requires_grad: false, op: OpType::Leaf, parents: vec![]
                        };
                        let grad_lap = grad_tensor.discrete_laplacian();
                        
                        // 2. 結果を親の勾配に加算
                        parent.accumulate_grad(&grad_lap.data);
                        parent.propagate();
                    }
                }
            }
        }
    }

    pub fn accumulate_grad(&mut self, grad_to_add: &[Complex]) {
        if self.grad.is_none() {
            self.grad = Some(grad_to_add.to_vec());
        } else {
            let my_grad = self.grad.as_mut().unwrap();
            for (i, g) in grad_to_add.iter().enumerate() {
                if i < my_grad.len() {
                    my_grad[i] = my_grad[i] + *g;
                }
            }
        }
    }
}