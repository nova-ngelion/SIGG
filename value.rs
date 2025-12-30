use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use crate::bytecode::FnId;
use std::rc::Rc;
use std::cell::RefCell;
use crate::pocket::tensor::Tensor; // 前回のTensorモジュールをインポート

#[derive(Clone, Copy, Debug)]
pub enum Boundary {
    Wrap,
    Clamp,
    Zero,
    Mirror,
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
    pub boundary: Boundary,
}

impl Grid {
    pub fn new(dims: Vec<usize>, fill: f32, boundary: Boundary) -> Self {
        let size = dims.iter().product::<usize>();
        Self { dims, data: vec![fill; size], boundary }
    }

    #[inline]
    pub fn rank(&self) -> usize { self.dims.len() }

    #[inline]
    pub(crate) fn idx2(&self, x: usize, y: usize) -> usize {
        let w = self.dims[0];
        y * w + x
    }

    #[inline]
    fn wrap_i(i: isize, n: usize) -> usize {
        let n = n as isize;
        let mut r = i % n;
        if r < 0 { r += n; }
        r as usize
    }

    pub fn get2(&self, x: isize, y: isize) -> f32 {
        let (w, h) = (self.dims[0], self.dims[1]);
        match self.boundary {
            Boundary::Wrap => {
                let xx = Self::wrap_i(x, w);
                let yy = Self::wrap_i(y, h);
                self.data[self.idx2(xx, yy)]
            }
            _ => unimplemented!("Only Boundary::Wrap is implemented in v0"),
        }
    }

    pub fn set2(&mut self, x: usize, y: usize, v: f32) {
        let idx = self.idx2(x, y);
        self.data[idx] = v;
    }
}

pub type GridRef = Arc<Grid>;

#[derive(Clone, Debug)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Gt, Le, Ge,
    BitAnd, And, Or,
}

// ============================================================================
// 拡張された Value enum（名前空間、FFI、高階関数サポート）
// ============================================================================

pub type HostFn = Arc<dyn Fn(Vec<Value>) -> Result<Value, crate::error::SiggError> + Send + Sync>;

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Number(f64),
    F32(f32),
    Time(Instant),
    Str(String),
    Grid(GridRef),
    Tuple(Vec<Value>),
    Vec(Vec<Value>),
    Handle(u32),
    Unit,
    Bin(Vec<u8>),
    Map(HashMap<String, Value>),
    Lambda(FnId),
    List(Vec<Value>),
    Tensor(Rc<RefCell<Tensor>>),
    Closure {
        fn_id: FnId,
        upvalues: Vec<Value>,
    },
    Function {
        id: FnId,
        name: String,
        params: Vec<String>,
    },
    // ★ HostFunction の func フィールドが Debug を実装していない
    HostFunction {
        name: String,
        func: HostFn,
    },
    Macro {
        name: String,
        params: Vec<String>,
        body: Box<crate::ast::Expr>,
    },
    Struct {
        namespace: Option<String>,
        name: String,
        fields: HashMap<String, Value>,
    },
    EnumVariant {
        namespace: Option<String>,
        enum_name: String,
        variant_name: String,
        data: Option<Box<Value>>,
    },  
    Snapshot {
        timestamp: Instant,
        seed: u64,
        values: Vec<Value>,
        metadata: HashMap<String, String>,
    }, 
    Namespace {
        name: String,
        members: HashMap<String, Value>,
    },
    If {
        cond: Box<Value>,
        then_branch: Box<Value>,
        else_branch: Option<Box<Value>>,
    },
    BinOp {
        op: BinOp,
        left: Box<Value>,
        right: Box<Value>,
    },
}
// ★ 手動で Debug を実装
impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "Int({})", n),
            Value::Float(fl) => write!(f, "Float({})", fl),
            Value::Bool(b) => write!(f, "Bool({})", b),
            Value::Number(n) => write!(f, "Number({})", n),
            Value::F32(x) => write!(f, "F32({})", x),
            Value::Time(_) => write!(f, "Time(..)"),
            Value::Str(s) => write!(f, "Str({:?})", s),
            Value::Grid(g) => write!(f, "Grid({:?})", g),
            Value::Tuple(xs) => f.debug_tuple("Tuple").field(xs).finish(),
            Value::Vec(xs) => f.debug_list().entries(xs).finish(),
            Value::Handle(h) => write!(f, "Handle({})", h),
            Value::Unit => write!(f, "Unit"),
            Value::Bin(b) => write!(f, "Bin(len={})", b.len()),
            Value::Map(m) => write!(f, "Map(len={})", m.len()),
            Value::Function { name, params, id } => {
                f.debug_struct("Function")
                    .field("id", id)
                    .field("name", name)
                    .field("params", params)
                    .finish()
            }
            // ★ HostFunction は func を表示しない
            Value::HostFunction { name, .. } => {
                f.debug_struct("HostFunction")
                    .field("name", name)
                    .field("func", &"<closure>")
                    .finish()
            }
            Value::Macro { name, params, .. } => {
                f.debug_struct("Macro")
                    .field("name", name)
                    .field("params", params)
                    .finish()
            }
            Value::Lambda(id) => write!(f, "Lambda({:?})", id),
            Value::Closure { fn_id, upvalues } => {
                f.debug_struct("Closure")
                    .field("fn_id", fn_id)
                    .field("upvalues", upvalues)
                    .finish()
            }
            Value::Struct { namespace, name, fields } => {
                f.debug_struct("Struct")
                    .field("namespace", namespace)
                    .field("name", name)
                    .field("fields", fields)
                    .finish()
            }
            Value::EnumVariant { namespace, enum_name, variant_name, data } => {
                f.debug_struct("EnumVariant")
                    .field("namespace", namespace)
                    .field("enum_name", enum_name)
                    .field("variant_name", variant_name)
                    .field("data", data)
                    .finish()
            }
            Value::Snapshot { seed, values, .. } => {
                f.debug_struct("Snapshot")
                    .field("seed", seed)
                    .field("values", values)
                    .finish()
            }
            Value::Namespace { name, members } => {
                f.debug_struct("Namespace")
                    .field("name", name)
                    .field("members", members)
                    .finish()
            }
            Value::If { cond, then_branch, else_branch } => {
                f.debug_struct("If")
                    .field("cond", cond)
                    .field("then_branch", then_branch)
                    .field("else_branch", else_branch)
                    .finish()
            }
            Value::BinOp { op, left, right } => {
                f.debug_struct("BinOp")
                    .field("op", op)
                    .field("left", left)
                    .field("right", right)
                    .finish()
            }
            // ★ Listの表示
            Value::List(list) => {
                write!(f, "[")?;
                for (i, v) in list.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            },
            // ★ Tensorの表示 (これを追加しないとエラーになります)
            Value::Tensor(t) => {
                let t = t.borrow();
                write!(f, "<Tensor shape={:?}>", t.shape)
            },
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Number(n) => write!(f, "{}", n),
            Value::F32(x) => write!(f, "{}", x),
            Value::Time(_) => write!(f, "<time>"),
            Value::Str(s) => write!(f, "{}", s),
            Value::Grid(g) => write!(f, "Grid(dims={:?}, boundary={:?})", g.dims, g.boundary),
            Value::Tuple(xs) => {
                write!(f, "(")?;
                for (i, v) in xs.iter().enumerate() {
                    if i != 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            Value::Vec(xs) => {
                write!(f, "[")?;
                for (i, v) in xs.iter().enumerate() {
                    if i != 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Handle(h) => write!(f, "<handle:{}>", h),
            Value::Unit => write!(f, "()"),
            Value::Bin(b) => write!(f, "<bin:{}>", b.len()),
            Value::Map(m) => write!(f, "<map:{}>", m.len()),
            Value::Function { name, params, .. } => {
                write!(f, "<function:{}(", name)?;
                for (i, p) in params.iter().enumerate() {
                    if i != 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ")>")
            }
            Value::HostFunction { name, .. } => write!(f, "<host-fn:{}>", name),
            Value::Macro { name, .. } => write!(f, "<macro:{}>", name),
            Value::Lambda(id) => write!(f, "<lambda:{:?}>", id),
            Value::Closure { fn_id, .. } => write!(f, "<closure:{:?}>", fn_id),
            Value::Struct { namespace, name, fields } => {
                if let Some(ns) = namespace {
                    write!(f, "{}::{} {{", ns, name)?;
                } else {
                    write!(f, "{} {{", name)?;
                }
                for (k, v) in fields {
                    write!(f, " {}: {},", k, v)?;
                }
                write!(f, " }}")
            }
            Value::EnumVariant { namespace, enum_name, variant_name, data } => {
                if let Some(ns) = namespace {
                    write!(f, "{}::{}::{}", ns, enum_name, variant_name)?;
                } else {
                    write!(f, "{}::{}", enum_name, variant_name)?;
                }
                if let Some(d) = data {
                    write!(f, "({})", d)?;
                }
                Ok(())
            }
            Value::Snapshot { seed, values, .. } => {
                write!(f, "<snapshot:seed={},len={}>", seed, values.len())
            }
            Value::Namespace { name, members } => {
                write!(f, "<namespace:{}:{}>", name, members.len())
            }
            Value::If { cond, then_branch, else_branch } => {
                write!(f, "if {} then {} else {}", 
                    cond, 
                    then_branch, 
                    else_branch.as_ref().map_or("()".to_string(), |e| format!("{}", e))
                )
            }
            Value::BinOp { op, left, right } => {
                let op_str = match op {
                    BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
                    BinOp::Mod => "%", BinOp::Eq => "==", BinOp::Ne => "!=",
                    BinOp::Lt => "<", BinOp::Gt => ">", BinOp::Le => "<=", BinOp::Ge => ">=",
                    BinOp::BitAnd => "&", BinOp::And => "&&", BinOp::Or => "||",
                };
                write!(f, "({} {} {})", left, op_str, right)
            }
            Value::Tensor(t) => {
                let t = t.borrow();
                write!(f, "<Tensor shape={:?}>", t.shape)
            },
            _ => write!(f, "<Value>"),
        }
    }
}