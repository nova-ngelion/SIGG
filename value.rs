
use std::sync::Arc;
use std::time::Instant;

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
pub enum Value {
    Number(f64),
    F32(f32),
    Time(Instant),
    Str(String),
    Grid(GridRef),
    Tuple(Vec<Value>),
    Unit,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(n) => write!(f, "{n}"),
            Value::F32(x) => write!(f, "{x}"),
            Value::Time(_) => write!(f, "<time>"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Grid(g) => write!(f, "Grid(dims={:?}, boundary={:?})", g.dims, g.boundary),
            Value::Tuple(xs) => {
                write!(f, "(")?;
                for (i, v) in xs.iter().enumerate() {
                    if i != 0 { write!(f, ", ")?; }
                    write!(f, "{v}")?;
                }
                write!(f, ")")
            }
            Value::Unit => write!(f, "()"),
        }
    }
}
