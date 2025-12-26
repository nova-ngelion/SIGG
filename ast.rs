
#[derive(Clone, Debug)]
pub struct Program {
    pub fns: Vec<FnDef>,
}

#[derive(Clone, Debug)]
pub struct FnDef {
    pub name: String,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Name(String),
    Tuple(Vec<Pattern>),
    Wildcard,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let { pat: Pattern, expr: Expr },
    Assign { name: String, expr: Expr },
    Expr(Expr),
    Repeat { count: Expr, body: Vec<Stmt> },
    Transition { count: Expr, body: Vec<Stmt> },
}

#[derive(Clone, Debug)]
pub enum UnOp { Neg }

#[derive(Clone, Debug)]
pub enum BinOp { Add, Sub, Mul, Div }

#[derive(Clone, Debug)]
pub enum Expr {
    Number(f64),
    Str(String),
    Var(String),
    Call { callee: String, args: Vec<Expr> },
    Tuple(Vec<Expr>),
    Unary { op: UnOp, rhs: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Group(Box<Expr>),
}
