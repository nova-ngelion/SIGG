#[derive(Clone, Debug)]
pub struct Program {
    pub imports: Vec<String>,
    pub namespaces: Vec<NamespaceDef>,
    pub fns: Vec<FnDef>,
}

#[derive(Clone, Debug)]
pub struct NamespaceDef {
    pub name: String,
    pub items: Vec<NamespaceItem>,
}

#[derive(Clone, Debug)]
pub enum NamespaceItem {
    Function(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, TypeAnnotation)>,
}

#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<EnumVariant>,
}

#[derive(Clone, Debug)]
pub struct EnumVariant {
    pub name: String,
    pub data_type: Option<TypeAnnotation>,
}

#[derive(Clone, Debug)]
pub enum TypeAnnotation {
    Simple(String),           // Int, Float, String, etc.
    Generic(String, Vec<TypeAnnotation>),  // Vec<Int>, Map<String, Float>
    Tuple(Vec<TypeAnnotation>),
    Function(Vec<TypeAnnotation>, Box<TypeAnnotation>),
    Optional(Box<TypeAnnotation>),
}

use crate::span::Span;

#[derive(Clone, Debug)]
pub struct FnDef {
    pub namespace: Option<String>,
    pub name: String,
    pub params: Vec<(String, Option<TypeAnnotation>)>,
    pub ret_type: Option<TypeAnnotation>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub enum Pattern {
    Name(String),
    Tuple(Vec<Pattern>),
    Wildcard,
    Struct { name: String, fields: Vec<(String, Pattern)> },
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let { pat: Pattern, type_ann: Option<TypeAnnotation>, expr: Expr },
    Assign { lhs: Expr, expr: Expr },
    Expr(Expr),
    Repeat { count: Expr, body: Vec<Stmt> },
    Transition { count: Expr, body: Vec<Stmt> },
    If { cond: Expr, then_branch: Vec<Stmt>, else_branch: Option<Vec<Stmt>> },
    Block { body: Vec<Stmt> },
    Return { expr: Expr },
    Import { path: String, alias: Option<String> },
    StructDef { name: String, fields: Vec<(String, TypeAnnotation)> },
    EnumDef { name: String, variants: Vec<EnumVariant> },
    MacroDef { name: String, params: Vec<String>, body: Expr },
    NamespaceDef(NamespaceDef),
    /// when cond { ... } (else { ... } optional)
    When {
        cond: Expr,
        then_branch: Vec<Stmt>,
        else_branch: Option<Vec<Stmt>>,
    },
    /// event { when c1 {...} when c2 {...} else {...} }
    /// first-match (else-if chain)
    Event {
        arms: Vec<(Expr, Vec<Stmt>)>,
        else_branch: Option<Vec<Stmt>>,
    },
}

#[derive(Clone, Debug)]
pub enum UnOp { Neg, Not }

#[derive(Clone, Debug)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Gt, Le, Ge,
    BitAnd, And, Or,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Number(f64),
    Str(String),
    Var(String),
    NamespacedVar { namespace: String, name: String },
    Call { callee: Box<Expr>, args: Vec<Expr> },
    Tuple(Vec<Expr>),
    Unary { op: UnOp, rhs: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Group(Box<Expr>),
    If { cond: Box<Expr>, then_branch: Box<Expr>, else_branch: Option<Box<Expr>> },
    Index { expr: Box<Expr>, index: Box<Expr> },
    Field { expr: Box<Expr>, name: String },
    Lambda { params: Vec<(String, Option<TypeAnnotation>)>, body: Vec<Stmt> },
    StructInit { namespace: Option<String>, name: String, fields: Vec<(String, Expr)> },
    EnumInit { namespace: Option<String>, enum_name: String, variant: String, data: Option<Box<Expr>> },
}
