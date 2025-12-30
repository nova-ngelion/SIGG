use std::collections::HashMap;
use crate::ast::*;
use crate::span::Span;

// ★ テストコードで必要な import を追加
#[cfg(test)]
use crate::parser::Parser;
#[cfg(test)]
use crate::bytecode::compile_with_namespaces;

pub struct TypeChecker {
    env: HashMap<String, Type>,
    namespaces: HashMap<String, NamespaceTypes>,
    errors: Vec<TypeError>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    Int,
    Float,
    Bool,
    String,
    Unit,
    Vec(Box<Type>),
    Map(Box<Type>, Box<Type>),
    Tuple(Vec<Type>),
    Function(Vec<Type>, Box<Type>),
    Struct(String, HashMap<String, Type>),
    Enum(String, Vec<EnumVariantType>),
    Generic(String, Vec<Type>),
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnumVariantType {
    pub name: String,
    pub data: Option<Type>,
}

#[derive(Clone, Debug)]
pub struct NamespaceTypes {
    pub functions: HashMap<String, Type>,
    pub structs: HashMap<String, Type>,
    pub enums: HashMap<String, Type>,
}

#[derive(Clone, Debug)]
pub struct TypeError {
    pub message: String,
    pub span: Option<Span>,
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut env = HashMap::new();
        
        // 組み込み関数の型を登録
        env.insert("print".to_string(), Type::Function(
            vec![Type::Unknown],
            Box::new(Type::Unit),
        ));
        
        env.insert("vec_map".to_string(), Type::Function(
            vec![
                Type::Vec(Box::new(Type::Generic("T".to_string(), vec![]))),
                Type::Function(
                    vec![Type::Generic("T".to_string(), vec![])],
                    Box::new(Type::Generic("U".to_string(), vec![])),
                ),
            ],
            Box::new(Type::Vec(Box::new(Type::Generic("U".to_string(), vec![])))),
        ));
        
        Self {
            env,
            namespaces: HashMap::new(),
            errors: Vec::new(),
        }
    }
    
    pub fn check_program(&mut self, prog: &Program) -> Result<(), Vec<TypeError>> {
        // 名前空間の型を登録
        for ns in &prog.namespaces {
            self.register_namespace(ns)?;
        }
        
        // 関数の型をチェック
        for func in &prog.fns {
            self.check_function(func)?;
        }
        
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }
    
    fn register_namespace(&mut self, ns: &NamespaceDef) -> Result<(), Vec<TypeError>> {
        let mut ns_types = NamespaceTypes {
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
        };
        
        for item in &ns.items {
            match item {
                NamespaceItem::Function(fn_def) => {
                    let fn_type = self.function_type(fn_def);
                    ns_types.functions.insert(fn_def.name.clone(), fn_type);
                }
                NamespaceItem::Struct(struct_def) => {
                    let struct_type = self.struct_type(struct_def);
                    ns_types.structs.insert(struct_def.name.clone(), struct_type);
                }
                NamespaceItem::Enum(enum_def) => {
                    let enum_type = self.enum_type(enum_def);
                    ns_types.enums.insert(enum_def.name.clone(), enum_type);
                }
            }
        }
        
        self.namespaces.insert(ns.name.clone(), ns_types);
        Ok(())
    }
    
    fn function_type(&self, fn_def: &FnDef) -> Type {
        let param_types: Vec<Type> = fn_def.params.iter()
            .map(|(_, ty_ann)| {
                ty_ann.as_ref()
                    .map(|t| self.annotation_to_type(t))
                    .unwrap_or(Type::Unknown)
            })
            .collect();
        
        let ret_type = fn_def.ret_type.as_ref()
            .map(|t| self.annotation_to_type(t))
            .unwrap_or(Type::Unit);
        
        Type::Function(param_types, Box::new(ret_type))
    }
    
    fn struct_type(&self, struct_def: &StructDef) -> Type {
        let fields: HashMap<String, Type> = struct_def.fields.iter()
            .map(|(name, ty_ann)| {
                (name.clone(), self.annotation_to_type(ty_ann))
            })
            .collect();
        
        Type::Struct(struct_def.name.clone(), fields)
    }
    
    fn enum_type(&self, enum_def: &EnumDef) -> Type {
        let variants: Vec<EnumVariantType> = enum_def.variants.iter()
            .map(|v| EnumVariantType {
                name: v.name.clone(),
                data: v.data_type.as_ref().map(|t| self.annotation_to_type(t)),
            })
            .collect();
        
        Type::Enum(enum_def.name.clone(), variants)
    }
    
    fn annotation_to_type(&self, ann: &TypeAnnotation) -> Type {
        match ann {
            TypeAnnotation::Simple(name) => {
                match name.as_str() {
                    "Int" => Type::Int,
                    "Float" => Type::Float,
                    "Bool" => Type::Bool,
                    "String" => Type::String,
                    "Unit" => Type::Unit,
                    _ => Type::Unknown,
                }
            }
            TypeAnnotation::Generic(name, args) => {
                match name.as_str() {
                    "Vec" if args.len() == 1 => {
                        Type::Vec(Box::new(self.annotation_to_type(&args[0])))
                    }
                    "Map" if args.len() == 2 => {
                        Type::Map(
                            Box::new(self.annotation_to_type(&args[0])),
                            Box::new(self.annotation_to_type(&args[1])),
                        )
                    }
                    _ => Type::Generic(name.clone(), 
                        args.iter().map(|a| self.annotation_to_type(a)).collect()),
                }
            }
            TypeAnnotation::Tuple(types) => {
                Type::Tuple(types.iter().map(|t| self.annotation_to_type(t)).collect())
            }
            TypeAnnotation::Function(params, ret) => {
                Type::Function(
                    params.iter().map(|p| self.annotation_to_type(p)).collect(),
                    Box::new(self.annotation_to_type(ret)),
                )
            }
            TypeAnnotation::Optional(inner) => {
                // Option<T> は Enum として扱う
                Type::Enum("Option".to_string(), vec![
                    EnumVariantType { name: "None".to_string(), data: None },
                    EnumVariantType { 
                        name: "Some".to_string(), 
                        data: Some(self.annotation_to_type(inner)),
                    },
                ])
            }
        }
    }
    
    fn check_function(&mut self, fn_def: &FnDef) -> Result<(), Vec<TypeError>> {
        // パラメータを環境に追加
        let mut local_env = self.env.clone();
        
        for (param_name, param_type) in &fn_def.params {
            if let Some(ty_ann) = param_type {
                let ty = self.annotation_to_type(ty_ann);
                local_env.insert(param_name.clone(), ty);
            }
        }
        
        // 関数本体をチェック
        let old_env = std::mem::replace(&mut self.env, local_env);
        
        for stmt in &fn_def.body {
            self.check_stmt(stmt)?;
        }
        
        self.env = old_env;
        Ok(())
    }
    
    fn check_stmt(&mut self, stmt: &Stmt) -> Result<(), Vec<TypeError>> {
        match stmt {
            Stmt::Assign { lhs, expr } => {
                let _lhs_type = self.infer_expr(lhs)?;
                let _rhs_type = self.infer_expr(expr)?;
                // 型の互換性チェック
                Ok(())
            }
            Stmt::Expr(e) => {
                self.infer_expr(e)?;
                Ok(())
            }
            Stmt::If { cond, then_branch, else_branch } => {
                let cond_type = self.infer_expr(cond)?;
                if cond_type != Type::Bool && cond_type != Type::Unknown {
                    self.errors.push(TypeError {
                        message: format!("Condition must be Bool, got {:?}", cond_type),
                        span: None,
                    });
                }
                
                for stmt in then_branch {
                    self.check_stmt(stmt)?;
                }
                
                if let Some(else_stmts) = else_branch {
                    for stmt in else_stmts {
                        self.check_stmt(stmt)?;
                    }
                }
                
                Ok(())
            }
            Stmt::Let { pat, type_ann, expr } => {
                let expr_type = self.infer_expr(expr)?;
                
                if let Some(ann) = type_ann {
                    let expected_type = self.annotation_to_type(ann);
                    if !self.types_compatible(&expr_type, &expected_type) {
                        self.errors.push(TypeError {
                            message: format!(
                                "Type mismatch: expected {:?}, got {:?}",
                                expected_type, expr_type
                            ),
                            span: None,
                        });
                    }
                }
                
                self.bind_pattern(pat, expr_type);
                Ok(())
            }
            _ => Ok(()),
        }
    }
    
    fn infer_expr(&mut self, expr: &Expr) -> Result<Type, Vec<TypeError>> {
        match expr {
            Expr::Number(_) => Ok(Type::Float),
            Expr::Str(_) => Ok(Type::String),
            Expr::Var(name) => {
                Ok(self.env.get(name).cloned().unwrap_or(Type::Unknown))
            }
            Expr::NamespacedVar { namespace, name } => {
                if let Some(ns) = self.namespaces.get(namespace) {
                    Ok(ns.functions.get(name).cloned().unwrap_or(Type::Unknown))
                } else {
                    Ok(Type::Unknown)
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let lhs_type = self.infer_expr(lhs)?;
                let rhs_type = self.infer_expr(rhs)?;
                
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        if self.is_numeric(&lhs_type) && self.is_numeric(&rhs_type) {
                            Ok(Type::Float)
                        } else {
                            Ok(Type::Unknown)
                        }
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        Ok(Type::Bool)
                    }
                    BinOp::And | BinOp::Or => Ok(Type::Bool),
                    _ => Ok(Type::Unknown),
                }
            }
            Expr::Call { callee, args } => {
                let func_type = self.infer_expr(callee)?;
                
                match func_type {
                    Type::Function(param_types, ret_type) => {
                        // 引数の型をチェック
                        for (i, arg) in args.iter().enumerate() {
                            let arg_type = self.infer_expr(arg)?;
                            if let Some(expected) = param_types.get(i) {
                                if !self.types_compatible(&arg_type, expected) {
                                    self.errors.push(TypeError {
                                        message: format!(
                                            "Argument {} type mismatch: expected {:?}, got {:?}",
                                            i, expected, arg_type
                                        ),
                                        span: None,
                                    });
                                }
                            }
                        }
                        Ok(*ret_type)
                    }
                    _ => Ok(Type::Unknown),
                }
            }
            Expr::Lambda { params, body: _ } => {
                let param_types: Vec<Type> = params.iter()
                    .map(|(_, ty_ann)| {
                        ty_ann.as_ref()
                            .map(|t| self.annotation_to_type(t))
                            .unwrap_or(Type::Unknown)
                    })
                    .collect();
                
                // ラムダ本体の型推論（簡略版）
                let ret_type = Box::new(Type::Unknown);
                
                Ok(Type::Function(param_types, ret_type))
            }
            Expr::Tuple(exprs) => {
                let types: Result<Vec<Type>, _> = exprs.iter()
                    .map(|e| self.infer_expr(e))
                    .collect();
                Ok(Type::Tuple(types?))
            }
            _ => Ok(Type::Unknown),
        }
    }
    
    // ★ パターンの型バインディングを修正
    fn bind_pattern(&mut self, pat: &Pattern, ty: Type) {
        match pat {
            Pattern::Name(name) => {
                self.env.insert(name.clone(), ty);
            }
            Pattern::Tuple(pats) => {
                if let Type::Tuple(types) = ty {
                    for (p, t) in pats.iter().zip(types.into_iter()) {
                        self.bind_pattern(p, t);
                    }
                }
            }
            Pattern::Struct { name: _, fields } => {
                if let Type::Struct(_, field_types) = ty {
                    for (field_name, field_pat) in fields {
                        if let Some(field_type) = field_types.get(field_name) {
                            self.bind_pattern(field_pat, field_type.clone());
                        }
                    }
                }
            }
            Pattern::Wildcard => {}
        }
    }
    
    fn types_compatible(&self, actual: &Type, expected: &Type) -> bool {
        match (actual, expected) {
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (a, b) if a == b => true,
            (Type::Int, Type::Float) | (Type::Float, Type::Int) => true,
            _ => false,
        }
    }
    
    fn is_numeric(&self, ty: &Type) -> bool {
        matches!(ty, Type::Int | Type::Float | Type::Unknown)
    }
    
    pub fn get_errors(&self) -> &[TypeError] {
        &self.errors
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_type_annotation_parsing() {
        let src = r#"
fn add(x: Int, y: Int) -> Int {
    x + y
}

fn main() {
    let result: Int = add(1, 2);
    print(result);
}
        "#;
        
        let mut parser = Parser::new(src);
        let prog = parser.parse_program().unwrap();
        
        assert_eq!(prog.fns.len(), 2);
        
        let add_fn = &prog.fns[0];
        assert_eq!(add_fn.params.len(), 2);
        assert!(add_fn.ret_type.is_some());
    }
    
    #[test]
    fn test_namespace_parsing() {
        let src = r#"
namespace Math {
    fn square(x: Float) -> Float {
        x * x
    }
    
    struct Point {
        x: Float,
        y: Float
    }
}

fn main() {
    let p = Math::Point { x: 1.0, y: 2.0 };
    let sq = Math::square(5.0);
    print(sq);
}
        "#;
        
        let mut parser = Parser::new(src);
        let prog = parser.parse_program().unwrap();
        
        assert_eq!(prog.namespaces.len(), 1);
        assert_eq!(prog.namespaces[0].name, "Math");
    }
    
    #[test]
    fn test_higher_order_functions() {
        let src = r#"
fn main() {
    let numbers = [1, 2, 3, 4, 5];
    
    let doubled = vec_map(numbers, |x| { x * 2 });
    let evens = vec_filter(numbers, |x| { x % 2 == 0 });
    let sum = vec_reduce(numbers, 0, |acc, x| { acc + x });
    
    print("doubled:", doubled);
    print("evens:", evens);
    print("sum:", sum);
}
        "#;
        
        let mut parser = Parser::new(src);
        let prog = parser.parse_program().unwrap();
        
        // コンパイルが成功することを確認
        let compiled = compile_with_namespaces(&prog);
        assert!(compiled.is_ok());
    }
    
    #[test]
    fn test_type_checker() {
        let src = r#"
fn add(x: Int, y: Int) -> Int {
    x + y
}

fn main() {
    let result: Int = add(1, 2);
    print(result);
}
        "#;
        
        let mut parser = Parser::new(src);
        let prog = parser.parse_program().unwrap();
        
        let mut checker = TypeChecker::new();
        let result = checker.check_program(&prog);
        
        assert!(result.is_ok());
        assert_eq!(checker.get_errors().len(), 0);
    }
    
    #[test]
    fn test_type_error_detection() {
        let src = r#"
fn add(x: Int, y: Int) -> Int {
    x + y
}

fn main() {
    let result: String = add(1, 2);  // 型エラー
    print(result);
}
        "#;
        
        let mut parser = Parser::new(src);
        let prog = parser.parse_program().unwrap();
        
        let mut checker = TypeChecker::new();
        let result = checker.check_program(&prog);
        
        // 型エラーが検出されることを確認
        assert!(result.is_err());
    }
}