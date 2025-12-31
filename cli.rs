use crate::bytecode;
use crate::error::SiggError;
use crate::parser::Parser as SiggParser;
use crate::vm::VM;

pub fn main() -> Result<(), SiggError> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Err(SiggError::runtime("usage: sigg <run|serve> ..."));
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 3 {
                return Err(SiggError::runtime("usage: sigg run <file.sigg>"));
            }
            let path = &args[2];
            let src = std::fs::read_to_string(path)
                .map_err(|e| SiggError::io(e.to_string()))?;

            // 1. パース (Source -> AST)
            let mut ps = SiggParser::new(&src);
            let ast_prog = ps.parse_program()?;

            // 2. コンパイル (AST -> Bytecode)
            let compiled = crate::bytecode::compile(&ast_prog)?;

            // 3. VMの準備と実行
            let mut vm = VM::new();
            // VM::new() 内の HostFunction 登録で Arc::new(*f) を使う修正を vm.rs で忘れないでください
            
            if let Err(e) = vm.exec_compiled(&compiled) {
                println!("Execution error: {:?}", e);
                return Err(e);
            }
            Ok(())
        }

        "serve" => {
            let mut addr = "127.0.0.1:9001".to_string();
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--addr" && i + 1 < args.len() {
                    addr = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            // 実際の通信処理は server.rs の中で行われます
            crate::server::serve(&addr)
        }

        _ => Err(SiggError::runtime("usage: sigg <run|serve> ...")),
    }
}
