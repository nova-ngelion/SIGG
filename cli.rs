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

            let mut ps = SiggParser::new(&src);
            let prog = ps.parse_program()?;

            let mut vm = VM::new();
            let compiled = bytecode::compile(&prog)?;
            vm.exec_compiled(&compiled)?;
            Ok(())
        }

        "serve" => {
            // sigg serve --addr 127.0.0.1:39999
            let mut addr = "127.0.0.1:39999".to_string();
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--addr" && i + 1 < args.len() {
                    addr = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            crate::server::serve(&addr)
        }

        _ => Err(SiggError::runtime("usage: sigg <run|serve> ...")),
    }
}
