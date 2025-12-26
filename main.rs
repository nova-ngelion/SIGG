fn main() {
    if let Err(e) = sigg::cli::main() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
