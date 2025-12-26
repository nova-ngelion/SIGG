fn main() {
    if let Err(e) = sigg::server::main_server_bin() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
