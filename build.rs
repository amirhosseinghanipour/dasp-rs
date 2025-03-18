fn main() {
    pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("blas")
        .unwrap();
    pkg_config::Config::new()
        .atleast_version("3.0")
        .probe("lapack")
        .unwrap();
}