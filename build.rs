fn main() {
    #[cfg(feature = "grpc")]
    {
        tonic_build::configure()
            .build_server(false) // Huoma is a client, not a server
            .build_client(true)
            .out_dir("src/ttn/gen")
            .compile_protos(&["proto/arn.proto"], &["proto"])
            .expect("failed to compile arn.proto");
    }
}
