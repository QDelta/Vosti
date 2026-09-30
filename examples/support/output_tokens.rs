// Architecture-neutral output-token record shared by executable Engine
// examples and their GPU smoke harness.

use std::path::Path;

pub fn write_outputs(path: &Path, outputs: &[Vec<u64>]) -> Result<(), String> {
    let mut encoded = String::new();
    for (request_index, tokens) in outputs.iter().enumerate() {
        encoded.push_str(&request_index.to_string());
        encoded.push('\t');
        encoded.push_str(
            &tokens
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        encoded.push('\n');
    }
    std::fs::write(path, encoded)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}
