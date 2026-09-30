// Test-driver multi-call engine lifetime. No model-family or engine semantics.
use crate::engine_benchmark::{run_prepared_requests, PreparedRequest};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::Path;
use vosti_verus::exec::engine::Engine;
use vosti_verus::boundary::tensor_runtime::{self as RT, CudaGraphOverlay};

#[derive(Deserialize)]
struct OutputPrefix {
    call: usize,
    request: usize,
    count: usize,
    suffix: Vec<u64>,
}

#[derive(Deserialize)]
struct Call {
    label: String,
    #[serde(default)]
    prompts: Vec<Vec<u64>>,
    max_tokens: usize,
    output_prefix_from: Option<OutputPrefix>,
}

#[derive(Deserialize)]
struct Calls {
    prefix_caching: bool,
    calls: Vec<Call>,
}

fn resolve_prompts(
    call: &Call,
    prior: &[(Vec<Vec<u64>>, Vec<Vec<u64>>)],
) -> Result<Vec<Vec<u64>>, String> {
    if let Some(source) = &call.output_prefix_from {
        let (prompts, outputs) = prior
            .get(source.call)
            .ok_or("future generated-prefix call")?;
        let prompt = prompts
            .get(source.request)
            .ok_or("invalid generated-prefix request")?;
        let output = outputs
            .get(source.request)
            .ok_or("missing generated-prefix output")?;
        if source.count > output.len() {
            return Err("generated-prefix count exceeds output".into());
        }
        let mut query = prompt.clone();
        query.extend_from_slice(&output[..source.count]);
        query.extend_from_slice(&source.suffix);
        Ok(vec![query])
    } else {
        Ok(call.prompts.clone())
    }
}

pub fn run(
    engine: &mut Engine,
    overlay: Option<&CudaGraphOverlay>,
    path: &Path,
    eos: &[u64],
) -> Result<(), String> {
    let input: Calls =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if input.calls.is_empty() || overlay.is_none() {
        return Err("multi-call driver requires calls and CUDA graphs".into());
    }
    let mut prior = Vec::new();
    let mut records: Vec<Value> = Vec::new();
    let mut request_base = 0u64;
    for (index, call) in input.calls.iter().enumerate() {
        if !engine.cs.live_requests.is_empty() {
            return Err("previous native call did not finish".into());
        }
        let reclaimed = if !input.prefix_caching {
            let capacity = engine.cs.num_blocks;
            let count = engine.cs.reclaim_cached_leaves_until(capacity);
            if engine.cs.free_blocks != capacity {
                return Err("native cache isolation left resident pages".into());
            }
            count
        } else {
            0
        };
        let prompts = resolve_prompts(call, &prior)?;
        if prompts.is_empty() || prompts.iter().any(Vec::is_empty) || call.max_tokens == 0 {
            return Err("native call has an empty prompt/batch or output budget".into());
        }
        let prepared: Vec<_> = prompts
            .iter()
            .map(|tokens| PreparedRequest {
                prompt_tokens: tokens.clone(),
                max_tokens: call.max_tokens,
                arrival_step: 0,
            })
            .collect();
        crate::engine_serving::set_python_environment(
            "VOSTI_LOGITS_OBSERVER_PHASE",
            &format!("call-{index}"),
        );
        let result = run_prepared_requests(engine, overlay, &prepared, request_base, eos)?;
        records.push(json!({"label": call.label, "request_id_base": request_base,
            "input_token_ids": prompts, "output_token_ids": result.outputs,
            "reclaimed_pages_before_call": reclaimed, "cache_isolated": !input.prefix_caching,
            "cache_page_tokens": vosti_verus::types::BLOCK_SIZE,
            "wall_s": result.wall_s, "graph_stats": overlay.map(RT::cuda_graph_overlay_stats_json)}));
        request_base = request_base
            .checked_add(prompts.len() as u64)
            .ok_or("native request ID overflow")?;
        prior.push((prompts, result.outputs));
    }
    let output = std::env::var("VOSTI_BENCH_CALL_RESULTS").map_err(|e| e.to_string())?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer(&mut file, &records).map_err(|e| e.to_string())?;
    println!("CALLS_FINISHED {}", records.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_prefix_uses_actual_prior_tokens() {
        let call: Call = serde_json::from_value(json!({"label":"reuse", "max_tokens":2,
            "output_prefix_from":{"call":0,"request":0,"count":2,"suffix":[8]}}))
        .unwrap();
        assert_eq!(
            resolve_prompts(&call, &[(vec![vec![1, 2]], vec![vec![3, 4, 5]])]).unwrap(),
            vec![vec![1, 2, 3, 4, 8]]
        );
        assert!(resolve_prompts(&call, &[]).is_err());
        assert!(resolve_prompts(&call, &[(vec![vec![1, 2]], vec![vec![3]])]).is_err());
    }
}
