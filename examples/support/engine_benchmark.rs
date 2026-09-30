// Architecture-neutral benchmark loop for executable Engine examples.
//
// Model-family examples remain responsible only for loading and qualifying
// their weights.  This module owns the common prepared-ID input format and the
// timing boundary from request admission through completed output-token IDs.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use vosti_verus::exec::engine::Engine;
use vosti_verus::exec::request_state::{EosTokenSet, RequestState, SamplerState, MAX_EOS_TOKEN_IDS};
use vosti_verus::boundary::tensor_runtime::{self as RT, CudaGraphOverlay};

#[cfg(feature = "openai-server")]
#[path = "engine_calls.rs"]
pub mod calls;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedRequest {
    pub prompt_tokens: Vec<u64>,
    pub max_tokens: usize,
    pub arrival_step: usize,
}

#[derive(Debug)]
pub struct RunResult {
    pub wall_s: f64,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
    pub outputs: Vec<Vec<u64>>,
    pub admitted_requests: usize,
    pub requests_with_reuse: usize,
    pub reused_prefix_blocks: usize,
}

pub struct PairedRunResult {
    pub warmup: RunResult,
    pub measured: RunResult,
    pub graph_warmup_stats: Option<String>,
    pub graph_stats: Option<String>,
}

impl PairedRunResult {
    pub fn report(&self, warmup_batch: usize, measured_batch: usize) {
        println!(
            "WARMUP batch={} prompt_tokens={} output_tokens={} wall_s={:.9}",
            warmup_batch, self.warmup.prompt_tokens, self.warmup.output_tokens, self.warmup.wall_s,
        );
        println!(
            "WARMUP_CACHE_STATS {{\"admitted_requests\":{},\"requests_with_reuse\":{},\"reused_prefix_blocks\":{}}}",
            self.warmup.admitted_requests,
            self.warmup.requests_with_reuse,
            self.warmup.reused_prefix_blocks,
        );
        println!(
            "BENCH batch={} prompt_tokens={} output_tokens={} wall_s={:.9} tok_s={:.6}",
            measured_batch,
            self.measured.prompt_tokens,
            self.measured.output_tokens,
            self.measured.wall_s,
            self.measured.output_tokens as f64 / self.measured.wall_s,
        );
        println!(
            "CACHE_STATS {{\"admitted_requests\":{},\"requests_with_reuse\":{},\"reused_prefix_blocks\":{}}}",
            self.measured.admitted_requests,
            self.measured.requests_with_reuse,
            self.measured.reused_prefix_blocks,
        );
        if let Some(stats) = self.graph_warmup_stats.as_ref() {
            println!("GRAPH_WARMUP_STATS {stats}");
        }
        if let Some(stats) = self.graph_stats.as_ref() {
            println!("GRAPH_STATS {stats}");
        }
    }
}

// Each nonempty, non-comment line is:
//
//     MAX_TOKENS<TAB>COMMA_SEPARATED_PROMPT_TOKEN_IDS
// or, for deterministic schedule stress:
//     MAX_TOKENS<TAB>COMMA_SEPARATED_PROMPT_TOKEN_IDS<TAB>ARRIVAL_STEP
// The optional arrival is a simulated engine-loop step, not wall-clock QPS.
//
// The companion Python harness creates this file after tokenization and before
// model initialization, keeping parsing and request construction outside the
// measured interval.
pub fn read_prepared_requests(path: &Path) -> Result<Vec<PreparedRequest>, String> {
    let input = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    parse_prepared_requests(&input, path)
}

fn parse_prepared_requests(input: &str, path: &Path) -> Result<Vec<PreparedRequest>, String> {
    let mut requests = Vec::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 2 && fields.len() != 3 {
            return Err(format!(
                "{}:{} requires two columns and an optional arrival step",
                path.display(),
                line_index + 1
            ));
        }
        let (raw_max_tokens, raw_prompt_tokens) = (fields[0], fields[1]);
        let arrival_step = if fields.len() == 3 {
            fields[2].parse::<usize>().map_err(|_| {
                format!(
                    "{}:{} has invalid arrival step",
                    path.display(),
                    line_index + 1
                )
            })?
        } else {
            0
        };
        let max_tokens = raw_max_tokens.parse::<usize>().map_err(|_| {
            format!(
                "{}:{} has invalid max token count {raw_max_tokens:?}",
                path.display(),
                line_index + 1,
            )
        })?;
        if max_tokens == 0 {
            return Err(format!(
                "{}:{} has zero max token count",
                path.display(),
                line_index + 1,
            ));
        }
        let prompt_tokens = raw_prompt_tokens
            .split(',')
            .map(|raw_token| {
                raw_token.parse::<u64>().map_err(|_| {
                    format!(
                        "{}:{} has invalid token ID {raw_token:?}",
                        path.display(),
                        line_index + 1,
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if prompt_tokens.is_empty() {
            return Err(format!(
                "{}:{} has an empty prompt",
                path.display(),
                line_index + 1,
            ));
        }
        requests.push(PreparedRequest {
            prompt_tokens,
            max_tokens,
            arrival_step,
        });
    }
    if requests.is_empty() {
        return Err(format!("{} contains no requests", path.display()));
    }
    Ok(requests)
}

pub fn run_prepared_requests(
    engine: &mut Engine,
    graph_overlay: Option<&CudaGraphOverlay>,
    prepared: &[PreparedRequest],
    request_id_base: u64,
    eos_token_ids: &[u64],
) -> Result<RunResult, String> {
    if prepared.is_empty() {
        return Err("benchmark workload must be nonempty".to_string());
    }
    if !engine.cs.live_requests.is_empty() {
        return Err("benchmark run requires an idle Engine".to_string());
    }

    let prompt_tokens = prepared.iter().try_fold(0usize, |total, request| {
        total
            .checked_add(request.prompt_tokens.len())
            .ok_or_else(|| "prompt token count overflow".to_string())
    })?;
    let expected_output_tokens = prepared.iter().try_fold(0usize, |total, request| {
        total
            .checked_add(request.max_tokens)
            .ok_or_else(|| "output token count overflow".to_string())
    })?;
    let last_arrival = prepared
        .iter()
        .map(|request| request.arrival_step)
        .max()
        .unwrap_or(0);
    let max_steps = prompt_tokens
        .checked_add(expected_output_tokens)
        .and_then(|value| value.checked_add(last_arrival))
        .and_then(|value| value.checked_add(prepared.len()))
        .and_then(|value| value.checked_add(16))
        .ok_or_else(|| "benchmark step bound overflow".to_string())?;
    if eos_token_ids.is_empty() || eos_token_ids.len() > MAX_EOS_TOKEN_IDS {
        return Err(format!(
            "benchmark requires 1..={MAX_EOS_TOKEN_IDS} EOS token ids",
        ));
    }
    let eos_token_ids = eos_token_ids.to_vec();
    let eos_token_set = EosTokenSet::from_nonempty_bounded(&eos_token_ids);

    // Intrusive correctness instrumentation, never enabled by performance runs.
    // Runs in a prepared pair append to one fresh invocation-local trace.
    let mut step_trace = std::env::var_os("VOSTI_ENGINE_STEP_TRACE")
        .map(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
        })
        .transpose()
        .map_err(|error| format!("cannot open engine step trace: {error}"))?;

    // Prepare the arrival order once, outside timing. Existing all-at-zero
    // workloads retain their request order without a per-step workload scan.
    let mut arrival_order: Vec<usize> = (0..prepared.len()).collect();
    arrival_order.sort_by_key(|&index| (prepared[index].arrival_step, index));
    let mut next_arrival = 0usize;
    let started = Instant::now();
    let mut outputs = vec![Vec::new(); prepared.len()];
    let mut cache_seen = vec![false; prepared.len()];
    let mut admitted_requests = 0usize;
    let mut requests_with_reuse = 0usize;
    let mut reused_prefix_blocks = 0usize;
    for step in 0..max_steps {
        let mut arrived = Vec::new();
        while next_arrival < arrival_order.len()
            && prepared[arrival_order[next_arrival]].arrival_step == step
        {
            let request_index = arrival_order[next_arrival];
            let request = &prepared[request_index];
            next_arrival += 1;
            let request_id = request_id_base
                .checked_add(
                    u64::try_from(request_index)
                        .map_err(|_| "request index exceeds u64".to_string())?,
                )
                .ok_or_else(|| "request ID overflow".to_string())?;
            engine.add_request(RequestState::from_parts(
                request_id,
                request.prompt_tokens.clone(),
                Vec::new(),
                SamplerState::empty(),
                request.max_tokens,
                eos_token_set,
                true,
            ));
            if step_trace.is_some() {
                arrived.push(request_id);
            }
        }
        if engine.cs.live_requests.is_empty() {
            if step >= last_arrival {
                break;
            }
            continue;
        }
        if step_trace.is_some() {
            crate::engine_serving::set_python_environment(
                "VOSTI_LOGITS_OBSERVER_STEP",
                &format!("{request_id_base}:{step}"),
            );
        }
        let (emitted, _samples, _reprs) = engine.step(graph_overlay);
        if let Some(trace) = step_trace.as_mut() {
            let scheduled: Vec<u64> = engine
                .last_step_prefix_reuse
                .iter()
                .map(|row| row.request_id)
                .collect();
            let tokens: Vec<String> = scheduled
                .iter()
                .map(|rid| {
                    emitted
                        .get(rid)
                        .map(|token| token.to_string())
                        .unwrap_or_else(|| "null".to_string())
                })
                .collect();
            let cached: Vec<String> = engine
                .last_step_prefix_reuse
                .iter()
                .map(|row| {
                    row.cached_prefix_blocks
                        .map(|blocks| blocks.to_string())
                        .unwrap_or_else(|| "null".to_string())
                })
                .collect();
            let prefix_pages: Vec<String> = (0..engine.cs.num_blocks)
                .filter_map(|bid| {
                    let block = engine.cs.blocks.get(&bid)?;
                    if block.prefix_depth == 0 {
                        return None;
                    }
                    let parent = block
                        .parent_block
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "null".to_string());
                    Some(format!(
                        "[{bid},{},{parent},{:?}]",
                        block.prefix_depth, block.tokens
                    ))
                })
                .collect();
            writeln!(trace,
                "{{\"engine_step\":\"{request_id_base}:{step}\",\"request_id_base\":{request_id_base},\"step\":{step},\"arrived\":{arrived:?},\"scheduled\":{scheduled:?},\"emitted\":[{}],\"cached_prefix_blocks\":[{}],\"prefix_pages\":[{}]}}",
                tokens.join(","), cached.join(","), prefix_pages.join(","),
            ).map_err(|error| format!("cannot write engine step trace: {error}"))?;
        }
        for observation in &engine.last_step_prefix_reuse {
            let request_index = observation
                .request_id
                .checked_sub(request_id_base)
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index < prepared.len())
                .ok_or_else(|| "scheduled request is outside benchmark workload".to_string())?;
            if cache_seen[request_index] {
                continue;
            }
            let reused = observation
                .cached_prefix_blocks
                .and_then(|blocks| usize::try_from(blocks).ok())
                .filter(|blocks| {
                    *blocks
                        <= prepared[request_index].prompt_tokens.len()
                            / vosti_verus::types::BLOCK_SIZE as usize
                })
                .ok_or_else(|| "invalid initial scheduled cache observation".to_string())?;
            cache_seen[request_index] = true;
            admitted_requests += 1;
            reused_prefix_blocks = reused_prefix_blocks
                .checked_add(reused)
                .ok_or_else(|| "reused prefix block count overflow".to_string())?;
            if reused > 0 {
                requests_with_reuse += 1;
            }
        }
        for (request_index, output) in outputs.iter_mut().enumerate() {
            let request_id = request_id_base
                .checked_add(
                    u64::try_from(request_index)
                        .map_err(|_| "request index exceeds u64".to_string())?,
                )
                .ok_or_else(|| "request ID overflow".to_string())?;
            if let Some(token) = emitted.get(&request_id) {
                output.push(*token);
            }
        }
    }
    if !engine.cs.live_requests.is_empty() {
        return Err(format!(
            "benchmark did not finish within the conservative {max_steps}-step bound",
        ));
    }
    let wall_s = started.elapsed().as_secs_f64();

    for (request_index, (output, request)) in outputs.iter().zip(prepared.iter()).enumerate() {
        if output.len() != request.max_tokens {
            return Err(format!(
                "request {request_index} emitted {} tokens, expected {}",
                output.len(),
                request.max_tokens,
            ));
        }
    }
    let output_tokens = outputs.iter().map(Vec::len).sum();
    if output_tokens != expected_output_tokens {
        return Err(format!(
            "benchmark emitted {output_tokens} tokens, expected {expected_output_tokens}",
        ));
    }

    Ok(RunResult {
        wall_s,
        prompt_tokens,
        output_tokens,
        outputs,
        admitted_requests,
        requests_with_reuse,
        reused_prefix_blocks,
    })
}

#[cfg(test)]
mod prepared_input_tests {
    use super::*;

    #[test]
    fn old_input_keeps_all_arrivals_at_zero() {
        let rows = parse_prepared_requests("2\t1,2\n3\t4,5\n", Path::new("test")).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.arrival_step).collect::<Vec<_>>(),
            vec![0, 0]
        );
        assert_eq!(rows[0].prompt_tokens, vec![1, 2]);
    }

    #[test]
    fn explicit_arrivals_preserve_request_order_and_geometry() {
        let rows = parse_prepared_requests("2\t1,2\t5\n3\t4,5\t0\n", Path::new("test")).unwrap();
        assert_eq!(
            rows.iter().map(|r| r.arrival_step).collect::<Vec<_>>(),
            vec![5, 0]
        );
        assert_eq!(
            rows.iter().map(|r| r.max_tokens).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn invalid_arrivals_and_extra_columns_fail() {
        for input in ["2\t1,2\t-1", "2\t1,2\tx", "2\t1,2\t0\textra"] {
            assert!(parse_prepared_requests(input, Path::new("test")).is_err());
        }
    }
}

pub fn run_prepared_pair<F>(
    engine: &mut Engine,
    graph_overlay: Option<&CudaGraphOverlay>,
    request_id_base: u64,
    warmup: &[PreparedRequest],
    measured: &[PreparedRequest],
    eos_token_ids: &[u64],
    before_measured: F,
) -> Result<PairedRunResult, String>
where
    F: FnOnce(),
{
    let warmup_result = run_prepared_requests(
        engine,
        graph_overlay,
        warmup,
        request_id_base,
        eos_token_ids,
    )?;
    let graph_warmup_stats = graph_overlay.map(RT::cuda_graph_overlay_stats_json);
    before_measured();
    let measured_request_id_base = request_id_base
        .checked_add(
            u64::try_from(warmup.len()).map_err(|_| "warmup batch size exceeds u64".to_string())?,
        )
        .ok_or_else(|| "measured request ID base overflow".to_string())?;
    let measured_result = run_prepared_requests(
        engine,
        graph_overlay,
        measured,
        measured_request_id_base,
        eos_token_ids,
    )?;
    let graph_stats = graph_overlay.map(RT::cuda_graph_overlay_stats_json);
    Ok(PairedRunResult {
        warmup: warmup_result,
        measured: measured_result,
        graph_warmup_stats,
        graph_stats,
    })
}
