// Benchmark-only driver linked against the frozen, unchanged Engine crate.
// Admission is outside timing; every timed step must emit for all four IDs.
use std::{env, fs, path::Path, time::Instant};
use pyo3::{prelude::*, types::PyAnyMethods};
use serde_json::{json, Value};
use vosti_verus::{engine::Engine, request_state::{AdmissionStatus, NewRequest},
    tensor_runtime::{self as RT, CudaGraphOverlay}, crate::types::BLOCK_SIZE};
use vosti_verus::boundary::model_families::{gemma3, gemma4, llama3};

mod engine_setup { include!(env!("VOSTI_ALIGNED_ENGINE_SETUP")); }

fn synchronize() {
    Python::with_gil(|py| {
        py.import_bound("torch").unwrap().getattr("cuda").unwrap()
            .call_method0("synchronize").unwrap();
    });
}

fn graph(overlay: &Option<CudaGraphOverlay>) -> Value {
    serde_json::from_str(&RT::cuda_graph_overlay_stats_json(overlay.as_ref().unwrap())).unwrap()
}

fn save(path: &Path, value: &Value) {
    use std::io::Write;
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();
    file.write_all(serde_json::to_string_pretty(value).unwrap().as_bytes()).unwrap();
}

fn add(engine: &mut Engine, rows: &[Value], donor: bool, next_id: &mut u64) -> Vec<u64> {
    rows.iter().map(|r| {
        let id = *next_id; *next_id += 1;
        let prompt: Vec<u64> = r[if donor {"donor"} else {"prompt"}].as_array().unwrap()
            .iter().map(|x| x.as_u64().unwrap()).collect();
        let status = engine.try_add_request(NewRequest {
            request_id: id, prompt_tokens: prompt,
            max_tokens: if donor {1} else {r["max_tokens"].as_u64().unwrap() as usize},
            eos_token_ids: vec![0], ignore_eos: true,
        });
        assert!(status == AdmissionStatus::Accepted, "cohort admission rejected");
        id
    }).collect()
}

fn main() {
    let data: Value = serde_json::from_str(&fs::read_to_string(env::var("ALIGNED_INPUTS").unwrap()).unwrap()).unwrap();
    let out = env::var("ALIGNED_OUTPUT").unwrap(); let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let (model, bundle, device) = engine_setup::deployment_environment("");
    let family = data["checkpoint"]["model"].as_str().unwrap();
    let (weights, runtime, perms, config) = match family {
        "gemma3" => gemma3::deployment::qualify_checkpoint(gemma3::deployment::load_checkpoint(&model, &device), &bundle),
        "gemma4" => gemma4::deployment::qualify_checkpoint(gemma4::deployment::load_checkpoint(&model, &device), &bundle),
        "llama3" => llama3::deployment::qualify_checkpoint(llama3::deployment::load_checkpoint(&model, &device), &bundle),
        _ => panic!("unsupported matrix family"),
    };
    let arch = match family { "gemma3" => "gemma3_text", "gemma4" => "gemma4_text", _ => "llama3" };
    let (mut engine, overlay) = engine_setup::initialize_qualified_engine(arch, 4,
        engine_setup::env_u64("VOSTI_NUM_BLOCKS",1152), weights, runtime, perms, config, vec![]);
    assert!(overlay.is_some());
    let mut next_id = 0;
    for case in data["cases"].as_array().unwrap() {
        let rows = case["rows"].as_array().unwrap();
        assert_eq!(rows.len(),4);
        let cached = case["cached_tokens"].as_u64().unwrap();
        let query = case["query_tokens"].as_u64().unwrap();
        let outputs = if query == 1 {128} else {1};
        if cached > 0 {
            add(&mut engine, rows, true, &mut next_id);
            while !engine.cs.live_requests.is_empty() { engine.step(overlay.as_ref()); }
        }
        let ids = add(&mut engine, rows, false, &mut next_id);
        let before = graph(&overlay);
        synchronize();
        let start = Instant::now();
        let (emitted,_,_) = engine.step(overlay.as_ref());
        synchronize();
        let prefill_s = start.elapsed().as_secs_f64();
        assert_eq!(emitted.len(),4,"prefill batch split");
        assert!(ids.iter().all(|id| emitted.contains_key(id)));
        let prefix: Vec<_> = ids.iter().map(|id| engine.last_step_prefix_reuse.iter()
            .find(|p| p.request_id == *id).unwrap().cached_prefix_blocks.unwrap()*BLOCK_SIZE).collect();
        assert_eq!(prefix,vec![cached;4],"non-exact cache reuse");
        let mut generated: Vec<Vec<u64>> = ids.iter().map(|id| vec![*emitted.get(id).unwrap()]).collect();
        let mut decode_batches = vec![];
        let decode_before = graph(&overlay);
        // No per-step synchronization. Engine::step retains its normal sample/commit work.
        let decode_start = Instant::now();
        for step in 1..outputs {
            assert_eq!(engine.cs.live_requests.len(),4);
            let lengths: Vec<_> = ids.iter().map(|id| {
                let r = engine.cs.live_requests.get(id).unwrap();
                (r.prompt_tokens.len()+r.generated_tokens.len()) as u64
            }).collect();
            assert_eq!(lengths,vec![cached+step+1;4]);
            let (emitted,_,_) = engine.step(overlay.as_ref());
            assert_eq!(emitted.len(),4,"decode batch split");
            assert_eq!(engine.last_step_prefix_reuse.len(),4);
            for (i,id) in ids.iter().enumerate() { generated[i].push(*emitted.get(id).unwrap()); }
            decode_batches.push(json!({"request_ids":ids,"seq_lens":lengths}));
        }
        synchronize();
        let decode_s = decode_start.elapsed().as_secs_f64();
        assert!(engine.cs.live_requests.is_empty());
        let after = graph(&overlay);
        if case["phase"] == "measured" {
            assert_eq!(before["capture_count"],after["capture_count"],"measured graph capture");
            assert!(after["poisoned_reason"].is_null());
            if query == 1 {
                assert_eq!(after["replay_count"].as_u64().unwrap()-decode_before["replay_count"].as_u64().unwrap(),127);
                assert_eq!(after["eager_count"],decode_before["eager_count"],"eager measured decode");
            }
        }
        let seconds = if query == 1 {decode_s} else {prefill_s};
        let tokens = if query == 1 {508} else {4*query};
        let record = json!({"name":case["name"],"phase":case["phase"],"cached_tokens":cached,
            "query_tokens":query,"prefix_lens":prefix,"query_lens":vec![query;4],"request_ids":ids,
            "batches":outputs,"decode_batches":decode_batches,"prefill_wall_s":prefill_s,
            "engine_wall_s":seconds,"measured_tokens":tokens,"tokens_per_s":tokens as f64/seconds,
            "generated_ids":generated,"graph_before":before,"graph_decode_before":decode_before,"graph_after":after});
        save(&out.join(format!("{}.json",case["name"].as_str().unwrap())), &record);
        println!("WAVE {} {} {:.3}ms",case["name"],case["phase"],seconds*1000.0);
    }
    save(&out.join("complete.json"), &json!({"complete":true}));
}
