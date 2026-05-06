// Llama-style transformer decode loop, fully self-contained, with the
// `metal-profile` GPU profiler enabled. No model download / tokenizer needed:
// random weights, random input embeddings, deterministic shapes.
//
// Run with:
//   cargo run -p candle-examples --example metal_profile_llm --release --features metal-profile
//
// To also produce Apple's native Xcode GPU capture document:
//   MTL_CAPTURE_ENABLED=1 cargo run -p candle-examples --example metal_profile_llm --release \
//     --features metal-profile -- --gputrace /tmp/candle-metal-profile-llm.gputrace
//
// Produces /tmp/candle-metal-profile-llm.json. Inspect programmatically:
//   jq '.traceEvents | length' /tmp/candle-metal-profile-llm.json
//   trace_processor -Q "select name, count(*) n, sum(dur)/1000.0 total_us from slice group by name order by sum(dur) desc limit 15;" \
//     /tmp/candle-metal-profile-llm.json

use std::path::PathBuf;

use anyhow::Result;
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_no_bias, rms_norm, Linear, RmsNorm, VarBuilder, VarMap};
use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    /// Chrome Trace JSON output path for Candle's structured profiler.
    #[arg(long, default_value = "/tmp/candle-metal-profile-llm.json")]
    profile_json: PathBuf,

    /// Optional Apple/Xcode `.gputrace` output path. Requires
    /// `MTL_CAPTURE_ENABLED=1` when running outside Xcode.
    #[arg(long)]
    gputrace: Option<PathBuf>,

    /// Number of decode tokens to profile after warm-up.
    #[arg(long, default_value_t = 32)]
    decode_steps: usize,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    vocab_size: usize,
    hidden: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ffn_hidden: usize,
    max_seq_len: usize,
    rms_eps: f64,
    rope_theta: f32,
}

impl Config {
    fn small() -> Self {
        // Llama-shaped, scaled down so we don't need real weights/checkpoints
        // and the example finishes in a couple of seconds while still
        // exercising the full kernel mix (RMSNorm / GQA matmul / SDPA-equivalent
        // / RoPE / SwiGLU / output projection / argmax).
        Self {
            vocab_size: 4096,
            hidden: 512,
            n_layers: 6,
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 64,
            ffn_hidden: 1408,
            max_seq_len: 64,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    cache_k: Option<Tensor>,
    cache_v: Option<Tensor>,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder, cos: &Tensor, sin: &Tensor) -> Result<Self> {
        let q = linear_no_bias(cfg.hidden, cfg.n_heads * cfg.head_dim, vb.pp("q_proj"))?;
        let k = linear_no_bias(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, vb.pp("k_proj"))?;
        let v = linear_no_bias(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, vb.pp("v_proj"))?;
        let o = linear_no_bias(cfg.n_heads * cfg.head_dim, cfg.hidden, vb.pp("o_proj"))?;
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
            cache_k: None,
            cache_v: None,
        })
    }

    fn apply_rope(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        let (b, h, s, d) = x.dims4()?;
        let cos = self
            .cos
            .i((position..position + s, ..))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let sin = self
            .sin
            .i((position..position + s, ..))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let cos = cos.broadcast_as((b, h, s, d / 2))?;
        let sin = sin.broadcast_as((b, h, s, d / 2))?;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?;
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        Ok((x.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?)
    }

    fn forward(&mut self, x: &Tensor, position: usize) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q.forward(x)?;
        let k = self.k.forward(x)?;
        let v = self.v.forward(x)?;
        let q = q
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.apply_rope(&q.contiguous()?, position)?;
        let k = self.apply_rope(&k.contiguous()?, position)?;

        // KV cache append.
        let k = match &self.cache_k {
            Some(prev) => Tensor::cat(&[prev, &k], 2)?,
            None => k,
        };
        let v = match &self.cache_v {
            Some(prev) => Tensor::cat(&[prev, &v], 2)?,
            None => v,
        };
        self.cache_k = Some(k.clone());
        self.cache_v = Some(v.clone());

        // Repeat KV heads to match Q heads (GQA).
        let rep = self.n_heads / self.n_kv_heads;
        let k = if rep == 1 {
            k
        } else {
            let (b, h, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, h, rep, s, d))?
                .reshape((b, h * rep, s, d))?
        };
        let v = if rep == 1 {
            v
        } else {
            let (b, h, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, h, rep, s, d))?
                .reshape((b, h * rep, s, d))?
        };

        // Scaled dot product.
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let scores = (scores * scale)?;
        // Causal masking is implicit during decode (single token per step).
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs.matmul(&v.contiguous()?)?;
        let context = context
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?
            .contiguous()?;
        Ok(self.o.forward(&context)?)
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden, cfg.ffn_hidden, vb.pp("gate_proj"))?;
        let up = linear_no_bias(cfg.hidden, cfg.ffn_hidden, vb.pp("up_proj"))?;
        let down = linear_no_bias(cfg.ffn_hidden, cfg.hidden, vb.pp("down_proj"))?;
        Ok(Self { gate, up, down })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?.silu()?;
        let u = self.up.forward(x)?;
        let h = (g * u)?;
        Ok(self.down.forward(&h)?)
    }
}

struct Block {
    attn_norm: RmsNorm,
    attn: Attention,
    mlp_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn new(cfg: &Config, vb: VarBuilder, cos: &Tensor, sin: &Tensor) -> Result<Self> {
        Ok(Self {
            attn_norm: rms_norm(cfg.hidden, cfg.rms_eps, vb.pp("attn_norm"))?,
            attn: Attention::new(cfg, vb.pp("attn"), cos, sin)?,
            mlp_norm: rms_norm(cfg.hidden, cfg.rms_eps, vb.pp("mlp_norm"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, position: usize) -> Result<Tensor> {
        let h = self.attn_norm.forward(x)?;
        let h = self.attn.forward(&h, position)?;
        let x = (x + h)?;
        let h = self.mlp_norm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        Ok((x + h)?)
    }
}

struct Model {
    embed: Tensor,
    blocks: Vec<Block>,
    out_norm: RmsNorm,
    out_proj: Linear,
}

impl Model {
    fn new(cfg: &Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        // Embedding table (initialized random; we drive it as a plain tensor
        // since the example uses random tokens that we then look up via index
        // select — keeps the kernel mix faithful to a real decode step).
        let embed = vb.pp("embed").get((cfg.vocab_size, cfg.hidden), "weight")?;

        // Precompute RoPE cos/sin tables.
        let inv_freq: Vec<f32> = (0..cfg.head_dim / 2)
            .map(|i| cfg.rope_theta.powf(-(2.0 * i as f32) / cfg.head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, cfg.head_dim / 2), device)?;
        let pos: Vec<f32> = (0..cfg.max_seq_len).map(|i| i as f32).collect();
        let pos = Tensor::from_vec(pos, (cfg.max_seq_len, 1), device)?;
        let freqs = pos.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(Block::new(cfg, vb.pp(format!("layers.{i}")), &cos, &sin)?);
        }
        let out_norm = rms_norm(cfg.hidden, cfg.rms_eps, vb.pp("out_norm"))?;
        let out_proj = linear_no_bias(cfg.hidden, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed,
            blocks,
            out_norm,
            out_proj,
        })
    }

    fn forward(&mut self, token: u32, position: usize) -> Result<Tensor> {
        // Embed lookup: gather one row from the embedding table.
        let idx = Tensor::from_slice(&[token], (1, 1), self.embed.device())?;
        let h = self
            .embed
            .index_select(&idx.flatten_all()?, 0)?
            .reshape((1, 1, ()))?;
        let mut h = h;
        for blk in &mut self.blocks {
            h = blk.forward(&h, position)?;
        }
        let h = self.out_norm.forward(&h)?;
        let logits = self.out_proj.forward(&h)?;
        Ok(logits.squeeze(1)?)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_metal(0)?;
    let metal = match &device {
        Device::Metal(m) => m,
        _ => anyhow::bail!("expected Metal device"),
    };

    let cfg = Config::small();
    println!(
        "config: hidden={} n_layers={} n_heads={} n_kv_heads={} head_dim={} ffn={} vocab={}",
        cfg.hidden,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.ffn_hidden,
        cfg.vocab_size,
    );

    // Random weights (no checkpoint loading).
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mut warmup_model = Model::new(&cfg, vb.clone(), &device)?;

    // Warm-up to amortize first-run kernel compilation / pipeline cache. Then
    // rebuild the model wrapper so KV caches start empty for the measured decode.
    let _ = warmup_model.forward(0, 0)?;
    metal.wait_until_completed()?;
    drop(warmup_model);
    let mut model = Model::new(&cfg, vb, &device)?;

    // Now install the profiler and run the actual decode loop.
    metal.install_profiler()?;
    println!("profiler installed");
    if let Some(path) = &args.gputrace {
        metal.capture(path)?;
        println!("Apple Metal capture started: {}", path.display());
    }

    let decode_steps = args.decode_steps;
    let t0 = std::time::Instant::now();
    let mut last_token: u32 = 7;
    for step in 0..decode_steps {
        let logits = model.forward(last_token, step)?;
        let next = logits.argmax(D::Minus1)?.flatten_all()?.to_vec1::<u32>()?[0];
        last_token = next;
        let _ = step;
    }
    metal.wait_until_completed()?;
    let elapsed = t0.elapsed();
    println!(
        "decoded {decode_steps} tokens in {:.3} ms ({:.1} tok/s)",
        elapsed.as_secs_f64() * 1e3,
        decode_steps as f64 / elapsed.as_secs_f64()
    );

    if args.gputrace.is_some() {
        metal.stop_capture()?;
        println!("Apple Metal capture stopped");
    }

    let n = metal.flush_profile(&args.profile_json)?;
    println!("wrote {} ({n} events)", args.profile_json.display());

    if let Some(events) = metal.profile_snapshot()? {
        let gpu_encoders: Vec<_> = events.iter().filter(|e| e.is_gpu_encoder()).collect();
        let total_ns: u64 = gpu_encoders.iter().map(|e| e.duration_ns()).sum();
        let mut by_label: std::collections::HashMap<&str, (usize, u64)> =
            std::collections::HashMap::new();
        for ev in &gpu_encoders {
            let entry = by_label.entry(ev.label.as_str()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += ev.duration_ns();
        }
        let mut rows: Vec<_> = by_label.into_iter().collect();
        rows.sort_by_key(|(_, (_, total))| std::cmp::Reverse(*total));
        println!("\n{:>4}  {:>12}  {:>12}  label", "n", "total_us", "avg_us");
        for (label, (n, total)) in rows.iter().take(15) {
            println!(
                "{:>4}  {:>12.3}  {:>12.3}  {}",
                n,
                (*total as f64) / 1e3,
                (*total as f64) / (*n as f64) / 1e3,
                label
            );
        }
        println!(
            "\ntotal GPU time across {} compute encoders: {:.3} ms",
            gpu_encoders.len(),
            (total_ns as f64) / 1e6
        );
    }

    Ok(())
}
