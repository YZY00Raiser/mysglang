import pandas as pd

# 完整的参数数据列表（包含PD disaggregation及所有补充参数）
data = [
    # ========== 原有完整参数（已保留） ==========
    # Model and tokenizer
    ["Model and tokenizer", "--model-path\n--model", "None", "Type: str", "A2, A3"],
    ["Model and tokenizer", "--tokenizer-path", "None", "Type: str", "A2, A3"],
    ["Model and tokenizer", "--tokenizer-mode", "auto", "auto, slow", "A2, A3"],
    ["Model and tokenizer", "--tokenizer-worker-num", "1", "Type: int", "A2, A3"],
    ["Model and tokenizer", "--skip-tokenizer-init", "False", "bool flag (set to enable)", "A2, A3"],
    ["Model and tokenizer", "--load-format", "auto", "auto, safetensors", "A2, A3"],
    ["Model and tokenizer", "--model-loader-extra-config", "{}", "Type: str", "A2, A3"],
    ["Model and tokenizer", "--trust-remote-code", "False", "bool flag (set to enable)", "A2, A3"],
    ["Model and tokenizer", "--context-length", "None", "Type: int", "A2, A3"],
    ["Model and tokenizer", "--is-embedding", "False", "bool flag (set to enable)", "A2, A3"],
    ["Model and tokenizer", "--enable-multimodal", "None", "bool flag (set to enable)", "A2, A3"],
    ["Model and tokenizer", "--revision", "None", "Type: str", "A2, A3"],
    ["Model and tokenizer", "--model-impl", "auto", "auto, sglang, transformers", "A2, A3"],

    # HTTP server
    ["HTTP server", "--host", "127.0.0.1", "Type: str", "A2, A3"],
    ["HTTP server", "--port", "30000", "Type: int", "A2, A3"],
    ["HTTP server", "--skip-server-warmup", "False", "bool flag (set to enable)", "A2, A3"],
    ["HTTP server", "--warmups", "None", "Type: str", "A2, A3"],
    ["HTTP server", "--nccl-port", "None", "Type: int", "A2, A3"],
    ["HTTP server", "--fastapi-root-path", "None", "Type: str", "A2, A3"],
    ["HTTP server", "--grpc-mode", "False", "bool flag (set to enable)", "A2, A3"],

    # Quantization and data type
    ["Quantization and data type", "--dtype", "auto", "auto, float16, bfloat16", "A2, A3"],
    ["Quantization and data type", "--quantization", "None", "modelslim", "A2, A3"],
    ["Quantization and data type", "--quantization-param-path", "None", "Type: str", "Special For GPU"],
    ["Quantization and data type", "--kv-cache-dtype", "auto", "auto", "A2, A3"],
    ["Quantization and data type", "--enable-fp32-lm-head", "False", "bool flag (set to enable)", "A2, A3"],
    ["Quantization and data type", "--modelopt-quant", "None", "Type: str", "Special For GPU"],
    ["Quantization and data type", "--modelopt-checkpoint-restore-path", "None", "Type: str", "Special For GPU"],
    ["Quantization and data type", "--modelopt-checkpoint-save-path", "None", "Type: str", "Special For GPU"],
    ["Quantization and data type", "--modelopt-export-path", "None", "Type: str", "Special For GPU"],
    ["Quantization and data type", "--quantize-and-serve", "False", "bool flag (set to enable)", "Special For GPU"],
    ["Quantization and data type", "--rl-quant-profile", "None", "Type: str", "Special For GPU"],

    # Memory and scheduling
    ["Memory and scheduling", "--mem-fraction-static", "None", "Type: float", "A2, A3"],
    ["Memory and scheduling", "--max-running-requests", "None", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--prefill-max-requests", "None", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--max-queued-requests", "None", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--max-total-tokens", "None", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--chunked-prefill-size", "None", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--max-prefill-tokens", "16384", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--schedule-policy", "fcfs", "lpm, fcfs", "A2, A3"],
    ["Memory and scheduling", "--enable-priority-scheduling", "False", "bool flag (set to enable)", "A2, A3"],
    ["Memory and scheduling", "--schedule-low-priority-values-first", "False", "bool flag (set to enable)", "A2, A3"],
    ["Memory and scheduling", "--priority-scheduling-preemption-threshold", "10", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--schedule-conservativeness", "1.0", "Type: float", "A2, A3"],
    ["Memory and scheduling", "--page-size", "128", "Type: int", "A2, A3"],
    ["Memory and scheduling", "--swa-full-tokens-ratio", "0.8", "Type: float", "A2, A3"],
    ["Memory and scheduling", "--disable-hybrid-swa-memory", "False", "bool flag (set to enable)", "A2, A3"],
    ["Memory and scheduling", "--abort-on-priority-when-disabled", "False", "bool flag (set to enable)", "A2, A3"],
    ["Memory and scheduling", "--enable-dynamic-chunking", "False", "bool flag (set to enable)", "A2, A3"],

    # Runtime options
    ["Runtime options", "--device", "None", "Type: str", "A2, A3"],
    ["Runtime options", "--tensor-parallel-size\n--tp-size", "1", "Type: int", "A2, A3"],
    ["Runtime options", "--pipeline-parallel-size\n--pp-size", "1", "Type: int", "A2, A3"],
    ["Runtime options", "--pp-max-micro-batch-size", "None", "Type: int", "A2, A3"],
    ["Runtime options", "--pp-async-batch-depth", "None", "Type: int", "A2, A3"],
    ["Runtime options", "--stream-interval", "1", "Type: int", "A2, A3"],
    ["Runtime options", "--stream-output", "False", "bool flag (set to enable)", "A2, A3"],
    ["Runtime options", "--random-seed", "None", "Type: int", "A2, A3"],
    ["Runtime options", "--constrained-json-whitespace-pattern", "None", "Type: str", "A2, A3"],
    ["Runtime options", "--constrained-json-disable-any-whitespace", "False", "bool flag (set to enable)", "A2, A3"],
    ["Runtime options", "--watchdog-timeout", "300", "Type: float", "A2, A3"],
    ["Runtime options", "--soft-watchdog-timeout", "300", "Type: float", "A2, A3"],
    ["Runtime options", "--dist-timeout", "None", "Type: int", "A2, A3"],
    ["Runtime options", "--base-gpu-id", "0", "Type: int", "A2, A3"],
    ["Runtime options", "--gpu-id-step", "1", "Type: int", "A2, A3"],
    ["Runtime options", "--sleep-on-idle", "False", "bool flag (set to enable)", "A2, A3"],
    ["Runtime options", "--custom-sigquit-handler", "None", "Optional[Callable]", "A2, A3"],

    # Logging
    ["Logging", "--log-level", "info", "Type: str", "A2, A3"],
    ["Logging", "--log-level-http", "None", "Type: str", "A2, A3"],
    ["Logging", "--log-requests", "False", "bool flag (set to enable)", "A2, A3"],
    ["Logging", "--log-requests-level", "2", "0, 1, 2, 3", "A2, A3"],
    ["Logging", "--log-requests-format", "text", "text, json", "A2, A3"],
    ["Logging", "--crash-dump-folder", "None", "Type: str", "A2, A3"],
    ["Logging", "--enable-metrics", "False", "bool flag (set to enable)", "A2, A3"],
    ["Logging", "--enable-metrics-for-all-schedulers", "False", "bool flag (set to enable)", "A2, A3"],
    ["Logging", "--tokenizer-metrics-custom-labels-header", "x-custom-labels", "Type: str", "A2, A3"],
    ["Logging", "--tokenizer-metrics-allowed-custom-labels", "None", "List[str]", "A2, A3"],
    ["Logging", "--bucket-time-to-first-token", "None", "List[float]", "A2, A3"],
    ["Logging", "--bucket-inter-token-latency", "None", "List[float]", "A2, A3"],
    ["Logging", "--bucket-e2e-request-latency", "None", "List[float]", "A2, A3"],
    ["Logging", "--collect-tokens-histogram", "False", "bool flag (set to enable)", "A2, A3"],
    ["Logging", "--prompt-tokens-buckets", "None", "List[str]", "A2, A3"],
    ["Logging", "--generation-tokens-buckets", "None", "List[str]", "A2, A3"],
    ["Logging", "--gc-warning-threshold-secs", "0.0", "Type: float", "A2, A3"],
    ["Logging", "--decode-log-interval", "40", "Type: int", "A2, A3"],
    ["Logging", "--enable-request-time-stats-logging", "False", "bool flag (set to enable)", "A2, A3"],
    ["Logging", "--kv-events-config", "None", "Type: str", "Special for GPU"],
    ["Logging", "--enable-trace", "False", "bool flag (set to enable)", "A2, A3"],
    ["Logging", "--oltp-traces-endpoint", "localhost:4317", "Type: str", "A2, A3"],

    # RequestMetricsExporter configuration
    ["RequestMetricsExporter configuration", "--export-metrics-to-file", "False", "bool flag (set to enable)", "A2, A3"],
    ["RequestMetricsExporter configuration", "--export-metrics-to-file-dir", "None", "Type: str", "A2, A3"],

    # API related
    ["API related", "--api-key", "None", "Type: str", "A2, A3"],
    ["API related", "--admin-api-key", "None", "Type: str", "A2, A3"],
    ["API related", "--served-model-name", "None", "Type: str", "A2, A3"],
    ["API related", "--weight-version", "default", "Type: str", "A2, A3"],
    ["API related", "--chat-template", "None", "Type: str", "A2, A3"],
    ["API related", "--completion-template", "None", "Type: str", "A2, A3"],
    ["API related", "--enable-cache-report", "False", "bool flag (set to enable)", "A2, A3"],
    ["API related", "--reasoning-parser", "None", "deepseek-r1", "A2, A3"],
    ["API related", "--tool-call-parser", "None", "llama, pythonic", "A2, A3"],
    ["API related", "--sampling-defaults", "model", "openai, model", "A2, A3"],

    # Data parallelism
    ["Data parallelism", "--data-parallel-size\n--dp-size", "1", "Type: int", "A2, A3"],
    ["Data parallelism", "--load-balance-method", "round_robin", "round_robin, total_requests, total_tokens", "A2, A3"],
    ["Data parallelism", "--prefill-round-robin-balance", "False", "bool flag (set to enable)", "A2, A3"],

    # Multi-node distributed serving
    ["Multi-node distributed serving", "--dist-init-addr\n--nccl-init-addr", "None", "Type: str", "A2, A3"],
    ["Multi-node distributed serving", "--nnodes", "1", "Type: int", "A2, A3"],
    ["Multi-node distributed serving", "--node-rank", "0", "Type: int", "A2, A3"],

    # Model override args
    ["Model override args", "--json-model-override-args", "{}", "Type: str", "A2, A3"],
    ["Model override args", "--preferred-sampling-params", "None", "Type: str", "A2, A3"],

    # LoRA
    ["LoRA", "--enable-lora", "False", "Bool flag (set to enable)", "A2, A3"],
    ["LoRA", "--max-lora-rank", "None", "Type: int", "A2, A3"],
    ["LoRA", "--lora-target-modules", "None", "all", "A2, A3"],
    ["LoRA", "--lora-paths", "None", "Type: List[str] / JSON objects", "A2, A3"],
    ["LoRA", "--max-loras-per-batch", "8", "Type: int", "A2, A3"],
    ["LoRA", "--max-loaded-loras", "None", "Type: int", "A2, A3"],
    ["LoRA", "--lora-eviction-policy", "lru", "lru, fifo", "A2, A3"],
    ["LoRA", "--lora-backend", "triton", "triton", "A2, A3"],
    ["LoRA", "--max-lora-chunk-size", "16", "16, 32, 64, 128", "Special for GPU"],

    # Kernel Backends (Attention, Sampling, Grammar, GEMM)
    ["Kernel Backends", "--attention-backend", "None", "ascend", "A2, A3"],
    ["Kernel Backends", "--prefill-attention-backend", "None", "ascend", "A2, A3"],
    ["Kernel Backends", "--decode-attention-backend", "None", "ascend", "A2, A3"],
    ["Kernel Backends", "--sampling-backend", "None", "pytorch, ascend", "A2, A3"],
    ["Kernel Backends", "--grammar-backend", "None", "xgrammar", "A2, A3"],
    ["Kernel Backends", "--mm-attention-backend", "None", "ascend_attn", "A2, A3"],
    ["Kernel Backends", "--nsa-prefill-backend", "flashmla_sparse", "flashmla_sparse, flashmla_decode, fa3, tilelang, aiter", "Special for GPU"],
    ["Kernel Backends", "--nsa-decode-backend", "fa3", "flashmla_prefill, flashmla_kv, fa3, tilelang, aiter", "Special for GPU"],
    ["Kernel Backends", "--fp8-gemm-backend", "auto", "auto, deep_gemm, flashinfer_trtllm, cutlass, triton, aiter", "Special for GPU"],
    ["Kernel Backends", "--disable-flashinfer-autotune", "False", "bool flag (set to enable)", "Special for GPU"],

    # Speculative decoding
    ["Speculative decoding", "--speculative-algorithm", "None", "EAGLE3, NEXTN", "A2, A3"],
    ["Speculative decoding", "--speculative-draft-model-path\n--speculative-draft-model", "None", "Type: str", "A2, A3"],
    ["Speculative decoding", "--speculative-draft-model-revision", "None", "Type: str", "A2, A3"],
    ["Speculative decoding", "--speculative-draft-load-format", "None", "auto", "A2, A3"],
    ["Speculative decoding", "--speculative-num-steps", "None", "Type: int", "A2, A3"],
    ["Speculative decoding", "--speculative-eagle-topk", "None", "Type: int", "A2, A3"],
    ["Speculative decoding", "--speculative-num-draft-tokens", "None", "Type: int", "A2, A3"],
    ["Speculative decoding", "--speculative-accept-threshold-single", "1.0", "Type: float", "Special for GPU"],
    ["Speculative decoding", "--speculative-accept-threshold-acc", "1.0", "Type: float", "Special for GPU"],
    ["Speculative decoding", "--speculative-token-map", "None", "Type: str", "A2, A3"],
    ["Speculative decoding", "--speculative-attention-mode", "prefill", "prefill, decode", "A2, A3"],
    ["Speculative decoding", "--speculative-moe-runner-backend", "None", "auto", "A2, A3"],
    ["Speculative decoding", "--speculative-moe-a2a-backend", "None", "ascend_fuseep", "A2, A3"],
    ["Speculative decoding", "--speculative-draft-attention-backend", "None", "ascend", "A2, A3"],
    ["Speculative decoding", "--speculative-draft-model-quantization", "None", "unquant", "A2, A3"],

    # Ngram speculative decoding
    ["Ngram speculative decoding", "--speculative-ngram-min-match-window-size", "1", "Type: int", "Experimental"],
    ["Ngram speculative decoding", "--speculative-ngram-max-match-window-size", "12", "Type: int", "Experimental"],
    ["Ngram speculative decoding", "--speculative-ngram-min-bfs-breadth", "1", "Type: int", "Experimental"],
    ["Ngram speculative decoding", "--speculative-ngram-max-bfs-breadth", "10", "Type: int", "Experimental"],
    ["Ngram speculative decoding", "--speculative-ngram-match-type", "BFS", "BFS, PROB", "Experimental"],
    ["Ngram speculative decoding", "--speculative-ngram-branch-length", "18", "Type: int", "Experimental"],
    ["Ngram speculative decoding", "--speculative-ngram-capacity", "10000000", "Type: int", "Experimental"],

    # Expert parallelism
    ["Expert parallelism", "--expert-parallel-size\n--ep-size\n--ep", "1", "Type: int", "A2, A3"],
    ["Expert parallelism", "--moe-a2a-backend", "none", "none, deepep, ascend_fuseep", "A2, A3"],
    ["Expert parallelism", "--moe-runner-backend", "auto", "auto, triton", "A2, A3"],
    ["Expert parallelism", "--flashinfer-mxfp4-moe-precision", "default", "default, bf16", "Special for GPU"],
    ["Expert parallelism", "--enable-flashinfer-allreduce-fusion", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Expert parallelism", "--deepep-mode", "auto", "normal, low_latency, auto", "A2, A3"],
    ["Expert parallelism", "--deepep-config", "None", "Type: str", "Special for GPU"],
    ["Expert parallelism", "--ep-num-redundant-experts", "0", "Type: int", "A2, A3"],
    ["Expert parallelism", "--ep-dispatch-algorithm", "None", "Type: str", "A2, A3"],
    ["Expert parallelism", "--init-expert-location", "trivial", "Type: str", "A2, A3"],
    ["Expert parallelism", "--enable-eplb", "False", "bool flag (set to enable)", "A2, A3"],
    ["Expert parallelism", "--eplb-algorithm", "auto", "Type: str", "A2, A3"],
    ["Expert parallelism", "--eplb-rebalance-layers-per-chunk", "None", "Type: int", "A2, A3"],
    ["Expert parallelism", "--eplb-min-rebalancing-utilization-threshold", "1.0", "Type: float", "A2, A3"],
    ["Expert parallelism", "--expert-distribution-recorder-mode", "None", "Type: str", "A2, A3"],
    ["Expert parallelism", "--expert-distribution-recorder-buffer-size", "None", "Type: int", "A2, A3"],
    ["Expert parallelism", "--enable-expert-distribution-metrics", "False", "bool flag (set to enable)", "A2, A3"],
    ["Expert parallelism", "--moe-dense-tp-size", "None", "Type: int", "A2, A3"],
    ["Expert parallelism", "--elastic-ep-backend", "None", "none, mooncake", "Special for GPU"],
    ["Expert parallelism", "--mooncake-ib-device", "None", "Type: str", "Special for GPU"],

    # Mamba Cache
    ["Mamba Cache", "--max-mamba-cache-size", "None", "Type: int", "A2, A3"],
    ["Mamba Cache", "--mamba-ssm-dtype", "float32", "float32, bfloat16", "A2, A3"],
    ["Mamba Cache", "--mamba-full-memory-ratio", "0.2", "Type: float", "A2, A3"],
    ["Mamba Cache", "--mamba-scheduler-strategy", "auto", "auto, no_buffer, extra_buffer", "A2, A3"],
    ["Mamba Cache", "--mamba-track-interval", "256", "Type: int", "A2, A3"],

    # Hierarchical cache
    ["Hierarchical cache", "--enable-hierarchical-cache", "False", "bool flag (set to enable)", "A2, A3"],
    ["Hierarchical cache", "--hicache-ratio", "2.0", "Type: float", "A2, A3"],
    ["Hierarchical cache", "--hicache-size", "0", "Type: int", "A2, A3"],
    ["Hierarchical cache", "--hicache-write-policy", "write_through", "write_back, write_through, write_through_selective", "A2, A3"],
    ["Hierarchical cache", "--radix-eviction-policy", "lru", "lru, lfu", "A2, A3"],
    ["Hierarchical cache", "--hicache-io-backend", "kernel", "kernel_ascend, direct", "A2, A3"],
    ["Hierarchical cache", "--hicache-mem-layout", "layer_first", "page_first_direct, page_first_kv_split", "A2, A3"],
    ["Hierarchical cache", "--hicache-storage-backend", "None", "file", "A2, A3"],
    ["Hierarchical cache", "--hicache-storage-prefetch-policy", "best_effort", "best_effort, wait_complete, timeout", "Special for GPU"],
    ["Hierarchical cache", "--hicache-storage-backend-extra-config", "None", "Type: str", "Special for GPU"],

    # LMCache
    ["LMCache", "--enable-lmcache", "False", "bool flag (set to enable)", "Special for GPU"],

    # Offloading
    ["Offloading", "--cpu-offload-gb", "0", "Type: int", "A2, A3"],
    ["Offloading", "--offload-group-size", "-1", "Type: int", "A2, A3"],
    ["Offloading", "--offload-num-in-group", "1", "Type: int", "A2, A3"],
    ["Offloading", "--offload-prefetch-step", "1", "Type: int", "A2, A3"],
    ["Offloading", "--offload-mode", "cpu", "Type: str", "A2, A3"],

    # Args for multi-item scoring
    ["Args for multi-item scoring", "--multi-item-scoring-delimiter", "None", "Type: int", "A2, A3"],

    # Optimization/debug options
    ["Optimization/debug options", "--disable-radix-cache", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--cuda-graph-max-bs", "None", "Type: int", "A2, A3"],
    ["Optimization/debug options", "--cuda-graph-bs", "None", "List[int]", "A2, A3"],
    ["Optimization/debug options", "--disable-cuda-graph", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--disable-cuda-graph-padding", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-profile-cuda-graph", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-cudagraph-gc", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-nccl-nvls", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--enable-symm-mem", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--disable-flashinfer-cutlass-moe-fp4-allgather", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--enable-tokenizer-batch-encode", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--disable-tokenizer-batch-encode", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--disable-outlines-disk-cache", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--disable-custom-all-reduce", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-mscclpp", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--enable-torch-symm-mem", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--disable-overlap-schedule", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-mixed-chunk", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-dp-attention", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-dp-lm-head", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-two-batch-overlap", "False", "bool flag (set to enable)", "Planned"],
    ["Optimization/debug options", "--enable-single-batch-overlap", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--tbo-token-distribution-threshold", "0.48", "Type: float", "Planned"],
    ["Optimization/debug options", "--enable-torch-compile", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-torch-compile-debug-mode", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-piecewise-cuda-graph", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--piecewise-cuda-graph-tokens", "None", "Type: JSON list", "A2, A3"],
    ["Optimization/debug options", "--piecewise-cuda-graph-compiler", "eager", "eager, inductor", "A2, A3"],
    ["Optimization/debug options", "--torch-compile-max-bs", "32", "Type: int", "A2, A3"],
    ["Optimization/debug options", "--piecewise-cuda-graph-max-tokens", "4096", "Type: int", "A2, A3"],
    ["Optimization/debug options", "--torchao-config", "", "Type: str", "Special for GPU"],
    ["Optimization/debug options", "--enable-nan-detection", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-p2p-check", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--triton-attention-reduce-in-fp32", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--triton-attention-num-kv-splits", "8", "Type: int", "Special for GPU"],
    ["Optimization/debug options", "--triton-attention-split-tile-size", "None", "Type: int", "Special for GPU"],
    ["Optimization/debug options", "--delete-ckpt-after-loading", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-memory-saver", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-weights-cpu-backup", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-draft-weights-cpu-backup", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--allow-auto-truncate", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-custom-logit-processor", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--flashinfer-mla-disable-ragged", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--disable-shared-experts-fusion", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--disable-chunked-prefix-cache", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--disable-fast-image-processor", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--keep-mm-feature-on-device", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-return-hidden-states", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-return-routed-experts", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--scheduler-recv-interval", "1", "Type: int", "A2, A3"],
    ["Optimization/debug options", "--numa-node", "None", "List[int]", "A2, A3"],
    ["Optimization/debug options", "--rl-on-policy-target", "None", "fsdp", "Planned"],
    ["Optimization/debug options", "--enable-layerwise-nvtx-marker", "False", "bool flag (set to enable)", "Special for GPU"],
    ["Optimization/debug options", "--enable-attn-tp-input-scattered", "False", "bool flag (set to enable)", "Experimental"],
    ["Optimization/debug options", "--enable-nsa-prefill-context-parallel", "False", "bool flag (set to enable)", "A2, A3"],
    ["Optimization/debug options", "--enable-fused-qk-norm-rope", "False", "bool flag (set to enable)", "Special for GPU"],

    # Dynamic batch tokenizer
    ["Dynamic batch tokenizer", "--enable-dynamic-batch-tokenizer", "False", "bool flag (set to enable)", "A2, A3"],
    ["Dynamic batch tokenizer", "--dynamic-batch-tokenizer-batch-size", "32", "Type: int", "A2, A3"],
    ["Dynamic batch tokenizer", "--dynamic-batch-tokenizer-batch-timeout", "0.002", "Type: float", "A2, A3"],

    # Debug tensor dumps
    ["Debug tensor dumps", "--debug-tensor-dump-output-folder", "None", "Type: str", "A2, A3"],
    ["Debug tensor dumps", "--debug-tensor-dump-layers", "None", "List[int]", "A2, A3"],
    ["Debug tensor dumps", "--debug-tensor-dump-input-file", "None", "Type: str", "A2, A3"],

    # ========== 新增补充参数（PD disaggregation及其他） ==========
    # PD disaggregation
    ["PD disaggregation", "--disaggregation-mode", "null", "null, prefill, decode", "A2, A3"],
    ["PD disaggregation", "--disaggregation-transfer-backend", "mooncake", "ascend", "A2, A3"],
    ["PD disaggregation", "--disaggregation-bootstrap-port", "8998", "Type: int", "A2, A3"],
    ["PD disaggregation", "--disaggregation-decode-tp", "None", "Type: int", "A2, A3"],
    ["PD disaggregation", "--disaggregation-decode-dp", "None", "Type: int", "A2, A3"],
    ["PD disaggregation", "--disaggregation-ib-device", "None", "Type: str", "Special for GPU"],
    ["PD disaggregation", "--disaggregation-decode-enable-offload-kvcache", "False", "bool flag (set to enable)", "A2, A3"],
    ["PD disaggregation", "--disaggregation-decode-enable-fake-auto", "False", "bool flag (set to enable)", "A2, A3"],
    ["PD disaggregation", "--num-reserved-decode-tokens", "512", "Type: int", "A2, A3"],
    ["PD disaggregation", "--disaggregation-decode-polling-interval", "1", "Type: int", "A2, A3"],

    # Encode prefill disaggregation
    ["Encode prefill disaggregation", "--encoder-only", "False", "bool flag (set to enable)", "A2, A3"],
    ["Encode prefill disaggregation", "--language-only", "False", "bool flag (set to enable)", "A2, A3"],
    ["Encode prefill disaggregation", "--encoder-transfer-backend", "zmq_to_scheduler", "zmq_to_scheduler, zmq_to_tokenizer, mooncake", "A2, A3"],
    ["Encode prefill disaggregation", "--encoder-urls", "[]", "List[str]", "A2, A3"],

    # Custom weight loader
    ["Custom weight loader", "--custom-weight-loader", "None", "List[str]", "A2, A3"],
    ["Custom weight loader", "--weight-loader-disable-mmap", "False", "bool flag (set to enable)", "A2, A3"],
    ["Custom weight loader", "--remote-instance-weight-loader-seed-instance-ip", "None", "Type: str", "A2, A3"],
    ["Custom weight loader", "--remote-instance-weight-loader-seed-instance-service-port", "None", "Type: int", "A2, A3"],
    ["Custom weight loader", "--remote-instance-weight-loader-send-weights-group-ports", "None", "Type: JSON list", "A2, A3"],
    ["Custom weight loader", "--remote-instance-weight-loader-backend", "nccl", "transfer_engine, nccl", "A2, A3"],
    ["Custom weight loader", "--remote-instance-weight-loader-start-seed-via-transfer-engine", "False", "bool flag (set to enable)", "Special for GPU"],

    # For PD-Multiplexing
    ["For PD-Multiplexing", "--enable-pdmux", "False", "bool flag (set to enable)", "Special for GPU"],
    ["For PD-Multiplexing", "--pdmux-config-path", "None", "Type: str", "Special for GPU"],
    ["For PD-Multiplexing", "--sm-group-num", "8", "Type: int", "Special for GPU"],

    # For Multi-Modal
    ["For Multi-Modal", "--mm-max-concurrent-calls", "32", "Type: int", "A2, A3"],
    ["For Multi-Modal", "--mm-per-request-timeout", "10.0", "Type: float", "A2, A3"],
    ["For Multi-Modal", "--enable-broadcast-mm-inputs-process", "False", "bool flag (set to enable)", "A2, A3"],
    ["For Multi-Modal", "--mm-process-config", "None", "Type: JSON / Dict", "A2, A3"],
    ["For Multi-Modal", "--mm-enable-dp-encoder", "False", "bool flag (set to enable)", "A2, A3"],
    ["For Multi-Modal", "--limit-mm-data-per-request", "None", "Type: JSON / Dict", "A2, A3"],

    # For checkpoint decryption
    ["For checkpoint decryption", "--decrypted-config-file", "None", "Type: str", "A2, A3"],
    ["For checkpoint decryption", "--decrypted-draft-config-file", "None", "Type: str", "A2, A3"],
    ["For checkpoint decryption", "--enable-prefix-mm-cache", "False", "bool flag (set to enable)", "A2, A3"],

    # For deterministic inference
    ["For deterministic inference", "--enable-deterministic-inference", "False", "bool flag (set to enable)", "Planned"],

    # For registering hooks
    ["For registering hooks", "--forward-hooks", "None", "Type: JSON list", "A2, A3"],

    # Configuration file support
    ["Configuration file support", "--config", "None", "Type: str", "A2, A3"],

    # Other Params (not supported for NPU)
    ["Other Params (not supported for NPU)", "--checkpoint-engine-wait-weights-before-ready", "False", "bool flag (set to enable)", "Not supported for NPU"],
    ["Other Params (not supported for NPU)", "--kt-weight-path", "None", "Type: str", "Not supported for NPU"],
    ["Other Params (not supported for NPU)", "--kt-method", "AMXINT4", "Type: str", "Not supported for NPU"],
    ["Other Params (not supported for NPU)", "--kt-cpuinfer", "None", "Type: int", "Not supported for NPU"],
    ["Other Params (not supported for NPU)", "--kt-threadpool-count", "2", "Type: int", "Not supported for NPU"],
    ["Other Params (not supported for NPU)", "--kt-num-gpu-experts", "None", "Type: int", "Not supported for NPU"],
    ["Other Params (not supported for NPU)", "--kt-max-deferred-experts-per-token", "None", "Type: int", "Not supported for NPU"],

    # Other Params (functional deficiencies on community)
    ["Other Params (functional deficiencies)", "--enable-double-sparsity", "False", "bool flag (set to enable)", "Functional deficiencies"],
    ["Other Params (functional deficiencies)", "--ds-channel-config-path", "None", "Type: str", "Functional deficiencies"],
    ["Other Params (functional deficiencies)", "--ds-heavy-channel-num", "32", "Type: int", "Functional deficiencies"],
    ["Other Params (functional deficiencies)", "--ds-heavy-token-num", "256", "Type: int", "Functional deficiencies"],
    ["Other Params (functional deficiencies)", "--ds-heavy-channel-type", "qk", "Type: str", "Functional deficiencies"],
    ["Other Params (functional deficiencies)", "--ds-sparse-decode-threshold", "4096", "Type: int", "Functional deficiencies"],
    ["Other Params (functional deficiencies)", "--tool-server", "None", "Type: str", "Functional deficiencies"]
]

# 创建DataFrame
df = pd.DataFrame(
    data,
    columns=["分类", "参数名", "默认值", "可选值/类型", "支持的服务器"]
)

# 生成Excel文件
output_file = "Ascend_NPU_Support_Features_Full.xlsx"
df.to_excel(output_file, index=False, engine="openpyxl")

print(f"✅ 完整的Excel文件已生成：{output_file}")
print(f"📊 共生成 {len(df)} 条参数记录（包含所有补充的PD disaggregation及其他参数）")
