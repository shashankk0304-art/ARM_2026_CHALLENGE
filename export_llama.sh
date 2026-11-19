#!/usr/bin/env bash
# Run from the ExecuTorch repository root on a capable PC, not on the KV260.
# Use the official Llama 3.2 1B INSTRUCT checkpoint, params.json and tokenizer.model.
set -euo pipefail
: "${LLAMA_CHECKPOINT:?Set to Llama-3.2-1B-Instruct consolidated.00.pth}"
: "${LLAMA_PARAMS:?Set to the matching params.json}"
OUT=${LLAMA_OUTPUT:-llama3_2_1b_instruct_8da4w_2048.pte}
python -m extension.llm.export.export_llm \
  base.model_class=llama3_2 \
  "base.checkpoint=${LLAMA_CHECKPOINT}" \
  "base.params=${LLAMA_PARAMS}" \
  'base.metadata="{\"get_bos_id\":128000, \"get_eos_ids\":[128009,128001]}"' \
  model.use_kv_cache=True \
  model.use_sdpa_with_kv_cache=True \
  model.dtype_override=fp32 \
  export.max_seq_length=2048 \
  export.max_context_length=2048 \
  "export.output_name=${OUT}" \
  quantization.qmode=8da4w \
  quantization.group_size=32 \
  'quantization.embedding_quantize=4,32' \
  backend.xnnpack.enabled=True \
  backend.xnnpack.extended_ops=True
echo "Exported ${OUT}; use the matching tokenizer.model and a runner built with XNNPACK + quantized kernels."
