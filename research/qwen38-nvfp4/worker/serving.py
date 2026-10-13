"""Common serving controls; recorded in every immutable generation manifest."""

ENGINE_OPTIONS = {
    "enforce_eager": False,
    "max_num_seqs": 16,
    "max_num_batched_tokens": 8192,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": False,
    "speculative_config": None,
}
