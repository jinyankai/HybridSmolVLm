try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print("successfully imported causal_conv1d package")
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
    print("causal_conv1d package not found. Will use slower conv1d implementation")

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None