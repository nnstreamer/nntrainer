import numpy as np

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # stack along last dim: (-x2, x1) -> [..., seq, dim/2, 2]
    res = np.stack((-x2, x1), axis=-1)
    # flatten last two dims: [..., seq, dim]
    return res.reshape(x.shape)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, heads, seq, dim]
    # cos, sin: [batch, seq, dim] (original, non-interleaved)
    
    # Interleave cos/sin
    # cos: [..., dim] -> [..., dim/2] -> [..., dim/2, 2] -> [..., dim]
    dim = cos.shape[-1]
    cos_half = cos[..., :dim//2]
    sin_half = sin[..., :dim//2]
    
    cos_interleaved = np.repeat(cos_half, 2, axis=-1)
    sin_interleaved = np.repeat(sin_half, 2, axis=-1)
    
    # Broadcast to [batch, heads, seq, dim]
    # cos_interleaved shape [batch, seq, dim] -> [batch, 1, seq, dim]
    cos_expanded = cos_interleaved[:, None, :, :]
    sin_expanded = sin_interleaved[:, None, :, :]
    
    q_embed = (q * cos_expanded) + (rotate_half(q) * sin_expanded)
    k_embed = (k * cos_expanded) + (rotate_half(k) * sin_expanded)
    
    return q_embed, k_embed, cos_expanded, sin_expanded

def cpp_logic_simulation(q, cos_interleaved, sin_interleaved):
    # Simulates the C++ loop logic
    # q: [batch, heads, seq, dim]
    # cos_interleaved: [batch, 1, seq, dim]
    # sin_interleaved: [batch, 1, seq, dim]
    
    # Broadcast cos/sin to q shape explicitly for simulation if needed, 
    # but numpy handles broadcasting in element-wise ops.
    # However, we want to simulate the loop structure.
    
    out = np.zeros_like(q)
    
    # C++ loop: for (i = 0; i < dim; i += 2)
    dim = q.shape[-1]
    for i in range(0, dim, 2):
        in0 = q[..., i]
        in1 = q[..., i+1]
        c = cos_interleaved[..., i]
        s = sin_interleaved[..., i]
        
        # float out0 = in0 * c - in1 * s;
        out[..., i] = in0 * c - in1 * s
        
        # float out1 = in1 * c + in0 * s;
        out[..., i+1] = in1 * c + in0 * s
        
    return out

def verify():
    batch_size = 1
    heads = 2
    seq_len = 5
    head_dim = 16
    
    np.random.seed(42)
    q = np.random.randn(batch_size, heads, seq_len, head_dim).astype(np.float32)
    k = np.random.randn(batch_size, heads, seq_len, head_dim).astype(np.float32)
    
    # Generate random freqs (half dim)
    # Simulate [theta_0, theta_1, ...]
    freqs = np.random.randn(batch_size, seq_len, head_dim // 2).astype(np.float32)
    # emb = cat(freqs, freqs) -> [theta_0, theta_1, ..., theta_0, theta_1, ...]
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_orig = np.cos(emb)
    sin_orig = np.sin(emb)
    
    # Reference implementation
    q_ref, k_ref, cos_used, sin_used = apply_rotary_pos_emb(q, k, cos_orig, sin_orig)
    
    # C++ Simulation
    # cos_used is already interleaved and broadcasted to [batch, 1, seq, dim]
    q_cpp = cpp_logic_simulation(q, cos_used, sin_used)
    k_cpp = cpp_logic_simulation(k, cos_used, sin_used)
    
    # Compare
    max_diff_q = np.abs(q_ref - q_cpp).max()
    max_diff_k = np.abs(k_ref - k_cpp).max()
    
    print(f"Max difference Q: {max_diff_q}")
    print(f"Max difference K: {max_diff_k}")
    
    if max_diff_q < 1e-5 and max_diff_k < 1e-5:
        print("SUCCESS: C++ logic matches Python reference.")
    else:
        print("FAILURE: Mismatch detected.")

if __name__ == "__main__":
    verify()
