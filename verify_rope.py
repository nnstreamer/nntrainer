import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    original_dtype = q.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Interleave them instead of usual shape
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    
    q_embed = (q.float() * cos) + (rotate_half(q).float() * sin)
    k_embed = (k.float() * cos) + (rotate_half(k).float() * sin)
    
    return q_embed.to(original_dtype), k_embed.to(original_dtype)

def cpp_logic_simulation(q, cos_interleaved, sin_interleaved):
    # Simulates the C++ loop logic
    # q: [batch, heads, seq, dim]
    # cos_interleaved: [batch, 1, seq, dim]
    # sin_interleaved: [batch, 1, seq, dim]
    
    # Broadcast cos/sin to q shape
    cos = cos_interleaved.expand_as(q)
    sin = sin_interleaved.expand_as(q)
    
    out = torch.zeros_like(q)
    
    # C++ loop: for (i = 0; i < dim; i += 2)
    for i in range(0, q.shape[-1], 2):
        in0 = q[..., i]
        in1 = q[..., i+1]
        c = cos[..., i]
        s = sin[..., i]
        
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
    
    q = torch.randn(batch_size, heads, seq_len, head_dim)
    k = torch.randn(batch_size, heads, seq_len, head_dim)
    
    # Generate random freqs (half dim)
    freqs = torch.randn(batch_size, seq_len, head_dim // 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_orig = emb.cos()
    sin_orig = emb.sin()
    
    # Reference implementation
    q_ref, k_ref = apply_rotary_pos_emb(q, k, cos_orig, sin_orig)
    
    # Prepare interleaved cos/sin for C++ simulation
    # In C++ precompute_freqs, we store interleaved values
    cos_interleaved = cos_orig[..., : head_dim // 2].repeat_interleave(2, dim=-1)
    sin_interleaved = sin_orig[..., : head_dim // 2].repeat_interleave(2, dim=-1)
    
    # Add unsqueeze_dim=1 to match broadcasting in apply_rotary_pos_emb
    cos_interleaved = cos_interleaved.unsqueeze(1)
    sin_interleaved = sin_interleaved.unsqueeze(1)
    
    # C++ Simulation
    q_cpp = cpp_logic_simulation(q, cos_interleaved, sin_interleaved)
    k_cpp = cpp_logic_simulation(k, cos_interleaved, sin_interleaved)
    
    # Compare
    max_diff_q = (q_ref - q_cpp).abs().max().item()
    max_diff_k = (k_ref - k_cpp).abs().max().item()
    
    print(f"Max difference Q: {max_diff_q}")
    print(f"Max difference K: {max_diff_k}")
    
    if max_diff_q < 1e-6 and max_diff_k < 1e-6:
        print("SUCCESS: C++ logic matches Python reference.")
    else:
        print("FAILURE: Mismatch detected.")

if __name__ == "__main__":
    verify()
