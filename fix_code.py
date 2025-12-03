import torch
import flashinfer

torch.manual_seed(0)
torch.cuda.manual_seed(0)

num_local_heads = 128
batch_size = 8
head_dim_ckv = 512
head_dim_kpe = 64
page_size = 1
dcp_size = 8
seq_len = 8192
mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    backend="fa3"
)
q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32).to(0)
kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * seq_len
kv_indices = torch.arange(0, batch_size * seq_len).to(0).int()
q_n = torch.softmax(torch.randn(
    batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
), dim=-1)
q_p = torch.softmax(torch.randn(
    batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
), dim=-1)
cache_kv_all = torch.softmax(torch.randn(
    batch_size * seq_len, 1, head_dim_ckv + head_dim_kpe, dtype=torch.bfloat16, device="cuda"
), dim=-1)
cache_kv = cache_kv_all[..., :head_dim_ckv]
k_p = cache_kv_all[..., head_dim_ckv:]    
for rank in range(dcp_size):
        local_paged_kernel_lens = ((kv_lens - rank - 1) // dcp_size) + 1
        kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * seq_len
        kv_indptr[1 : batch_size + 1] = torch.cumsum(local_paged_kernel_lens, dim=0)
        kv_indptr = kv_indptr[: batch_size + 1]
        indice = []
        offset = 0
        for original_len, l in zip(kv_lens, local_paged_kernel_lens):
            # Convert tensor scalar to Python int
            l_int = int(l.item())
            original_len_int = int(original_len.item())
            indice.append(torch.arange(l_int) * dcp_size + rank + offset)
            offset += original_len_int
