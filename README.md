### Status:
- Tensor parallelism applied to attention block + mlp
- Ring attention (context parallel) working
- optional fp8 with torchao + fa3 fp8 kernels + ring attention comms are in fp8
- Standard torch.compile works, `fullgraph=True` not yet working
- fa3/fa2/flexattention backends
- Accuracy tested for basic DiT
- Applied to Wan 2.1 (results at end)



### Potential Improvements
- fp8 with bigger head size
- tp to embeddings & head, sequence parallel the norms
- staring at the profiles, the ringattention comms occasionally block, would be worth calculating if we're at capacity or if there's an implementation bug.


### Results

Setup: install the package, need to install fa3 separately (can used `install.sh` to reduce compilation time)

Entrypoint: `torchrun --nproc_per_node=8 benchmark/profile_dit.py`

Model size + test from the default script. Run on 8xh100 with `sudo nvidia-smi -i 0 --lock-gpu-clocks 1830,1830`

Baseline (no parallelism)
```
Average inference time: 1150.28 ms
Throughput: 0.87 images/s
```

cp4 + tp2 + torch.compile:
```
Average inference time: 168.31 ms
Throughput: 5.94 images/s
```
which is 6.8x faster (still a little way to go to 8x).


cp4 + tp2 + torch.compile + **fp8**:
```
Average inference time: 160.48 ms
Throughput: 6.23 images/s
```
now 7.1x faster - not nearly as big of a speedup as I would expect, the fp8 attention kernels themselves are only ~10% faster.

The example given only has headdim=32, and it seems like fa3 pads headdims to be at least 64
cf. https://github.com/Dao-AILab/flash-attention/blob/d6dbdaf1d978b05e0eb3653d5cef7c551f2a4e07/hopper/flash_api.cpp#L47

When I bump the headdim to 128, the fp8 kernels are ~50% faster, and the comms are now the bottleneck.


### Library details:

```python
def setup_distributed(tp_dim: int, cp_dim: int) -> Context:
```
inits the global process group & device mesh, and returns a `Context` object, which just keeps track of the two process groups.

```python
class Context:
    tp_pg: dist.ProcessGroup
    cp_pg: dist.ProcessGroup
```

I debated putting these in a global object, but felt a little more explicit to pass them to the nn.Modules themselves (at the cost of a bit of extra bookkeeping).
There is also some chance that this is why torch.compile(fullgraph=True) breaks, so it might be better to change it...


**Tensor parallel**:`tp.py` contains `ColumnParallelLinear`, `RowParallelLinear`

**Ring attention**: `ringattention.py` contains ring_attention itself, as well as two helper functions `split_tensor_cp`, `gather_tensor_cp` to be used sometime before and after the ringattention call.

Often we can apply this around the main loop

```python
x = split_tensor_cp(x, dim=1, ctx=self.ctx)
for blk in self.blocks:
    x = blk(x)
x = gather_tensor_cp(x, dim=1, ctx=self.ctx)
```


**Accuracy tests** in `test_parallel_accuracy.py`:

```sh
torchrun --nproc_per_node=8 test_parallel_accuracy.py
```
Results:
- Applying tp and cp gives mean error 0.005
- torch.compile reduces accuracy (mean err 0.005 -> 0.01), though a trained model might be more stable
- fp8 mean error is 0.1, again might do better with a trained model



## Wan 2.1

Tried to make non-invasive changes via patching

All of my code is in `Wan2.1/wan_patched.py`, apart from actually applying the patch in `generate.py` with

```python
from wan_patched import parallelize_wan
wan_t2v.model = parallelize_wan(wan_t2v.model)
```

There's definitely loads of low hanging fruit - e.g. keeping context parallel across the whole inner loop.

(right now I only do context parallel around the attention calls, just need to work out the details to make it work with rope + sort out the rest of the layernorms etc)

Also haven't even profiled properly, this is just a v0 application of tp / ringattention!


#### Results
1.4B model:

Single gpu: 107s

8 x h100: 37s

This is just cp4 x tp2 (no fp8, no torch.compile)

Output passes the eye test but didn't check deterministically yet


