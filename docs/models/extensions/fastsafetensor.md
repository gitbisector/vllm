Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

To enable this feature, use the `--load-format fastsafetensors` command-line argument

Requirements
------------

The loader requires a CUDA or ROCm device and the `fastsafetensors` package,
which is an optional dependency:

```console
pip install vllm[fastsafetensors]
```

GPUDirect Storage
-----------------

GPUDirect Storage is used only when tensor parallelism is disabled. With
`--tensor-parallel-size` greater than 1 it is turned off, because initializing
the GDS DMA subsystem opens a driver handle for every visible GPU and so creates
a CUDA context on each of them.

When the checkpoint sits on a filesystem without GDS support, the loader falls
back to buffered reads. The fallback only happens before the first tensor is
yielded; a GDS failure part-way through a load is raised rather than retried,
because restarting would re-read the shards already loaded.

Tuning
------

`VLLM_FASTSAFETENSORS_QUEUE_SIZE` (default `0`) pipelines shard loading: the
producer prepares the next shard's device buffer while the consumer copies the
current shard into model parameters. Each increment keeps one additional
shard-sized buffer resident on the device at peak, so the default preserves the
non-pipelined memory footprint and does not shrink the set of models that fit.
Raise it to overlap I/O with the copy when there is device memory to spare.
