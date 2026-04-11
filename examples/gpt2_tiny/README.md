# gpt2_tiny

这个目录预留给 transformer / attention 方向的小型实验。

更适合放：

- 简化版 GPT-2 图或子图
- LayerNorm / Gelu / MatMul / Attention 相关样例
- transformer 方向的图优化案例

现在也放了一个文本续写的最小测试样例：

```bash
./build_phase4/miniort_run_gpt --config examples/gpt2_tiny/story_generate.cfg
```

如果你想把故事开头写长一点，直接改：

- [story_prompt.txt](/Volumes/ww/code/onnxruntime/minionnxruntime/examples/gpt2_tiny/story_prompt.txt)

如果你想调生成长度或开关项，改：

- [story_generate.cfg](/Volumes/ww/code/onnxruntime/minionnxruntime/examples/gpt2_tiny/story_generate.cfg)

如果你已经导出了 cache 版模型，还可以直接切到 KV cache 对比配置：

- [kv_cache_generate.cfg](/Volumes/ww/code/onnxruntime/minionnxruntime/examples/gpt2_tiny/kv_cache_generate.cfg)

它默认指向：

- `models/gpt2/model.kv_prefill.onnx`
- `models/gpt2/model.kv_decode.onnx`

对应导出方式见 [docs/phase5_gpt2_kv_cache_bringup.md](/Volumes/ww/code/onnxruntime/minionnxruntime/docs/phase5_gpt2_kv_cache_bringup.md)

如果你想看这条 GPT-2 线从 baseline 到 KV cache 的完整开发记录和对比过程，见：

- [docs/phase5_gpt2_development_journal.md](/Volumes/ww/code/onnxruntime/minionnxruntime/docs/phase5_gpt2_development_journal.md)
