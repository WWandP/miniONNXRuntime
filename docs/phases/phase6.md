# Phase6

## 看什么

- GPT prompt 怎么进入模型
- baseline 文本生成怎么跑
- KV cache 路径怎么跑
- `output_text`、`full_token_ids`、`provider execution summary` 分别代表什么

## 运行命令

baseline：

```bash
./scripts/run_phase.sh phase6
```

KV cache：

```bash
./scripts/run_phase.sh phase6-kv
```

## 输出重点

- `last_token_topk`
- `full_token_ids`
- `input_text`
- `input_token_ids`
- `output_text`
- `summary`
- `provider execution summary`

## 进一步说明

- GPT 模型资产已经随仓库提供
- `phase6` 适合看 baseline 文本生成
- `phase6-kv` 适合看 KV cache 路径
- 如果你要看更底层的开发记录，可以回到仓库里的 GPT 相关开发文档

## 适合谁看

想把项目从视觉模型扩展到文本模型时看这里。
