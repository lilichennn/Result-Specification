# ICASSP 2027 Code

## 环境配置

安装 [uv](https://docs.astral.sh/uv/) 后，在仓库根目录执行：

```bash
uv sync --locked
```

运行主项目脚本时使用 `uv run`，例如：

```bash
uv run python scripts/preprocess.py --dataset bird --split dev
```

根目录锁文件同时包含当前实验所需的 DeepEye-SQL 运行依赖。
CHESS 仍保留其原始 `requirements.txt` 作为独立基线的环境定义。

不要提交 `.venv` 或任何 `.env`。本地模型配置保存在
`config/.env`。
