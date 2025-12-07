# Jarvis WeChat Assistant – Pipeline & Modules

## 层级概览

- **Layer 0 – Storage**
  - `db.py` — SQLite schema & helper
    - tables: `contacts`, `conversations`, `messages`, `reply_candidates`, `training_examples`, `style_profiles`, `contact_hparams`, `llm_failover_events`, `artifacts`
  - `wechat_assistant.db` — 单一数据库文件（Memori 也共用）

- **Layer 1 – Ingestion / Connectors**
  - **WeChat**: UI 自动化脚本将聊天文本传给 `brain.draft_reply`，`brain.py` 负责落库 `messages`
  - **Web / Browser**: `skyvern_task.py` / `skyvern_integration.py` 把网页内容写到 `artifacts`
  - **Filesystem（计划）**: `fs_indexer.py` 扫描本地文件，写入 `artifacts(kind='file', source='local_fs')`

- **Layer 2 – Tools & Memory**
  - **RAG**: `lightrag_integration.py`
    - 每个 `wechat_id` 一个 LightRAG 工作目录
    - `backfill_training_examples_to_rag` / `ensure_index_for_contact` 将 `training_examples` 写入 LightRAG
    - `ensure_index_from_artifacts` 将 `artifacts`（网页/本地文件）按 `(kind, source)` 聚合进 LightRAG
    - `rag_query_with_meta_for_contact` 对特定联系人的检索，返回 `(context_text, meta)`
  - **RPA**: `rpa_engine.py` + `ocr_backends.py` — 屏幕截图、模板匹配、OCR、点击/输入
  - **Offline style 模型**: `nn_vocab.py` / `nn_model.py` / `nn_train.py` / `nn_infer.py` — 基于 `training_examples` 训练 per-contact 小 GRU 模型

- **Layer 3 – Brain / Orchestrator**
  - `brain.py`
    - 解析聊天文本为 `ParsedMessage`
    - 确认 contact/conversation、写入 `messages`
    - 拉取 `style_profile`（`model_id` hash 作为版本）
    - 从 `contact_hparams` 读取 `rag_mode` + `half_life_days`
    - 通过 `wechat_rag.get_rag_context` 拉取 RAG 文本 + meta
    - `_build_prompt` 组 Prompt，先试本地 `StyleNN`，再走 `_call_llm`（Memori 打 tags）
    - 写入 `reply_candidates`；LLM 失败时记录 `llm_failover_events` 并启发式回复
    - `record_sent_reply` 将真实发送的消息写入 `training_examples`（闭环）

- **Layer 4 – Analytics / Dashboard**
  - `memori_integration.py` — Memori + OpenAI instrumentation（tags: wechat_id/conversation_id/message_id/style_profile_version/rag_* 等）
  - `analytics_memori_wechat.py` —
    - `load_memori_events` / `load_training_examples`
    - `ensure_llm_pairs_view` → `v_wechat_llm_pairs`
    - `ensure_llm_dashboard_views` → `v_wechat_llm_candidate_metrics`, `v_wechat_llm_metrics_daily`, `v_wechat_llm_metrics_style_overall`, `v_wechat_llm_metrics_contact_overall`, `v_wechat_llm_failover_daily`
    - `style_profile_metrics` / `contact_prompt_version_metrics` / `style_profile_before_after` 等
  - 前端：`llm_dashboard_app.py`（Streamlit）、`llm_dashboard_notebook.py`

- **Layer 5 – CLI / 工具脚本**
  - `inspect_samples.py`, `prune_samples.py`, `backfill_rag_from_db.py`, `wechat_rag_cli.py`, `skyvern_task.py`, `title_crawler_demo.py`
  - 规划：`fs_indexer.py`、`jarvis_cli.py`、`launcher.py`、`jarvis` 统一入口

## 端到端数据流（WeChat 示例）

1. **UI 抓取**：RPA/监听将聊天文本传给 `brain.draft_reply(chat_text, wechat_id, display_name)`。
2. **落库**：`Database.get_or_create_contact/conversation`，`log_message` 写入 `messages`。
3. **风格 & RAG**：读取 `style_profile`（优先用 `model_id` 作为版本）、`contact_hparams`；调用 `wechat_rag.get_rag_context` 拿到 RAG 文本 + meta。
4. **生成**：`_build_prompt` 拼接，优先 `StyleNN`，否则 `_call_llm`（Memori 打全量 tags：style_profile_version、conversation_id、message_id、rag_*、prompt_version）。
5. **候选落库**：写入 `reply_candidates`；LLM 出错 → `llm_failover_events` + `_heuristic_replies`。
6. **真实发送**：`record_sent_reply` 把上下文+真实回复写入 `training_examples`。
7. **回填 RAG**：`backfill_rag_from_db` 将 `training_examples` 和 `artifacts` 写入 LightRAG。
8. **观测**：Memori 事件与本地表共享 SQLite；`ensure_llm_dashboard_views` 建视图；`llm_dashboard_app` 展示 chosen/edited/latency/风格前后/Failover 等。

## 未来扩展建议

- 抽象大脑管线：`core/brain_core.py` + `channel_wechat/brain_wechat.py`，让 CLI/桌面等入口复用。
- 文件系统索引：`fs_indexer.py` + LightRAG 全局 namespace，按路径 hash 去重、增量更新。
- 全局 RAG：`global_rag_query(question, sources=[fs,wechat,...])`，统一问答。
- Launcher & CLI：`launcher.py` 统一启动/停止组件；`jarvis_cli` 作为唯一入口（system/rag/index 子命令）。
````
