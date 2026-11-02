"""modules 层：core 接口的具体实现。

- llm         ：OpenAI 兼容的 LLM 客户端（DeepSeek / OpenAI / Kimi / Qwen ...）
- embedder    ：本地 BGE 嵌入模型
- vectorstore ：内存向量索引，含 BM25 关键词降级
- chunking    ：网页正文分块
- tools       ：联网搜索 + 网页正文抓取
- mcp_client  ：MCP Client（stdio 常驻会话，供 LangGraph 节点同步调用外部工具服务）
- tracing     ：本地 trace 落盘 + 可选 Langfuse 上报
- agents      ：Planner / Researcher / Knowledge / Writer / Reviewer 五个角色实现

这一层允许依赖第三方 SDK，services 层通过 core 接口或本层具体实现组装业务。
"""
