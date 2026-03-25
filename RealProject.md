"python.terminal.activateEnvironment": false

# fastapicloud_dividend

A project created with FastAPI CLI.

## Quick Start

### Start the development server

```bash
uv run fastapi dev
```

Visit http://localhost:8000

### Deploy to FastAPI Cloud

> FastAPI Cloud is currently in private beta. Join the waitlist at https://fastapicloud.com

```bash
uv sync
uv run fastapi login
uv run fastapi deploy
```

## Project Structure

- `main.py` - Your FastAPI application
- `pyproject.toml` - Project dependencies

## Learn More

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [FastAPI Cloud](https://fastapicloud.com)
在真实工程项目中，不使用 LangChain、LangGraph、CrewAI 等框架，纯用 Python + LLM API 手动实现 AI Agent 不仅完全可行，而且在许多场景下是更优选择。 Anthropic 官方明确建议开发者”从直接使用 LLM API 开始” (当然，各有各的立场)，而非框架。Octomind 等公司在生产环境使用框架12个月后选择完全移除，称”移除后团队更快乐、更高效”。核心原因在于：LLM 应用本质上只需要字符串处理、API 调用和循环——这些 Python 原生就能很好完成。框架的额外抽象层常常成为调试噩梦和灵活性枷锁。

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

Yes — it’s an orchestrated, tool-enabled agent.
The system controls the workflow, while the LLM handles reasoning, routing, and tool selection.

In production, we usually avoid fully autonomous agents and prefer controlled orchestration for reliability and compliance.


We use an orchestrated, tool-enabled agent where the LLM routes requests to SQL, RAG, or web search, with strict guardrails, validation, and fallbacks for production reliability.

4️⃣ 必须补的 5 个生产级能力（缺一个都不算 production）
🔒 1. Guardrails

content safety

PII detection

SQL injection prevention

📊 2. Observability

记录：

chosen route

top-k docs

SQL template

token usage

latency

🔁 3. Retry & Fallback

tool error → retry

confidence low → RAG

everything fails → human escalation

💰 4. Cost Control

max tool calls

max tokens

route cache（same intent reuse）

🧪 5. Evaluation

golden questions

retrieval recall

tool accuracy

hallucination rate

5️⃣ 你现在这个 agent，JD 怎么说才“高级”

把你原来的话升级成：

“I designed a production-grade, tool-routed GenAI agent where the LLM dynamically selects between SQL queries, RAG pipelines, and web search, with strict guardrails, validation layers, and observability to ensure reliability and compliance.”

这句话 非常企业，非常加分。
For authoritative, single-source queries such as contacts or IDs, we intentionally bypassed LLM generation and returned structured SQL results directly to ensure accuracy, brevity, and user trust.
The LLM was used strictly for intent routing rather than answer generation.


In enterprise settings, I would build RAG using Azure Cognitive Search as the vector store for retrieval, then assemble retrieved chunks into a structured prompt in Python, and call Azure OpenAI for the final answer.
I would leverage a framework like LangChain or an internal orchestration library to manage the retrieval-generation workflow, ensure guardrails, logging, and integrate it with CI/CD pipelines for production deployment.


A data lake is a cloud-scale, immutable tape archive with modern indexing and access control.

We store original invoice documents immutably in a data lake so extraction logic can evolve without losing historical accuracy



5-nano
import os
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://haystacked.cognitiveservices.azure.com/",
    api_key=subscription_key
)


import os
from openai import AzureOpenAI

endpoint = "https://haystacked.cognitiveservices.azure.com/"
model_name = "gpt-5-nano"
deployment = "gpt-5-nano"

subscription_key = "<your-api-key>"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)


# fastapicloud_dividend

A project created with FastAPI CLI.

## Quick Start

### Start the development server

```bash
uv run fastapi dev
```

Visit http://localhost:8000

### Deploy to FastAPI Cloud

> FastAPI Cloud is currently in private beta. Join the waitlist at https://fastapicloud.com

```bash
uv run fastapi login
uv run fastapi deploy
```

## Project Structure

- `main.py` - Your FastAPI application
- `pyproject.toml` - Project dependencies

## Learn More

- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [FastAPI Cloud](https://fastapicloud.com)
在真实工程项目中，不使用 LangChain、LangGraph、CrewAI 等框架，纯用 Python + LLM API 手动实现 AI Agent 不仅完全可行，而且在许多场景下是更优选择。 Anthropic 官方明确建议开发者”从直接使用 LLM API 开始” (当然，各有各的立场)，而非框架。Octomind 等公司在生产环境使用框架12个月后选择完全移除，称”移除后团队更快乐、更高效”。核心原因在于：LLM 应用本质上只需要字符串处理、API 调用和循环——这些 Python 原生就能很好完成。框架的额外抽象层常常成为调试噩梦和灵活性枷锁。

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

作者：陶刚
链接：https://www.zhihu.com/question/1985692342427088079/answer/1994349202730394376
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

理解框架的价值需要先看清它们试图解决的核心痛点。LangChain 的核心抽象包括统一的模型接口（避免供应商锁定）、Chains（多步骤串联）、Agents（自主决策）、Tools（外部系统连接）和 Memory（状态管理）。这些抽象确实解决了真实问题：不同 LLM API 格式不统一、单次调用无法完成复杂任务、LLM 无法执行实际动作。LangGraph 则专注于状态图工作流编排，提供 StateGraph、条件边、检查点和 Human-in-the-loop 等能力。它解决了复杂分支逻辑的表达、长运行任务的断点恢复等高级需求。CrewAI 聚焦多 Agent 协作，通过角色定义（Role-Goal-Backstory 框架）和任务委托机制实现团队协作模式。AutoGPT 则尝试实现完全自主的 AI 代理，具备自我提示和反思能力。然而，这些框架也带来显著代价。LangChain 被批评为”抽象套抽象”，调试时需要理解大量内部框架代码；LangGraph 学习曲线陡峭，需要理解图论概念；CrewAI 的多 Agent 调用成本是 LangGraph 的 2-3 倍；AutoGPT 则经常陷入无限循环，OpenAI 官方都称类似系统只是”教育性质”。

Octomind 案例是最具说服力的生产环境实践。这家 AI 测试自动化公司使用 LangChain 超过12个月后决定完全移除。迁移原因包括：抽象层过深导致调试困难、无法实现”动态调整 Agent 可用工具”等需求、API 频繁变更带来的维护负担。移除后的反馈是："一旦我们移除了它（LangChain），就不再需要把我们的需求翻译成 LangChain 认可的概念。我们可以直接写代码了。"

大多数 LLM 应用所需要的，无非就是字符串处理、API 调用、循环，如果做 RAG 的话可能再加个向量数据库。你根本不需要好几层抽象和一大堆依赖，来管理基本的字符串拼接、HTTP 请求和 for/while 循环。


Simon Willison（Django 创始人）对 Agent 的定义极为简洁：”An LLM agent runs tools in a loop to achieve a goal.” 他自己开发了轻量级 LLM CLI 工具，而非使用重型框架。

Chip Huyen（ML 工程专家）则警告过早引入框架会”抽象掉关键细节，使调试困难”，并强调复合误差问题：若每步准确率95%，10步后准确率将降至60%。

Max Woolf 在博客中尖锐批评："LangChain 的问题在于，它把简单的事情变得相对复杂……这种 Agent 工作流就像一座脆弱的纸牌屋，凭良心讲，我没法把它部署到生产环境中。"LangChain CEO Harrison Chase 承认早期版本”抽象过度”，并推出 LangGraph 作为更底层的替代方案。他将框架类比为 Keras——提供高层封装便于上手，但关键是要构建在更低层抽象之上


