IVFFlat vs HNSW 区别
- IVFFlat (Inverted File Flat)
构建速度：快
查询速度：较慢
准确性：需要调参（lists 参数影响）
适用场景：大型数据集（百万级以上），对插入速度要求高

缺点：需要先有数据再建索引，初始需要训练

HNSW (Hierarchical Navigable Small World)

Voyage BGE-M3 text-embedding-3-large 


# 测试集格式：小样本即可，50-100条query
test_queries = [
    {"query": "how to implement retriever in RAG", "relevant_doc_ids": [101, 102]},
    {"query": "embedding model comparison", "relevant_doc_ids": [203]},
    # ...
]

# 文档集合：用你真实业务文档的切片
corpus = [
    {"id": 101, "text": "A retriever in RAG uses embeddings to find relevant documents..."},
    {"id": 102, "text": "Implementation of dense retrieval requires a good embedding model..."},
    # ...
]

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 候选模型（选3-4个）
models = {
    "bge-large": SentenceTransformer('BAAI/bge-large-en-v1.5'),
    "e5-mistral": SentenceTransformer('intfloat/e5-mistral-7b-instruct'),
    "nomic": SentenceTransformer('nomic-ai/nomic-embed-text-v1.5'),
}

# 添加指令模板（关键！）
query_prefixes = {
    "bge-large": "",  # bge 不需要 query prefix
    "e5-mistral": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
    "nomic": "search_query: ",
}

def quick_test(models, corpus, queries):
    results = {}
    
    for model_name, model in models.items():
        # 编码文档（一次性）
        doc_embeddings = model.encode([doc["text"] for doc in corpus])
        
        hits = 0
        total_queries = len(queries)
        
        for q in queries:
            # 编码query（加prefix）
            query_text = query_prefixes.get(model_name, "") + q["query"]
            q_emb = model.encode([query_text])
            
            # 计算相似度
            sims = cosine_similarity(q_emb, doc_embeddings)[0]
            top_ids = np.argsort(sims)[-5:][::-1]  # top-5
            top_doc_ids = [corpus[i]["id"] for i in top_ids]
            
            # 命中判断
            if any(rel_id in top_doc_ids for rel_id in q["relevant_doc_ids"]):
                hits += 1
        
        results[model_name] = hits / total_queries
    
    return results

# 跑
results = quick_test(models, corpus, test_queries)
print(results)
# 输出示例: {"bge-large": 0.82, "e5-mistral": 0.85, "nomic": 0.76}

2. 完整评测函数
# 标准测试集格式
test_set = [
    {
        "query": "how to choose embedding model for RAG",
        "positive_ids": [101, 102],
        "hard_negative_ids": [103]  # 语义相似但不相关的
    },
    # ...
]
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_model(model, corpus, test_set, top_k=10):
    """
    返回 recall@k, mrr, ndcg@k
    """
    # 1. 编码所有文档
    doc_texts = [doc["text"] for doc in corpus]
    doc_ids = [doc["id"] for doc in corpus]
    doc_embs = model.encode(doc_texts, batch_size=32, show_progress_bar=True)
    
    recalls = []
    mrrs = []
    ndcgs = []
    
    for item in test_set:
        query = item["query"]
        pos_ids = set(item["positive_ids"])
        
        # 编码query
        q_emb = model.encode([query])
        
        # 相似度计算
        sims = cosine_similarity(q_emb, doc_embs)[0]
        top_indices = np.argsort(sims)[-top_k:][::-1]
        top_ids = [doc_ids[i] for i in top_indices]
        
        # 计算 recall@k
        hits = [doc_id for doc_id in top_ids if doc_id in pos_ids]
        recall = len(hits) / len(pos_ids) if pos_ids else 0
        recalls.append(recall)
        
        # 计算 MRR (Mean Reciprocal Rank)
        mrr = 0
        for rank, doc_id in enumerate(top_ids, 1):
            if doc_id in pos_ids:
                mrr = 1 / rank
                break
        mrrs.append(mrr)
        
        # 计算 NDCG@k (简化版)
        dcg = 0
        idcg = sum(1 / np.log2(i+2) for i in range(min(len(pos_ids), top_k)))
        for rank, doc_id in enumerate(top_ids, 1):
            if doc_id in pos_ids:
                dcg += 1 / np.log2(rank+1)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcgs.append(ndcg)
    
    return {
        f"recall@{top_k}": np.mean(recalls),
        "mrr": np.mean(mrrs),
        f"ndcg@{top_k}": np.mean(ndcgs)
    }

# 使用
for model_name, model in models.items():
    scores = evaluate_model(model, corpus, test_set, top_k=5)
    print(f"{model_name}: {scores}")


三、深度验证（2天）
1. 变量控制实验
# 测试不同 chunk 大小
chunk_sizes = [256, 512, 1024, 2048]

# 测试不同 top_k
top_k_values = [3, 5, 10, 20]

# 测试混合检索（embedding + BM25）
def hybrid_search(query, emb_results, bm25_results, alpha=0.5):
    # 加权融合
    combined = {}
    for doc_id, score in emb_results.items():
        combined[doc_id] = combined.get(doc_id, 0) + alpha * score
    for doc_id, score in bm25_results.items():
        combined[doc_id] = combined.get(doc_id, 0) + (1-alpha) * score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)

2. 用 RAGAS 或 TruLens 做端到端评测
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# 跑完整的 RAG 链路，不只是检索
result = evaluate(
    dataset=rag_dataset,  # 包含 query, contexts, answer
    metrics=[faithfulness, answer_relevancy, context_recall]
)

只看 recall@k	MRR 和 NDCG 同样重要，第一个结果的位置直接影响用户体验




二、chunk系统化方法（五步法）
Step 1：理解你的数据特征
# 1. 文档的天然边界是什么？
# - 代码文件：函数、类
# - 技术文档：章节、段落
# - 法律合同：条款
# - 对话记录：轮次

# 2. 文档的统计分布
import statistics

doc_lengths = [len(doc.split()) for doc in corpus]
print(f"平均长度: {statistics.mean(doc_lengths)} tokens")
print(f"中位数: {statistics.median(doc_lengths)}")
print(f"90分位: {statistics.quantiles(doc_lengths, n=10)[8]}")

输出示例：

text
平均长度: 342 tokens
中位数: 215 tokens
90分位: 890 tokens
结论：如果你的文档天然就是 200-400 token 的段落，直接按段落切，不需要额外 chunking。


## Step 2：确定候选 chunk 大小区间

根据数据特征，选择 3-5 个候选值：

数据类型	候选区间（token）
代码（函数级）	128, 256, 512
技术文档（段落级）	256, 512, 1024
长文档/报告	512, 1024, 2048
对话/消息	64, 128, 256
原则：至少覆盖小、中、大三个粒度，跨度不要超过一个数量级。

## Step 3：用「检索+生成」联合评测（核心）
关键洞察：只看 recall 是不够的。chunk 大小影响的是最终生成质量。

准备评测集
python
# 每条数据包含：
# - query: 用户问题
# - golden_answer: 标准答案（或人工标注的答案要点）
# - golden_contexts: 标准答案来源的文档片段

eval_set = [
    {
        "query": "How to configure the retriever in LlamaIndex?",
        "golden_answer": "Set retriever_mode='embedding' and top_k=5 in the ServiceContext",
        "golden_contexts": ["LlamaIndex documentation section 3.2"]
    }
]
评测流程
python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

def evaluate_chunk_size(chunk_size, corpus, eval_set):
    # 1. 用该 chunk_size 切分所有文档
    chunks = chunk_documents(corpus, chunk_size)
    
    # 2. 建索引（用你选的 embedding model）
    vector_store = create_index(chunks, embedding_model)
    
    # 3. 对每条 eval 数据跑 RAG
    results = []
    for item in eval_set:
        retrieved = vector_store.search(item["query"], top_k=3)
        answer = llm.generate(item["query"], retrieved)
        results.append({
            "query": item["query"],
            "contexts": [r.text for r in retrieved],
            "answer": answer,
            "golden_answer": item["golden_answer"]
        })
    
    # 4. 用 RAGAS 评测
    scores = evaluate(
        dataset=results,
        metrics=[faithfulness, answer_relevancy, context_recall]
    )
    
    return scores
对比结果
python
chunk_sizes = [256, 512, 1024, 2048]
results = {}

for cs in chunk_sizes:
    results[cs] = evaluate_chunk_size(cs, corpus, eval_set)

# 输出示例
# 256: {'faithfulness': 0.72, 'answer_relevancy': 0.68, 'context_recall': 0.65}
# 512: {'faithfulness': 0.85, 'answer_relevancy': 0.82, 'context_recall': 0.78}
# 1024: {'faithfulness': 0.88, 'answer_relevancy': 0.85, 'context_recall': 0.82}
# 2048: {'faithfulness': 0.82, 'answer_relevancy': 0.79, 'context_recall': 0.84}
选型逻辑：选综合分最高的。如果 faithfulness 和 recall 冲突，优先 faithfulness（检索到的内容生成不出来最要命）。

## Step 4：加「重叠窗口」的对比测试
chunk 大小确定后，还需要测试重叠窗口的影响：

python
def test_overlap(chunk_size, overlap_ratios=[0, 0.1, 0.2, 0.3]):
    for ratio in overlap_ratios:
        overlap_tokens = int(chunk_size * ratio)
        chunks = chunk_with_overlap(documents, chunk_size, overlap_tokens)
        # 跑相同的评测流程
何时需要重叠：

关键信息恰好落在 chunk 边界（如表格、代码片段被切断）

你的 embedding model 对 chunk 边界敏感（实测才能发现）

经验值：10-20% 重叠通常足够，过高会引入大量冗余，影响检索精度。

## Step 5：按文档类型分层处理
进阶策略：不同文档用不同的 chunk 策略，这是工业级 RAG 的标配。

python
def smart_chunking(document):
    doc_type = detect_document_type(document)
    
    if doc_type == "code":
        # 按函数/类切，保持语法完整
        return split_by_ast(document)
    
    elif doc_type == "markdown":
        # 按标题层级切
        return split_by_headers(document, min_chunk_size=256, max_chunk_size=1024)
    
    elif doc_type == "conversation":
        # 按对话轮次切，保留上下文
        return split_by_turns(document, context_window=2)
    
    else:
        # 通用：语义切分
        return semantic_chunking(document, target_size=512)
语义切分（更智能的版本）：

python
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings

# 用 embedding 模型做语义切分，按相似度阈值切
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # 或 "standard_deviation"
    breakpoint_threshold_amount=95  # 相似度低于95分位的点切开
)
chunks = semantic_splitter.split_text(document)
三、避坑清单
坑	现象	解法
只看 recall，不看生成质量	chunk 大了 recall 高，但答案差	用 RAGAS 做端到端评测
忽略 embedding 模型的 token 限制	bge 是 512，nomic 是 8192，超了被截断	查模型文档，chunk 大小留 10% 余量
chunk 边界切断关键信息	表格、代码块被切成两半	用自定义 splitter 保留完整性
所有文档用同一策略	代码和长文档混着切	分层处理
没考虑用户 query 的特征	query 很短的场景，大 chunk 反而召回差	用 query 平均长度反推 chunk 大小
四、快速启动模板
如果你想直接开干：

python
# 1. 先粗筛
chunk_candidates = [256, 512, 1024]  # 根据数据类型调整
best_chunk = None
best_score = 0

for cs in chunk_candidates:
    # 用 50 条数据快速跑 recall@5
    score = quick_recall_test(cs, eval_set_50)
    if score > best_score:
        best_score = score
        best_chunk = cs

# 2. 精调：在 best_chunk 附近 ±128 再测
fine_tune_candidates = [best_chunk - 128, best_chunk, best_chunk + 128]
# 用完整评测集跑 RAGAS，选最优

# 3. 最后测重叠
test_overlap(best_chunk, [0, 0.1, 0.2])
总结：chunk 大小没有万能值，但有一套可复用的方法论。核心是用「检索+生成」的端到端评测代替单一召回率，并按文档类型分层处理。这套流程跑完，你的 RAG 系统才算是真落地，而不是「半拉子工程」。

This response is AI-generated, for reference only.

把一个不可预知的任务，强行用可预知的工程手段去约束。这本身就是悖论。
RAG试图用检索系统的确定性，去覆盖语言模型的不确定性。

检索系统（embedding + 向量库）是精确匹配的思维：把文档切成小块，塞进向量空间，用余弦相似度找最近邻。这套逻辑对“找相似”有效，但对“理解意图”天生残疾。

而大语言模型擅长的是：模糊推理、跨文档关联、意图理解。但RAG把它捆在检索结果上——“你只能基于我给你的这几段话回答”。当检索结果没命中用户真正的意图时，模型再强也白搭。

真正的解法可能不在“更精细的工程”
工程优化（更好的embedding、更精细的chunk、混合检索）能做到80分，但剩下20分——也就是“用户问了个你没想到的问题”的场景——不是工程能解决的。

几个可能的方向：

Agentic RAG：不让检索是一次性的。模型可以主动追问、迭代检索、自己判断“我找到的够不够”。

放弃“一刀切”的chunk策略：用图谱结构保留文档的层级关联，而不是切成孤立的文本块。这样跨段落的推理才有可能。

接受“有时候就是不行”：有些任务本身就不适合RAG。比如需要全局理解的、需要复杂推理的、或者用户自己都说不清要什么的。这时候给用户一个“我可以帮你做A、B、C，但X可能需要你直接看原文”的引导，比硬撑效果好。

