from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import APIKeyHeader
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel
from typing import List, Dict
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
import json
from starlette.responses import StreamingResponse

# 初始化应用
app = FastAPI(title="宫学研习社")

# 配置安全
API_KEY = "test"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API 密钥"
        )
    return api_key


# 加载本地文档
loader = Docx2txtLoader("langchain/file/gl.docx")
data = loader.load()
# 拆分分档
text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_spliter.split_documents(data)

#使用本地ollama生成embeddings
embeddings = OllamaEmbeddings(
    model="deepseek-r1:8b",
    temperature=0,  # 生成确定性嵌入
    base_url="http://localhost:11434"  # Ollama API地址
)
vectorstore = Chroma.from_documents(docs,
                                    embedding=embeddings,
                                    persist_directory="db/chroma_db" # 向量数据库存储路径
 )


# 定义数据模型
class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []


class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.token_buffer = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_buffer += token
        if "\n" in token:
            # 批量发送
            yield f"data: {json.dumps({'token': self.token_buffer})}\n\n"
            self.token_buffer = ""


# 路由
@app.post("/query", response_model=QueryResponse, dependencies=[Depends(get_api_key)])
async def query(
        request: QueryRequest,
        background_tasks: BackgroundTasks
):
    # 连接本地Ollama模型
    llm = Ollama(
        model="deepseek-r1:8b",  # 模型名称（与ollama list显示的一致）
        temperature=0.7,  # 生成文本的随机性（0-1）
        base_url="http://localhost:11434"  # Ollama API地址（默认）
    )

    # 处理历史对话
    chat_history = [(msg["question"], msg["answer"]) for msg in request.history]

    # 创建对话链
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    # 处理流式输出
    if request.stream:
        def stream_generator():
            # 自定义回调处理器
            callback = StreamingCallbackHandler()
            llm = Ollama(model="llama2", temperature=0.1, callbacks=[callback])

            # 重新创建链
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True , # 返回来源文档用于溯源
                memory=memory
            )

            # 执行查询
            result = qa_chain({"question": request.query, "chat_history": chat_history})

            # 发送最后的令牌
            if callback.token_buffer:
                yield f"data: {json.dumps({'token': callback.token_buffer})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 非流式输出
    result = qa_chain({"question": request.query, "chat_history": chat_history})

    # 提取答案和来源
    answer = result["answer"]
    sources = []

    # 可选：从文档中提取来源
    if "source_documents" in result:
        sources = [doc.page_content[:100] for doc in result["source_documents"]]

    return {"answer": answer, "sources": sources}


# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}


# 启动应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)