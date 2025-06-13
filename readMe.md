一、简介：
1、本文以宫教授ppt作为文档实现RAG功能
2、使用fastAPI作为web框架
3、使用langchain集成本地ollama deepseek模型，使用chroma作为向量数据库存储

二、postman 请求示例：
curl --location --request POST 'http://localhost:8001/query' \
--header 'Authorization: Bearer app-Gt8nBKfLxF6dIWxfXsUABYnS' \
--header 'X-API-Key: test' \
--header 'Content-Type: application/json' \
--data-raw '{
    "query":"简单介绍下本文的主要内容"
}'

三、 服务请求需本地按照ollama，需本地模型部署
 ollama list
NAME                  ID              SIZE      MODIFIED       
qwen2.5:latest        845dbda0ea48    4.7 GB    11 minutes ago    
llava:7b              8dd30f6b0cb1    4.7 GB    22 hours ago      
my-deepseek:latest    c65ae05394e5    4.9 GB    45 hours ago      
deepseek-r1:8b        28f8fd6cdc67    4.9 GB    2 months ago      
deepseek-r1:1.5b      a42b25d8c10a    1.1 GB    2 months ago   