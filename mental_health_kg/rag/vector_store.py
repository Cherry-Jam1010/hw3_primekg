"""
RAG 向量数据库构建模块
将精神障碍子图的文本描述切块后存入向量数据库
"""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
import hashlib

# 可选的向量数据库
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextChunker:
    """文本分块器"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, source: str = "", metadata: Dict = None) -> List[Dict]:
        """将长文本切分成小块"""
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        text = text.strip()
        
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += " " + para if current_chunk else para
            else:
                if current_chunk:
                    chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk,
                        'source': source,
                        'metadata': metadata or {},
                        'char_count': len(current_chunk)
                    })
                
                # 处理重叠
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para
        
        # 添加最后一个块
        if current_chunk:
            chunk_id = hashlib.md5(current_chunk.encode()).hexdigest()[:8]
            chunks.append({
                'id': chunk_id,
                'text': current_chunk,
                'source': source,
                'metadata': metadata or {},
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    def chunk_dataframe(self, df: pd.DataFrame, 
                       text_column: str, 
                       id_column: str = None,
                       metadata_columns: List[str] = None) -> List[Dict]:
        """将 DataFrame 中的文本列切分"""
        all_chunks = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column])
            source_id = str(row[id_column]) if id_column and id_column in row else str(idx)
            
            metadata = {}
            if metadata_columns:
                for col in metadata_columns:
                    if col in row:
                        metadata[col] = str(row[col])
            
            chunks = self.chunk_text(text, source=source_id, metadata=metadata)
            all_chunks.extend(chunks)
        
        return all_chunks


class VectorStore:
    """向量存储基类"""
    
    def __init__(self, collection_name: str = "mental_health_kg"):
        self.collection_name = collection_name
        self.chunks = []
        self.embeddings = None
    
    def add_chunks(self, chunks: List[Dict]):
        """添加文本块"""
        self.chunks.extend(chunks)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索最相关的文本块"""
        raise NotImplementedError
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """获取指定ID的文本块"""
        for chunk in self.chunks:
            if chunk.get('id') == chunk_id:
                return chunk
        return None


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量存储"""
    
    def __init__(self, persist_directory: str = "data/chroma_db",
                 collection_name: str = "mental_health_kg"):
        super().__init__(collection_name)
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Mental Health Knowledge Graph RAG Store"}
        )
        
        # 加载嵌入模型
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.encoder = None
            print("Warning: sentence-transformers not installed. Using TF-IDF fallback.")
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量"""
        if self.encoder:
            return self.encoder.encode(texts, convert_to_numpy=True)
        else:
            # 简单的 TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=384)
            return vectorizer.fit_transform(texts).toarray()
    
    def add_chunks(self, chunks: List[Dict]):
        """添加文本块到向量数据库"""
        super().add_chunks(chunks)
        
        if not chunks:
            return
        
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        metadatas = [{**chunk.get('metadata', {}), 'source': chunk.get('source', '')}
                     for chunk in chunks]
        
        # 编码
        embeddings = self._encode_texts(texts)
        
        # 添加到集合
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"已添加 {len(chunks)} 个文本块到向量数据库")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索最相关的文本块"""
        # 编码查询
        query_embedding = self._encode_texts([query])[0]
        
        # 搜索
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # 格式化结果
        formatted_results = []
        if results and 'documents' in results:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'text': doc,
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else 0,
                    'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def search_by_relation(self, relation_type: str, top_k: int = 10) -> List[Dict]:
        """按关系类型搜索"""
        results = self.collection.query(
            query_texts=[relation_type],
            n_results=top_k,
            where={"relation_type": relation_type}
        )
        
        return results


class SimpleVectorStore(VectorStore):
    """简单的内存向量存储（无需额外依赖）"""
    
    def __init__(self, collection_name: str = "mental_health_kg"):
        super().__init__(collection_name)
        
        # 简单的 TF-IDF 向量化
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.text_vectors = None
        self.texts = []
    
    def add_chunks(self, chunks: List[Dict]):
        """添加文本块"""
        super().add_chunks(chunks)
        
        if not chunks:
            return
        
        self.texts = [chunk['text'] for chunk in chunks]
        self.text_vectors = self.vectorizer.fit_transform(self.texts)
        
        print(f"已添加 {len(chunks)} 个文本块到向量存储")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索最相关的文本块"""
        if self.text_vectors is None or len(self.texts) == 0:
            return []
        
        # 编码查询
        query_vector = self.vectorizer.transform([query])
        
        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vector, self.text_vectors)[0]
        
        # 获取 top_k
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'id': self.chunks[idx].get('id', str(idx)),
                'score': float(similarities[idx]),
                'metadata': self.chunks[idx].get('metadata', {})
            })
        
        return results


def prepare_mental_health_documents(edges_df: pd.DataFrame, 
                                    node_descriptions: Dict = None) -> List[Dict]:
    """
    从 PrimeKG 边数据准备 RAG 文档
    """
    documents = []
    
    # 疾病描述模板
    disease_template = "{disease} 是一种{disorder_type}，常见症状包括{symptoms}，常用治疗药物包括{drugs}。"
    
    # 按疾病分组
    disease_groups = edges_df.groupby('x_name')
    
    for disease, group in disease_groups:
        # 获取相关的症状、药物等
        phenotypes = group[group['y_type'] == 'phenotype']['y_name'].tolist()
        drugs = group[group['y_type'] == 'drug']['y_name'].tolist()
        genes = group[group['y_type'] == 'gene']['y_name'].tolist()
        pathways = group[group['y_type'] == 'pathway']['y_name'].tolist()
        
        # 构建文档文本
        doc_texts = []
        
        # 基本信息
        basic_info = f"{disease} 的基本信息："
        if phenotypes:
            basic_info += f" 症状表现：{', '.join(phenotypes[:10])}"
        doc_texts.append(basic_info)
        
        # 治疗信息
        if drugs:
            treatment_info = f"{disease} 的治疗药物：{', '.join(drugs[:10])}"
            doc_texts.append(treatment_info)
        
        # 生物学机制
        if genes:
            mechanism_info = f"{disease} 相关的基因：{', '.join(genes[:10])}"
            doc_texts.append(mechanism_info)
        
        # 详细描述
        if node_descriptions and disease in node_descriptions:
            doc_texts.append(f"{disease} 详细说明：{node_descriptions[disease]}")
        
        # 为每个文本片段创建文档
        for text in doc_texts:
            if len(text) > 50:  # 过滤太短的文本
                chunk_id = hashlib.md5(text.encode()).hexdigest()[:8]
                documents.append({
                    'id': chunk_id,
                    'text': text,
                    'source': disease,
                    'metadata': {
                        'disease': disease,
                        'num_symptoms': len(phenotypes),
                        'num_drugs': len(drugs),
                        'type': 'disease_info'
                    }
                })
        
        # 为每条边创建三元组文档
        for _, row in group.head(20).iterrows():  # 限制数量
            triplet_text = f"{row['x_name']} --[{row.get('relation', 'related_to')}]--> {row['y_name']}"
            triplet_text += f"（关系类型：{row.get('display_relation', '相关')})"
            
            chunk_id = hashlib.md5(triplet_text.encode()).hexdigest()[:8]
            documents.append({
                'id': chunk_id,
                'text': triplet_text,
                'source': disease,
                'metadata': {
                    'disease': disease,
                    'relation': row.get('relation', ''),
                    'display_relation': row.get('display_relation', ''),
                    'y_type': row.get('y_type', ''),
                    'y_name': row.get('y_name', ''),
                    'type': 'triplet'
                }
            })
    
    return documents


def build_vector_store(edges_df: pd.DataFrame, 
                      output_dir: str = "data",
                      vector_store_type: str = "simple") -> VectorStore:
    """
    构建向量存储
    
    Args:
        edges_df: 边数据 DataFrame
        output_dir: 输出目录
        vector_store_type: 向量存储类型 ("chroma" 或 "simple")
    """
    print("正在准备文档...")
    
    # 准备文档
    documents = prepare_mental_health_documents(edges_df)
    print(f"生成了 {len(documents)} 个文档")
    
    # 创建向量存储
    if vector_store_type == "chroma" and CHROMADB_AVAILABLE:
        store = ChromaVectorStore(
            persist_directory=os.path.join(output_dir, "chroma_db"),
            collection_name="mental_health_kg"
        )
    else:
        store = SimpleVectorStore(collection_name="mental_health_kg")
    
    # 添加文档
    store.add_chunks(documents)
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, "rag_documents.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            'num_documents': len(documents),
            'chunks': documents
        }, f, ensure_ascii=False, indent=2)
    
    print(f"RAG 文档已保存: {metadata_path}")
    
    return store


if __name__ == '__main__':
    print("向量数据库构建模块")
    print("需要提供边数据 DataFrame 来构建向量存储")
