"""
RAG 增强问答链模块
实现 RAG 增强的问答系统，对比纯 LLM 与 RAG 的回答质量
"""
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass

# LLM API 配置（需要根据实际情况设置）
# 支持 OpenAI、Anthropic、本地模型等


@dataclass
class QAResult:
    """问答结果"""
    question: str
    answer: str
    sources: List[Dict]  # 引用的源文档
    triplets: List[Dict]  # 引用的三元组
    confidence: float  # 置信度
    is_rag: bool  # 是否使用 RAG
    response_time: float  # 响应时间


class BaseLLM:
    """LLM 基类"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """生成文本"""
        raise NotImplementedError
    
    def batch_generate(self, prompts: List[str], system_prompt: str = "") -> List[str]:
        """批量生成"""
        return [self.generate(p, system_prompt) for p in prompts]


class MockLLM(BaseLLM):
    """模拟 LLM（用于测试）"""
    
    def __init__(self, model_name: str = "mock-llm"):
        super().__init__(model_name)
        self.responses = {
            "depression": "抑郁症（Depression）是一种常见的精神障碍，主要表现为持续的情绪低落、兴趣减退、快感缺失等症状。",
            "anxiety": "焦虑障碍（Anxiety Disorder）是一类以过度担忧、恐惧和回避行为为特征的精神障碍。",
            "schizophrenia": "精神分裂症（Schizophrenia）是一种严重的精神障碍，特征包括阳性症状（如幻觉、妄想）和阴性症状（如情感淡漠、社交退缩）。",
            "treatment": "精神障碍的治疗通常包括药物治疗和心理治疗。常用药物包括SSRIs、SNRIs、抗精神病药等。",
            "symptom": "精神障碍的常见症状包括情绪改变、认知障碍、行为异常、睡眠问题等。"
        }
    
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """模拟生成"""
        prompt_lower = prompt.lower()
        
        # 简单匹配
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response
        
        # 默认回复
        return "根据相关医学知识，这个问题涉及精神健康领域。建议咨询专业医生获取准确信息。"


class RAGChain:
    """RAG 问答链"""
    
    def __init__(self, vector_store, llm: BaseLLM = None):
        self.vector_store = vector_store
        self.llm = llm or MockLLM()
        
        # 系统提示词
        self.system_prompt = """你是一个专业的精神健康知识助手，基于 PrimeKG 知识图谱提供准确的信息。
请根据提供的上下文（Context）回答问题。如果上下文中没有相关信息，请如实说明。
始终保持专业、客观的态度，明确说明信息来源。
重要提示：本系统仅供学习参考，不用于临床诊断或治疗决策。"""
    
    def format_sources(self, sources: List[Dict]) -> str:
        """格式化来源信息"""
        if not sources:
            return "未找到相关来源"
        
        formatted = "\n\n**参考来源：**\n"
        for i, source in enumerate(sources, 1):
            text = source.get('text', '')[:200]
            metadata = source.get('metadata', {})
            disease = metadata.get('disease', '未知')
            rel_type = metadata.get('display_relation', metadata.get('relation', '相关'))
            
            formatted += f"{i}. [{disease}] {rel_type}: {text[:100]}...\n"
        
        return formatted
    
    def format_triplets(self, triplets: List[Dict]) -> str:
        """格式化三元组证据"""
        if not triplets:
            return ""
        
        formatted = "\n\n**PrimeKG 三元组证据：**\n"
        for triplet in triplets[:5]:  # 最多显示5个
            head = triplet.get('head', triplet.get('x_name', ''))
            rel = triplet.get('relation', triplet.get('display_relation', ''))
            tail = triplet.get('tail', triplet.get('y_name', ''))
            formatted += f"- ({head}) --[{rel}]--> ({tail})\n"
        
        return formatted
    
    def build_prompt(self, question: str, context: str) -> str:
        """构建提示词"""
        prompt = f"""**问题：**{question}

**Context（来自 PrimeKG 知识图谱）：**
{context}

请基于以上 Context 回答问题。如果 Context 中没有相关信息，请说明并基于你的知识回答。
"""
        return prompt
    
    def answer(self, question: str, top_k: int = 5, 
               use_rag: bool = True) -> QAResult:
        """
        回答问题
        
        Args:
            question: 用户问题
            top_k: 检索的top-k结果
            use_rag: 是否使用RAG
        
        Returns:
            QAResult: 问答结果
        """
        import time
        start_time = time.time()
        
        sources = []
        context = ""
        triplets = []
        
        if use_rag and self.vector_store:
            # 检索相关文档
            retrieved = self.vector_store.search(question, top_k=top_k)
            sources = retrieved
            
            # 提取三元组信息
            for source in retrieved:
                metadata = source.get('metadata', {})
                if metadata.get('type') == 'triplet':
                    triplets.append(metadata)
            
            # 构建上下文
            if sources:
                context_parts = [s.get('text', '') for s in sources]
                context = "\n\n".join(context_parts)
        
        # 构建提示词
        if use_rag and context:
            prompt = self.build_prompt(question, context)
        else:
            prompt = f"**问题：**{question}\n\n请回答这个问题。"
        
        # 生成回答
        answer = self.llm.generate(prompt, self.system_prompt)
        
        # 估计置信度
        if use_rag and sources:
            avg_score = sum(s.get('score', s.get('distance', 0.5)) for s in sources) / len(sources)
            # 转换为置信度（距离越小，置信度越高）
            confidence = max(0.5, 1 - avg_score) if 'distance' in sources[0] else 0.8
        else:
            confidence = 0.5
        
        response_time = time.time() - start_time
        
        return QAResult(
            question=question,
            answer=answer,
            sources=sources,
            triplets=triplets[:5],
            confidence=confidence,
            is_rag=use_rag,
            response_time=response_time
        )
    
    def compare_answers(self, question: str) -> tuple:
        """对比 RAG 和纯 LLM 的回答"""
        pure_llm_result = self.answer(question, use_rag=False)
        rag_result = self.answer(question, use_rag=True)
        
        return pure_llm_result, rag_result


class QuestionClassifier:
    """问题类型分类器"""
    
    def __init__(self):
        self.intent_patterns = {
            'symptom': [
                r'哪些症状', r'有什么症状', r'表现', r'表征', r'symptom',
                r'会出现', r'特征', r'临床表'
            ],
            'drug': [
                r'药物', r'用什么药', r'治疗.*药', r'drug', r'medication',
                r'吃什么药', r'药品', r'用药'
            ],
            'disease': [
                r'是什么', r'属于', r'分类', r'disease', r'障碍',
                r'类型', r'疾病'
            ],
            'gene': [
                r'基因', r'gene', r'遗传', r'DNA', r'变异'
            ],
            'pathway': [
                r'通路', r'pathway', r'机制', r'生物过程'
            ],
            'comparison': [
                r'比较', r'区别', r'差异', r'difference', r'对比',
                r'哪个.*好', r'有什么不同'
            ],
            'similar': [
                r'相似', r'类似', r'similar', r'相关', r'关联'
            ]
        }
    
    def classify(self, question: str) -> str:
        """分类问题类型"""
        question_lower = question.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return intent
        
        return 'general'


class MentalHealthQASystem:
    """精神健康问答系统"""
    
    def __init__(self, vector_store=None, llm: BaseLLM = None):
        self.rag_chain = RAGChain(vector_store, llm)
        self.classifier = QuestionClassifier()
        
        # 预定义问题模板
        self.question_templates = {
            'symptom': [
                "{disease} 有哪些常见症状？",
                "{disease} 的主要临床表现是什么？",
                "哪些症状与 {disease} 相关？"
            ],
            'drug': [
                "{disease} 有哪些治疗药物？",
                "哪些药物可以用于治疗 {disease}？",
                "{disease} 的常用药物是什么？"
            ],
            'similar': [
                "{disease} 与哪些疾病相似？",
                "与 {disease} 相关的疾病有哪些？",
                "哪些疾病与 {disease} 容易混淆？"
            ]
        }
    
    def answer_with_context(self, question: str) -> Dict:
        """带上下文的回答"""
        # 分类问题
        intent = self.classifier.classify(question)
        
        # 获取 RAG 和纯 LLM 的回答
        pure_result, rag_result = self.rag_chain.compare_answers(question)
        
        return {
            'question': question,
            'intent': intent,
            'pure_llm': {
                'answer': pure_result.answer,
                'sources': [],
                'confidence': pure_result.confidence
            },
            'rag': {
                'answer': rag_result.answer,
                'sources': rag_result.sources,
                'triplets': rag_result.triplets,
                'confidence': rag_result.confidence
            },
            'comparison': {
                'has_rag_advantage': len(rag_result.sources) > 0 and 
                                    len(pure_result.answer) < len(rag_result.answer),
                'rag_sources_count': len(rag_result.sources)
            }
        }
    
    def generate_follow_up_questions(self, disease: str) -> List[str]:
        """生成追问问题"""
        questions = []
        
        for intent, templates in self.question_templates.items():
            for template in templates:
                questions.append(template.format(disease=disease))
        
        return questions[:5]


def demo_qa_system():
    """演示问答系统"""
    print("=" * 60)
    print("精神健康问答系统演示")
    print("=" * 60)
    
    # 创建模拟的向量存储
    class MockVectorStore:
        def search(self, query, top_k=5):
            return [
                {
                    'text': f"抑郁症是一种情感障碍，主要表现为情绪低落、兴趣减退。",
                    'id': '1',
                    'score': 0.9,
                    'metadata': {'disease': 'Depression', 'type': 'info'}
                },
                {
                    'text': "抑郁症常见症状包括：持续的情绪低落、失眠或嗜睡、食欲改变、注意力不集中、自责自罪感。",
                    'id': '2',
                    'score': 0.85,
                    'metadata': {'disease': 'Depression', 'type': 'symptom'}
                }
            ]
    
    # 初始化系统
    vector_store = MockVectorStore()
    llm = MockLLM()
    qa_system = MentalHealthQASystem(vector_store, llm)
    
    # 示例问题
    questions = [
        "抑郁症可能关联哪些症状？",
        "焦虑障碍有哪些治疗药物？",
        "精神分裂症是什么？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        
        result = qa_system.answer_with_context(question)
        
        print(f"问题类型: {result['intent']}")
        print(f"\n[纯 LLM 回答]")
        print(result['pure_llm']['answer'])
        print(f"置信度: {result['pure_llm']['confidence']:.2f}")
        
        print(f"\n[RAG 增强回答]")
        print(result['rag']['answer'])
        print(f"置信度: {result['rag']['confidence']:.2f}")
        print(f"引用来源: {result['rag']['sources_count']} 个")
        
        print("=" * 40)


if __name__ == '__main__':
    demo_qa_system()
