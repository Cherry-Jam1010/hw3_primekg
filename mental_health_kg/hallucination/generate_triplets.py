"""
LLM 幻觉检测模块
从精神障碍子图中生成真实/伪造三元组，评估 LLM 的幻觉倾向
"""
from __future__ import annotations

import os
import json
import random
import pandas as pd
import numpy as np
from collections import defaultdict

# 随机种子
random.seed(42)
np.random.seed(42)


class TripletGenerator:
    """三元组生成器"""
    
    def __init__(self, edges_df):
        self.edges_df = edges_df
        self.entities = set()
        self.relations = set()
        self.triplets = []
        
        # 按关系类型分组的三元组
        self.triplets_by_relation = defaultdict(list)
        
        # 节点类型映射（基于 PrimeKG 列）
        self.node_types = defaultdict(set)
        
        self._process_data()
    
    def _process_data(self):
        """处理数据"""
        for _, row in self.edges_df.iterrows():
            head = str(row['x_name'])
            relation = str(row.get('relation', row.get('display_relation', 'related_to')))
            tail = str(row['y_name'])
            
            if pd.notna(head) and pd.notna(relation) and pd.notna(tail):
                self.entities.add(head)
                self.entities.add(tail)
                self.relations.add(relation)
                
                triplet = (head, relation, tail)
                self.triplets.append(triplet)
                self.triplets_by_relation[relation].append(triplet)
                
                # 记录节点类型
                self.node_types[head].add(row['x_type'])
                self.node_types[tail].add(row['y_type'])
        
        print(f"共 {len(self.entities)} 个实体, {len(self.relations)} 种关系")
        print(f"共 {len(self.triplets)} 个三元组")
    
    def get_real_triplets(self, n=50, relation_filter=None) -> List[Tuple]:
        """获取真实三元组"""
        if relation_filter:
            triplets = []
            for rel in relation_filter:
                if rel in self.triplets_by_relation:
                    triplets.extend(self.triplets_by_relation[rel])
        else:
            triplets = self.triplets.copy()
        
        if len(triplets) <= n:
            return triplets
        
        return random.sample(triplets, n)
    
    def get_entities_by_type(self, entity_type='disease') -> set:
        """获取特定类型的实体"""
        entities = set()
        for entity, types in self.node_types.items():
            if entity_type in types:
                entities.add(entity)
        return entities
    
    def generate_false_triplets(self, n=50, strategy='replace_tail') -> List[Tuple]:
        """
        生成伪造三元组
        策略:
        - replace_tail: 替换尾实体
        - replace_head: 替换头实体
        - replace_relation: 替换关系
        - swap: 交换头尾
        """
        false_triplets = []
        
        for _ in range(n):
            # 随机选择一个真实三元组作为模板
            real_triplet = random.choice(self.triplets)
            head, relation, tail = real_triplet
            
            # 随机选择策略
            if strategy == 'random':
                strategy = random.choice(['replace_tail', 'replace_head', 'replace_relation'])
            
            if strategy == 'replace_tail':
                # 替换尾实体（保持头和关系不变）
                other_tails = [t for h, r, t in self.triplets if h == head and t != tail]
                if other_tails:
                    new_tail = random.choice(other_tails)
                    false_triplet = (head, relation, new_tail)
                else:
                    new_tail = random.choice(list(self.entities - {tail}))
                    false_triplet = (head, relation, new_tail)
            
            elif strategy == 'replace_head':
                # 替换头实体
                other_heads = [h for h, r, t in self.triplets if t == tail and h != head]
                if other_heads:
                    new_head = random.choice(other_heads)
                    false_triplet = (new_head, relation, tail)
                else:
                    new_head = random.choice(list(self.entities - {head}))
                    false_triplet = (new_head, relation, tail)
            
            elif strategy == 'replace_relation':
                # 替换关系
                other_relations = list(self.relations - {relation})
                if other_relations:
                    new_relation = random.choice(other_relations)
                    false_triplet = (head, new_relation, tail)
                else:
                    false_triplet = (head, "unknown_relation", tail)
            
            else:
                false_triplet = real_triplet
            
            # 确保伪造的三元组不在真实数据中
            if false_triplet not in self.triplets:
                false_triplets.append(false_triplet)
            else:
                # 如果恰好在真实数据中，尝试另一种策略
                new_tail = random.choice(list(self.entities - {tail}))
                false_triplets.append((head, relation, new_tail))
        
        return false_triplets


class HallucinationDataset:
    """幻觉检测数据集"""
    
    def __init__(self, triplet_generator: TripletGenerator):
        self.generator = triplet_generator
        
        # 关系分类
        self.relation_categories = {
            'disease_symptom': ['phenotype', 'associated_phenotype', 'manifestation', 'symptom'],
            'disease_drug': ['drug', 'treats', 'interacts', 'medication'],
            'disease_disease': ['disease', 'comorbidity', 'similar_to', 'parent_of'],
            'disease_gene': ['gene', 'associated_gene'],
            'disease_pathway': ['pathway', 'biological_process'],
        }
    
    def create_question(self, triplet: Tuple, is_real: bool) -> Dict:
        """
        将三元组转化为自然语言判断题
        """
        head, relation, tail = triplet
        
        # 关系到自然语言的映射
        relation_phrases = {
            'phenotype': '表现为',
            'associated_phenotype': '关联症状',
            'symptom': '有症状',
            'drug': '使用药物',
            'treats': '治疗',
            'interacts': '与药物相互作用',
            'associated_gene': '与基因相关',
            'gene': '相关基因',
            'disease': '是一种',
            'comorbidity': '与...共病',
            'similar_to': '相似于',
            'parent_of': '属于',
            'pathway': '涉及通路',
            'biological_process': '涉及生物过程',
            'anatomy': '涉及解剖结构',
            'effect': '有效果',
            'contraindication': '禁忌',
            'overlaps': '重叠',
            'prevents': '预防',
            ' Side Effect': '副作用',
        }
        
        phrase = relation_phrases.get(relation, relation)
        
        templates = [
            f"{head} {phrase} {tail}。这个说法是正确的吗？",
            f"根据医学知识，{head} {phrase} {tail}。请判断这是否正确。",
            f"请判断：{head} 与 {tail} 之间存在 {phrase} 的关系。",
            f"有一种说法认为：{head} {phrase} {tail}。这是真的吗？",
        ]
        
        question = random.choice(templates)
        
        return {
            'question': question,
            'triplet': triplet,
            'is_real': is_real,
            'head': head,
            'relation': relation,
            'tail': tail,
            'relation_type': self._categorize_relation(relation)
        }
    
    def _categorize_relation(self, relation: str) -> str:
        """分类关系类型"""
        for category, keywords in self.relation_categories.items():
            if any(kw.lower() in relation.lower() for kw in keywords):
                return category
        return 'other'
    
    def generate_dataset(self, real_count=30, false_count=30, 
                        relation_focus=None) -> List[Dict]:
        """生成幻觉检测数据集"""
        dataset = []
        
        # 获取真实三元组
        if relation_focus:
            rel_keywords = self.relation_categories.get(relation_focus, [])
            real_triplets = self.generator.get_real_triplets(
                real_count, 
                relation_filter=rel_keywords if rel_keywords else None
            )
        else:
            real_triplets = self.generator.get_real_triplets(real_count)
        
        # 转换为问题
        for triplet in real_triplets:
            dataset.append(self.create_question(triplet, is_real=True))
        
        # 生成伪造三元组
        false_triplets = self.generator.generate_false_triplets(false_count)
        
        # 转换为问题
        for triplet in false_triplets:
            dataset.append(self.create_question(triplet, is_real=False))
        
        # 打乱顺序
        random.shuffle(dataset)
        
        return dataset
    
    def get_relation_specific_dataset(self, relation_type: str, 
                                      real_count=15, false_count=15) -> List[Dict]:
        """生成特定关系类型的测试集"""
        keywords = self.relation_categories.get(relation_type, [])
        
        real_triplets = self.generator.get_real_triplets(
            real_count * 2, 
            relation_filter=keywords if keywords else None
        )[:real_count]
        
        false_triplets = []
        for triplet in random.sample(real_triplets, min(10, len(real_triplets))):
            head, relation, tail = triplet
            other_entities = list(self.generator.entities - {head, tail})
            if other_entities:
                new_tail = random.choice(other_entities)
                false_triplets.append((head, relation, new_tail))
        
        while len(false_triplets) < false_count:
            false_triplets.append(self.generator.generate_false_triplets(1)[0])
        
        dataset = []
        for triplet in real_triplets:
            dataset.append(self.create_question(triplet, is_real=True))
        for triplet in false_triplets[:false_count]:
            dataset.append(self.create_question(triplet, is_real=False))
        
        random.shuffle(dataset)
        return dataset


def format_triplet_for_display(triplet: Tuple) -> str:
    """格式化三元组用于显示"""
    head, relation, tail = triplet
    return f"({head}) --[{relation}]--> ({tail})"


def analyze_llm_responses(responses: List[Dict]) -> Dict:
    """分析 LLM 的响应"""
    results = {
        'total': len(responses),
        'correct': 0,
        'incorrect': 0,
        'by_relation_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_real_vs_false': {'real': {'correct': 0, 'total': 0}, 
                             'false': {'correct': 0, 'total': 0}},
        'hallucination_rate': 0
    }
    
    for resp in responses:
        is_correct = resp.get('is_correct', False)
        is_real = resp.get('is_real', True)
        rel_type = resp.get('relation_type', 'other')
        
        results['by_relation_type'][rel_type]['total'] += 1
        if is_correct:
            results['correct'] += 1
            results['by_relation_type'][rel_type]['correct'] += 1
        else:
            results['incorrect'] += 1
        
        key = 'real' if is_real else 'false'
        results['by_real_vs_false'][key]['total'] += 1
        if is_correct:
            results['by_real_vs_false'][key]['correct'] += 1
    
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    # 计算各类幻觉率
    for rel_type, stats in results['by_relation_type'].items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
            stats['hallucination_rate'] = 1 - stats['accuracy']
    
    # 真实三元组的错误率 = 遗漏率
    real_stats = results['by_real_vs_false']['real']
    results['real_correct_rate'] = real_stats['correct'] / real_stats['total'] if real_stats['total'] > 0 else 0
    
    # 伪造三元组的正确拒绝率 = 幻觉检测率
    false_stats = results['by_real_vs_false']['false']
    results['false_detection_rate'] = false_stats['correct'] / false_stats['total'] if false_stats['total'] > 0 else 0
    
    # 总体幻觉率 = 1 - 准确率
    results['hallucination_rate'] = 1 - results['accuracy']
    
    return results


def generate_synthetic_results(dataset: List[Dict], llm_accuracy=0.75) -> List[Dict]:
    """
    生成模拟的 LLM 判断结果
    （在没有实际调用 LLM API 时使用）
    """
    results = []
    
    for item in dataset:
        # 模拟 LLM 的判断
        is_real = item['is_real']
        
        # 假设 LLM 有一定概率正确判断
        if random.random() < llm_accuracy:
            # LLM 正确判断
            if is_real:
                llm_answer = 'true'
            else:
                llm_answer = 'false'
            is_correct = True
        else:
            # LLM 错误判断
            if is_real:
                llm_answer = 'false'
            else:
                llm_answer = 'true'
            is_correct = False
        
        # 随机生成理由（简化版）
        reasons = [
            "根据知识图谱数据，这个关系是正确的。",
            "从医学知识来看，这种关系是合理的。",
            "知识图谱中存在这种关联。",
            "这个说法与已知医学知识一致。",
            "这是幻觉，知识图谱中没有这种关系。",
            "这个关系在数据中不存在。",
            "知识图谱不支持这种关联。",
            "这是错误的，正确的三元组应该是不同的。"
        ]
        
        result = {
            **item,
            'llm_answer': llm_answer,
            'llm_reason': random.choice(reasons) if llm_answer == 'true' else random.choice(reasons[4:]),
            'is_correct': is_correct
        }
        
        results.append(result)
    
    return results


if __name__ == '__main__':
    print("请提供 PrimeKG 子图数据来生成幻觉检测数据集")
