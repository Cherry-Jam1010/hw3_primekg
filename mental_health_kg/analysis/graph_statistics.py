"""
知识图谱统计分析与可视化模块
对精神障碍子图进行深入的统计分析和可视化
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
COLORS = {
    'disease': '#e74c3c',      # 红色
    'symptom': '#27ae60',      # 绿色
    'drug': '#3498db',          # 蓝色
    'gene': '#9b59b6',         # 紫色
    'pathway': '#f39c12',      # 橙色
    'phenotype': '#1abc9c',    # 青色
    'other': '#95a5a6'         # 灰色
}


class GraphStatistics:
    """图统计类"""
    
    def __init__(self, edges_df, G):
        self.edges_df = edges_df
        self.G = G
        self.analysis_results = {}
    
    def basic_statistics(self):
        """基础统计"""
        stats = {
            '节点数': self.G.number_of_nodes(),
            '边数': self.G.number_of_edges(),
            '有向边数': self.G.number_of_edges(),
            '平均度': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes() if self.G.number_of_nodes() > 0 else 0,
            '图密度': nx.density(self.G),
            '弱连通分量数': nx.number_weakly_connected_components(self.G),
            '强连通分量数': nx.number_strongly_connected_components(self.G),
            '平均聚类系数': nx.average_clustering(self.G.to_undirected()),
        }
        
        # 计算直径（在最大弱连通分量上）
        largest_wcc = max(nx.weakly_connected_components(self.G), key=len)
        if len(largest_wcc) > 1:
            subgraph = self.G.subgraph(largest_wcc)
            try:
                stats['直径(最大弱连通分量)'] = nx.diameter(subgraph.to_undirected())
            except:
                stats['直径(最大弱连通分量)'] = -1
        
        self.analysis_results['basic_stats'] = stats
        return stats
    
    def node_type_statistics(self):
        """节点类型统计"""
        # 从边数据中推断节点类型
        node_types = defaultdict(str)
        
        for _, row in self.edges_df.iterrows():
            x_name, y_name = row['x_name'], row['y_name']
            x_type, y_type = row['x_type'], row['y_type']
            
            # 如果已有类型信息，优先使用
            if not node_types[x_name]:
                node_types[x_name] = x_type
            if not node_types[y_name]:
                node_types[y_name] = y_type
        
        # 统计类型分布
        type_counts = Counter(node_types.values())
        
        self.analysis_results['node_types'] = dict(type_counts)
        return dict(type_counts)
    
    def relation_type_statistics(self):
        """关系类型统计"""
        rel_counts = self.edges_df['display_relation'].value_counts()
        
        # 也按 relation 列统计
        rel_counts_detail = self.edges_df['relation'].value_counts()
        
        self.analysis_results['relation_types'] = {
            'display_relation': dict(rel_counts.head(20)),
            'relation': dict(rel_counts_detail.head(20))
        }
        
        return dict(rel_counts.head(20))
    
    def centrality_analysis(self):
        """中心性分析"""
        # 度中心性
        degree_centrality = nx.degree_centrality(self.G)
        
        # 入度/出度
        in_degree = dict(self.G.in_degree())
        out_degree = dict(self.G.out_degree())
        
        # PageRank
        try:
            pagerank = nx.pagerank(self.G, alpha=0.85)
        except:
            pagerank = {}
        
        # Betweenness 中心性（采样以提高速度）
        if self.G.number_of_nodes() > 1000:
            # 对于大图，使用近似算法
            betweenness = {}
            sample_nodes = list(self.G.nodes())[:500]
            betweenness = nx.betweenness_centrality(self.G, k=min(100, len(sample_nodes)))
        else:
            betweenness = nx.betweenness_centrality(self.G)
        
        # 合并结果
        centrality_results = []
        for node in self.G.nodes():
            centrality_results.append({
                'node': node,
                'degree': self.G.degree(node),
                'in_degree': in_degree.get(node, 0),
                'out_degree': out_degree.get(node, 0),
                'degree_centrality': round(degree_centrality.get(node, 0), 6),
                'pagerank': round(pagerank.get(node, 0), 6),
                'betweenness_centrality': round(betweenness.get(node, 0), 6)
            })
        
        # 排序
        centrality_results.sort(key=lambda x: x['degree'], reverse=True)
        
        self.analysis_results['centrality'] = centrality_results[:100]
        return centrality_results[:100]
    
    def disease_symptom_analysis(self):
        """疾病-症状关系分析"""
        disease_symptom_edges = self.edges_df[
            (self.edges_df['x_type'] == 'disease') & 
            (self.edges_df['y_type'].isin(['phenotype', 'symptom', 'effect']))
        ]
        
        # 每个疾病关联的症状数量
        disease_symptom_count = disease_symptom_edges.groupby('x_name').size()
        
        # 每个症状关联的疾病数量
        symptom_disease_count = disease_symptom_edges.groupby('y_name').size()
        
        # 共享症状最多的疾病对
        symptom_map = defaultdict(set)
        for _, row in disease_symptom_edges.iterrows():
            symptom_map[row['y_name']].add(row['x_name'])
        
        shared_symptoms = []
        symptoms_list = list(symptom_map.keys())
        for i, s1 in enumerate(symptoms_list):
            for s2 in symptoms_list[i+1:]:
                shared = symptom_map[s1] & symptom_map[s2]
                if len(shared) >= 2:
                    shared_symptoms.append({
                        'disease_1': list(shared)[0] if len(shared) > 0 else '',
                        'disease_2': list(shared)[1] if len(shared) > 1 else '',
                        'symptom_1': s1,
                        'symptom_2': s2,
                        'shared_count': len(shared)
                    })
        
        shared_symptoms.sort(key=lambda x: x['shared_count'], reverse=True)
        
        self.analysis_results['disease_symptom'] = {
            'avg_symptoms_per_disease': round(disease_symptom_count.mean(), 2) if len(disease_symptom_count) > 0 else 0,
            'avg_diseases_per_symptom': round(symptom_disease_count.mean(), 2) if len(symptom_disease_count) > 0 else 0,
            'top_diseases_by_symptoms': dict(disease_symptom_count.nlargest(20)),
            'top_symptoms_by_diseases': dict(symptom_disease_count.nlargest(20)),
            'shared_symptom_diseases': shared_symptoms[:20]
        }
        
        return self.analysis_results['disease_symptom']
    
    def disease_drug_analysis(self):
        """疾病-药物关系分析"""
        disease_drug_edges = self.edges_df[
            (self.edges_df['x_type'] == 'disease') & 
            (self.edges_df['y_type'] == 'drug')
        ]
        
        # 每个疾病关联的药物数量
        disease_drug_count = disease_drug_edges.groupby('x_name').size()
        
        # 每个药物关联的疾病数量
        drug_disease_count = disease_drug_edges.groupby('y_name').size()
        
        # 常用精神药物
        psych_drugs = [
            'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram',
            'venlafaxine', 'duloxetine', 'desvenlafaxine',
            'imipramine', 'amitriptyline', 'clomipramine',
            'haloperidol', 'risperidone', 'olanzapine', 'quetiapine', 'aripiprazole',
            'lithium', 'valproate', 'carbamazepine', 'lamotrigine',
            'methylphenidate', 'amphetamine', 'atomoxetine',
            'lorazepam', 'alprazolam', 'clonazepam'
        ]
        
        # 统计精神药物使用
        psych_drug_usage = {}
        for drug, count in drug_disease_count.items():
            if any(pd.lower() in drug.lower() for pd in psych_drugs):
                psych_drug_usage[drug] = count
        
        self.analysis_results['disease_drug'] = {
            'avg_drugs_per_disease': round(disease_drug_count.mean(), 2) if len(disease_drug_count) > 0 else 0,
            'avg_diseases_per_drug': round(drug_disease_count.mean(), 2) if len(drug_disease_count) > 0 else 0,
            'top_diseases_by_drugs': dict(disease_drug_count.nlargest(20)),
            'top_drugs_by_diseases': dict(drug_disease_count.nlargest(20)),
            'psychiatric_drug_usage': psych_drug_usage
        }
        
        return self.analysis_results['disease_drug']
    
    def generate_visualizations(self, output_dir='results'):
        """生成可视化图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 节点类型分布饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        if 'node_types' in self.analysis_results:
            node_types = self.analysis_results['node_types']
            labels = list(node_types.keys())
            sizes = list(node_types.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Node Type Distribution in Mental Health Subgraph')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'node_type_distribution.png'), dpi=150)
            plt.close()
        
        # 2. 关系类型分布柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        if 'relation_types' in self.analysis_results:
            rel_types = self.analysis_results['relation_types'].get('display_relation', {})
            if rel_types:
                top_rels = dict(list(rel_types.items())[:15])
                ax.barh(list(top_rels.keys()), list(top_rels.values()), color='steelblue')
                ax.set_xlabel('Count')
                ax.set_title('Top Relation Types')
                ax.invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'relation_type_distribution.png'), dpi=150)
                plt.close()
        
        # 3. 度分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        degrees = [d for n, d in self.G.degree()]
        ax.hist(degrees, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution')
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'degree_distribution.png'), dpi=150)
        plt.close()
        
        # 4. 疾病-症状数量关系
        if 'disease_symptom' in self.analysis_results:
            ds = self.analysis_results['disease_symptom']
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Top 疾病关联症状数
            top_diseases = ds['top_diseases_by_symptoms']
            if top_diseases:
                diseases = list(top_diseases.keys())[:15]
                counts = list(top_diseases.values())[:15]
                axes[0].barh(diseases, counts, color='coral')
                axes[0].set_xlabel('Number of Symptoms')
                axes[0].set_title('Top Diseases by Number of Symptoms')
                axes[0].invert_yaxis()
            
            # Top 症状关联疾病数
            top_symptoms = ds['top_symptoms_by_diseases']
            if top_symptoms:
                symptoms = list(top_symptoms.keys())[:15]
                counts = list(top_symptoms.values())[:15]
                axes[1].barh(symptoms, counts, color='teal')
                axes[1].set_xlabel('Number of Diseases')
                axes[1].set_title('Top Symptoms by Number of Associated Diseases')
                axes[1].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'disease_symptom_analysis.png'), dpi=150)
            plt.close()
        
        # 5. 核心节点中心性比较
        if 'centrality' in self.analysis_results:
            central_nodes = self.analysis_results['centrality'][:20]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            
            # 度中心性
            names = [n['node'][:30] for n in central_nodes]
            dc = [n['degree_centrality'] for n in central_nodes]
            pr = [n['pagerank'] for n in central_nodes]
            bc = [n['betweenness_centrality'] for n in central_nodes]
            
            axes[0].barh(names, dc, color='steelblue')
            axes[0].set_title('Degree Centrality')
            axes[0].invert_yaxis()
            
            axes[1].barh(names, pr, color='coral')
            axes[1].set_title('PageRank')
            axes[1].invert_yaxis()
            
            axes[2].barh(names, bc, color='teal')
            axes[2].set_title('Betweenness Centrality')
            axes[2].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'centrality_analysis.png'), dpi=150)
            plt.close()
        
        print(f"可视化图表已保存到: {output_dir}")
    
    def export_results(self, output_path='results/graph_analysis_results.json'):
        """导出分析结果"""
        def convert(obj):
            """将 numpy/pandas 类型转换为原生 Python 类型"""
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        converted = convert(self.analysis_results)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        print(f"分析结果已导出: {output_path}")
    
    def print_summary(self):
        """打印分析摘要"""
        print("\n" + "=" * 60)
        print("图结构分析摘要")
        print("=" * 60)
        
        if 'basic_stats' in self.analysis_results:
            print("\n【基础统计】")
            for key, value in self.analysis_results['basic_stats'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if 'node_types' in self.analysis_results:
            print("\n【节点类型分布】")
            for ntype, count in sorted(self.analysis_results['node_types'].items(), 
                                        key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {ntype}: {count}")
        
        if 'relation_types' in self.analysis_results:
            print("\n【主要关系类型】")
            rels = self.analysis_results['relation_types'].get('display_relation', {})
            for rel, count in list(rels.items())[:10]:
                print(f"  {rel}: {count}")
        
        if 'disease_symptom' in self.analysis_results:
            ds = self.analysis_results['disease_symptom']
            print("\n【疾病-症状分析】")
            print(f"  平均每疾病关联症状数: {ds['avg_symptoms_per_disease']}")
            print(f"  平均每症状关联疾病数: {ds['avg_diseases_per_symptom']}")
            if ds['top_diseases_by_symptoms']:
                top = list(ds['top_diseases_by_symptoms'].items())[0]
                print(f"  症状最多的疾病: {top[0]} ({top[1]} 个症状)")
        
        if 'centrality' in self.analysis_results:
            print("\n【核心疾病节点 Top 10】")
            for i, node in enumerate(self.analysis_results['centrality'][:10]):
                print(f"  {i+1}. {node['node'][:50]} (度={node['degree']}, PR={node['pagerank']:.4f})")
        
        print("\n" + "=" * 60)


def run_full_analysis(edges_df, G, output_dir='results'):
    """运行完整分析流程"""
    stats = GraphStatistics(G, edges_df)
    
    # 执行各项分析
    stats.basic_statistics()
    stats.node_type_statistics()
    stats.relation_type_statistics()
    stats.centrality_analysis()
    stats.disease_symptom_analysis()
    stats.disease_drug_analysis()
    
    # 生成可视化
    stats.generate_visualizations(output_dir)
    
    # 导出结果
    os.makedirs(output_dir, exist_ok=True)
    stats.export_results(os.path.join(output_dir, 'graph_analysis_results.json'))
    
    # 打印摘要
    stats.print_summary()
    
    return stats


if __name__ == '__main__':
    # 测试
    print("请先运行 subgraph_extraction.py 下载数据并提取子图")
    print("然后导入本模块进行详细分析")
