"""
PrimeKG 数据下载与精神障碍子图提取模块
从 Harvard Dataverse 下载 PrimeKG 数据，并提取精神障碍子图
"""

import os
import requests
import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import hashlib
import json

# ============== 配置 ==============

# PrimeKG 数据下载链接（Harvard Dataverse）
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"

# 精神障碍关键词列表（基于心理学 DSM-5/ICD-11 分类）
MENTAL_DISORDER_KEYWORDS = [
    # 情感障碍
    'depression', 'depressive', 'major depressive', 'dysthymia', 'bipolar',
    # 焦虑障碍
    'anxiety', 'anxious', 'panic', 'phobia', 'social anxiety', 'agoraphobia',
    'generalized anxiety',
    # 创伤相关
    'ptsd', 'post-traumatic', 'trauma', 'stress disorder', 'adjustment disorder',
    # 精神分裂症谱系
    'schizophrenia', 'schizoaffective', 'psychosis', 'delusional', 'hallucination',
    # 神经发育障碍
    'autism', 'autistic', 'adhd', 'attention deficit', 'hyperactivity',
    # 强迫及相关障碍
    'ocd', 'obsessive-compulsive', 'hoarding', 'trichotillomania', 'excoriation',
    # 饮食障碍
    'eating disorder', 'anorexia', 'bulimia', 'binge eating', 'pica',
    # 人格障碍
    'personality disorder', 'borderline', 'narcissistic', 'antisocial', 'avoidant',
    'dependent', 'paranoid personality',
    # 分离障碍
    'dissociative', 'dissociative identity', 'amnesia', 'fugue',
    # 躯体症状障碍
    'somatoform', 'somatoform disorder', 'conversion disorder', 'body dysmorphic',
    # 睡眠障碍
    'insomnia', 'sleep disorder', 'narcolepsy', 'sleep apnea', 'hypersomnia',
    # 痴呆与认知障碍
    'dementia', 'alzheimer', 'cognitive impairment', 'memory disorder',
    'neurocognitive disorder', 'parkinson',
    # 成瘾障碍
    'substance abuse', 'addiction', 'alcoholism', 'drug dependence', 'addictive',
    # 其他精神障碍
    'mental disorder', 'psychiatric', 'mental illness', 'mental health',
    'mood disorder', 'affective disorder', 'neurosis', 'psychopathology'
]

# 症状关键词
SYMPTOM_KEYWORDS = [
    'symptom', 'manifestation', 'sign', 'clinical manifestation',
    'feeling', 'emotion', 'mood', 'behavior', 'cognitive',
    'sadness', 'hopelessness', 'worthlessness', 'guilt', 'anhedonia',
    'fatigue', 'sleep', 'appetite', 'concentration', 'suicidal',
    'worry', 'fear', 'panic', 'avoidance', 'hyperarousal',
    'hallucination', 'delusion', 'disorganized', 'negative symptom',
    'obsession', 'compulsion', 'ritual', 'anxiety',
    'weight loss', 'weight gain', 'binge', 'purge',
    'irritability', 'mood swing', 'euphoria', 'grandiosity',
    'tremor', 'rigidity', 'bradykinesia', 'postural',
    'tremor', 'seizure', 'headache', 'pain',
    'insomnia', 'hypersomnia', 'nightmare', 'sleep disturbance'
]

# 药物关键词
DRUG_KEYWORDS = [
    'drug', 'medication', 'pharmaceutical', 'medicine', 'treatment',
    'ssri', 'snri', 'maoi', 'tricyclic', 'antidepressant', 'anxiolytic',
    'antipsychotic', 'neuroleptic', 'stimulant', 'benzodiazepine',
    'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram',
    'venlafaxine', 'duloxetine', 'desvenlafaxine',
    'imipramine', 'amitriptyline', 'nortriptyline', 'clomipramine',
    'haloperidol', 'risperidone', 'olanzapine', 'quetiapine', 'aripiprazole',
    'clozapine', 'ziprasidone', 'paliperidone', 'asenapine',
    'lithium', 'valproate', 'carbamazepine', 'lamotrigine', 'oxcarbazepine',
    'methylphenidate', 'amphetamine', 'atomoxetine', 'bupropion',
    'lorazepam', 'diazepam', 'alprazolam', 'clonazepam', 'temazepam',
    'zolpidem', 'eszopiclone', 'trazodone', 'mirtazapine',
    'clonidine', 'guanfacine', 'propranolol',
    'naltrexone', 'acamprosate', 'disulfiram', 'buprenorphine'
]

# 关系类型映射
RELATION_TYPES = {
    'disease_to_phenotype': ['phenotype', 'associated_phenotype', 'disease_phenotype'],
    'disease_to_drug': ['drug', 'treats', 'interacts', 'contraindication'],
    'disease_to_gene': ['gene', 'associated_gene', 'disease_gene'],
    'disease_to_pathway': ['pathway', 'biological_process'],
    'disease_to_anatomy': ['anatomy', 'affected_structure'],
    'disease_to_disease': ['disease', 'comorbidity', 'similar_to', 'parent_of'],
    'disease_to_function': ['function', 'molecular_function']
}

# 节点类型映射
NODE_TYPE_COLUMNS = ['x_type', 'y_type']


def download_primekg(output_path='data/primekg_raw.csv'):
    """下载 PrimeKG 数据"""
    if os.path.exists(output_path):
        print(f"PrimeKG 数据已存在: {output_path}")
        return output_path
    
    print("正在从 Harvard Dataverse 下载 PrimeKG 数据...")
    print("这可能需要几分钟时间，请耐心等待...")
    
    try:
        response = requests.get(PRIMEKG_URL, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r下载进度: {progress:.1f}%", end='', flush=True)
        
        print(f"\n下载完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"\n下载失败: {e}")
        print("请手动下载：https://dataverse.harvard.edu/api/access/datafile/6180620")
        return None


def load_primekg(csv_path='data/primekg_raw.csv'):
    """加载 PrimeKG 数据"""
    print(f"正在加载 PrimeKG 数据: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"加载完成: {len(df)} 条边")
    return df


def extract_mental_health_subgraph(df, output_path='data/mental_subgraph.csv'):
    """提取精神障碍子图"""
    print("正在提取精神障碍子图...")
    
    # 查找精神障碍相关节点
    # PrimeKG 的 x_name 和 y_name 列包含节点名称
    disorder_pattern = '|'.join(MENTAL_DISORDER_KEYWORDS)
    
    # 找出与精神障碍相关的边
    disorder_mask = (
        df['x_name'].str.lower().str.contains(disorder_pattern, na=False, regex=True) |
        df['y_name'].str.lower().str.contains(disorder_pattern, na=False, regex=True)
    )
    
    mental_edges = df[disorder_mask].copy()
    
    # 去重
    mental_edges = mental_edges.drop_duplicates()
    
    print(f"找到 {len(mental_edges)} 条与精神障碍相关的边")
    
    # 获取涉及的节点
    disorder_nodes = set(mental_edges['x_name'].unique()) | set(mental_edges['y_name'].unique())
    print(f"涉及 {len(disorder_nodes)} 个节点")
    
    # 保存子图
    mental_edges.to_csv(output_path, index=False)
    print(f"子图已保存: {output_path}")
    
    return mental_edges, disorder_nodes


def build_networkx_graph(edges_df):
    """构建 NetworkX 图"""
    print("正在构建 NetworkX 图...")
    
    G = nx.DiGraph()  # 使用有向图
    
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['x_name'], row['y_name'],
            relation=row.get('relation', 'related_to'),
            display_relation=row.get('display_relation', 'related'),
            source=row.get('source', 'unknown')
        )
    
    print(f"图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    return G


def analyze_node_types(edges_df):
    """分析节点类型分布"""
    print("\n=== 节点类型分析 ===")
    
    # PrimeKG 的 x_type 和 y_type 列
    x_types = edges_df['x_type'].value_counts()
    y_types = edges_df['y_type'].value_counts()
    
    print("X侧节点类型分布:")
    for ntype, count in x_types.head(10).items():
        print(f"  {ntype}: {count}")
    
    print("\nY侧节点类型分布:")
    for ntype, count in y_types.head(10).items():
        print(f"  {ntype}: {count}")
    
    return {'x_types': x_types.to_dict(), 'y_types': y_types.to_dict()}


def analyze_relation_types(edges_df):
    """分析关系类型分布"""
    print("\n=== 关系类型分析 ===")
    
    # 按 display_relation 统计
    rel_dist = edges_df['display_relation'].value_counts()
    
    print("关系类型分布:")
    for rel, count in rel_dist.head(20).items():
        print(f"  {rel}: {count}")
    
    return rel_dist.to_dict()


def find_core_disease_nodes(G, top_n=20):
    """找出核心疾病节点（按度中心性）"""
    print("\n=== 核心疾病节点分析 ===")
    
    # 计算度中心性
    degree_centrality = nx.degree_centrality(G)
    
    # 计算 PageRank
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except:
        pagerank = {}
    
    # 合并排序
    core_nodes = []
    for node in G.nodes():
        core_nodes.append({
            'node': node,
            'degree': G.degree(node),
            'degree_centrality': degree_centrality.get(node, 0),
            'pagerank': pagerank.get(node, 0)
        })
    
    # 按度排序
    core_nodes.sort(key=lambda x: x['degree'], reverse=True)
    
    print(f"Top {top_n} 核心节点:")
    for i, node_info in enumerate(core_nodes[:top_n]):
        print(f"  {i+1}. {node_info['node']} (度={node_info['degree']}, PR={node_info['pagerank']:.4f})")
    
    return core_nodes[:top_n]


def analyze_comorbidity(G, edges_df, top_n=30):
    """分析共病模式"""
    print("\n=== 共病模式分析 ===")
    
    # 找出所有疾病-疾病边
    disease_disease_edges = edges_df[
        (edges_df['x_type'] == 'disease') & (edges_df['y_type'] == 'disease')
    ]
    
    # 统计共享症状/表型的疾病对
    # 疾病A和疾病B共病的证据：它们共享相同的症状或表型
    symptom_map = defaultdict(set)  # 症状 -> 相关的疾病
    
    for _, row in edges_df.iterrows():
        if row['y_type'] in ['phenotype', 'symptom', 'effect']:
            symptom_map[row['y_name']].add(row['x_name'])
    
    # 计算共病对
    comorbid_pairs = []
    symptoms_list = list(symptom_map.keys())
    
    for i, sym1 in enumerate(symptoms_list):
        for sym2 in symptoms_list[i+1:]:
            diseases1 = symptom_map[sym1]
            diseases2 = symptom_map[sym2]
            shared = diseases1 & diseases2
            if len(shared) >= 2:
                for d1 in shared:
                    for d2 in shared:
                        if d1 != d2:
                            comorbid_pairs.append((d1, d2, sym1, sym2))
    
    # 统计共病频率
    from collections import Counter
    pair_counter = Counter()
    for d1, d2, s1, s2 in comorbid_pairs:
        pair_counter[(d1, d2)] += 1
        pair_counter[(d2, d1)] += 1
    
    print(f"发现 {len(pair_counter)} 对共病疾病")
    print("\nTop 共病对:")
    for (d1, d2), count in pair_counter.most_common(top_n):
        print(f"  {d1} <-> {d2}: {count} 次共病")
    
    return pair_counter.most_common(top_n)


def categorize_diseases(edges_df, disorder_keywords):
    """对精神障碍进行大类分类"""
    print("\n=== 疾病大类分类 ===")
    
    # 找出所有疾病节点
    diseases = set(edges_df[edges_df['x_type'] == 'disease']['x_name'].unique())
    diseases |= set(edges_df[edges_df['y_type'] == 'disease']['y_name'].unique())
    
    # 分类规则
    categories = {
        'depressive_disorders': [],
        'anxiety_disorders': [],
        'trauma_stress_disorders': [],
        'psychotic_disorders': [],
        'neurodevelopmental_disorders': [],
        'obsessive_compulsive_disorders': [],
        'eating_disorders': [],
        'personality_disorders': [],
        'neurocognitive_disorders': [],
        'other_mental_disorders': []
    }
    
    category_keywords = {
        'depressive_disorders': ['depression', 'depressive', 'dysthymia', 'melancholia'],
        'anxiety_disorders': ['anxiety', 'panic', 'phobia', 'agoraphobia', 'worry'],
        'trauma_stress_disorders': ['ptsd', 'trauma', 'stress', 'adjustment'],
        'psychotic_disorders': ['schizophrenia', 'schizoaffective', 'psychosis', 'delusion'],
        'neurodevelopmental_disorders': ['autism', 'adhd', 'attention deficit', 'hyperactivity', 'tourette'],
        'obsessive_compulsive_disorders': ['ocd', 'obsessive', 'compulsive', 'hoarding'],
        'eating_disorders': ['eating', 'anorexia', 'bulimia', 'binge', 'appetite disorder'],
        'personality_disorders': ['personality', 'borderline', 'narcissistic', 'antisocial'],
        'neurocognitive_disorders': ['dementia', 'alzheimer', 'cognitive', 'memory', 'parkinson']
    }
    
    for disease in diseases:
        disease_lower = disease.lower()
        categorized = False
        for cat, keywords in category_keywords.items():
            if any(kw in disease_lower for kw in keywords):
                categories[cat].append(disease)
                categorized = True
                break
        if not categorized:
            categories['other_mental_disorders'].append(disease)
    
    # 打印统计
    for cat, diseases_list in categories.items():
        print(f"  {cat}: {len(diseases_list)} 种疾病")
    
    return categories


def generate_subgraph_summary(G, edges_df, output_path='data/subgraph_summary.json'):
    """生成子图摘要统计"""
    print("\n=== 生成子图摘要 ===")
    
    summary = {
        'basic_stats': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'num_connected_components': nx.number_weakly_connected_components(G)
        },
        'node_type_distribution': {},
        'relation_type_distribution': {},
        'top_diseases': [],
        'disease_categories': {}
    }
    
    # 节点类型分布
    for node in G.nodes():
        # 尝试推断节点类型
        node_lower = node.lower()
        if any(kw in node_lower for kw in MENTAL_DISORDER_KEYWORDS):
            ntype = 'mental_disorder'
        elif any(kw in node_lower for kw in SYMPTOM_KEYWORDS):
            ntype = 'symptom'
        elif any(kw in node_lower for kw in DRUG_KEYWORDS):
            ntype = 'drug'
        else:
            ntype = 'other'
        
        if ntype not in summary['node_type_distribution']:
            summary['node_type_distribution'][ntype] = 0
        summary['node_type_distribution'][ntype] += 1
    
    # 关系类型分布
    rel_dist = edges_df['display_relation'].value_counts()
    summary['relation_type_distribution'] = rel_dist.to_dict()
    
    # Top 疾病节点
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
    summary['top_diseases'] = [
        {'name': name, 'degree_centrality': round(score, 4)}
        for name, score in sorted_nodes[:30]
    ]
    
    # 保存摘要
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"摘要已保存: {output_path}")
    return summary


def main():
    """主函数"""
    print("=" * 60)
    print("PrimeKG 精神障碍子图提取与分析")
    print("=" * 60)
    
    # 数据目录
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. 下载/加载 PrimeKG 数据
    csv_path = os.path.join(data_dir, 'primekg_raw.csv')
    if not os.path.exists(csv_path):
        csv_path = download_primekg(csv_path)
        if csv_path is None:
            print("无法下载数据，程序退出")
            return
    else:
        print(f"数据文件已存在: {csv_path}")
    
    # 2. 加载数据
    df = load_primekg(csv_path)
    
    # 3. 提取精神障碍子图
    mental_edges, mental_nodes = extract_mental_health_subgraph(
        df, 
        output_path=os.path.join(data_dir, 'mental_subgraph.csv')
    )
    
    # 4. 构建图
    G = build_networkx_graph(mental_edges)
    
    # 5. 图分析
    print("\n" + "=" * 60)
    print("图结构分析")
    print("=" * 60)
    
    # 节点类型分析
    analyze_node_types(mental_edges)
    
    # 关系类型分析
    analyze_relation_types(mental_edges)
    
    # 核心节点分析
    find_core_disease_nodes(G)
    
    # 共病分析
    analyze_comorbidity(G, mental_edges)
    
    # 疾病分类
    categorize_diseases(mental_edges, MENTAL_DISORDER_KEYWORDS)
    
    # 6. 生成摘要
    summary = generate_subgraph_summary(G, mental_edges)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print(f"子图包含 {summary['basic_stats']['num_nodes']} 个节点")
    print(f"子图包含 {summary['basic_stats']['num_edges']} 条边")
    print("=" * 60)


if __name__ == '__main__':
    main()
