"""
Flask Web 应用主程序
精神障碍知识图谱助手 - 四个页面
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'mental_health_kg_secret_key_2024'
app.config['JSON_AS_ASCII'] = False

# 全局数据
DATA_DIR = 'data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

# 尝试加载数据
subgraph_df = None
graph_data = None
model_results = None
hallucination_results = None

def normalize_search_name(name):
    if not isinstance(name, str):
        return ''

    return name.strip().lower()


def format_entity(name, relation, entity_type=None):
    return {
        'name': name,
        'search_name': name,
        'relation': relation,
    }

def load_data():
    """加载数据"""
    global subgraph_df, graph_data, model_results, hallucination_results
    
    # 加载子图数据
    subgraph_path = os.path.join(DATA_DIR, 'mental_subgraph.csv')
    if os.path.exists(subgraph_path):
        subgraph_df = pd.read_csv(subgraph_path)
        print(f"加载了 {len(subgraph_df)} 条边数据")
    
    # 加载图分析结果
    analysis_path = os.path.join(RESULTS_DIR, 'graph_analysis_results.json')
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        print("加载了图分析结果")
    
    # 加载模型结果
    model_results_path = os.path.join(RESULTS_DIR, 'model_comparison_report.md')
    if os.path.exists(model_results_path):
        with open(model_results_path, 'r', encoding='utf-8') as f:
            model_results = f.read()
        print("加载了模型对比结果")
    
    # 加载幻觉检测结果
    hallucination_path = os.path.join(RESULTS_DIR, 'hallucination_evaluation.json')
    if os.path.exists(hallucination_path):
        with open(hallucination_path, 'r', encoding='utf-8') as f:
            hallucination_results = json.load(f)
        print("加载了幻觉检测结果")

# 在启动时加载数据
load_data()


# ==================== 路由定义 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')


@app.route('/graph')
def graph_page():
    """知识图谱浏览页"""
    return render_template('graph.html')


@app.route('/qa')
def qa_page():
    """疾病知识问答页"""
    return render_template('qa.html')


@app.route('/hallucination')
def hallucination_page():
    """LLM幻觉检测页"""
    return render_template('hallucination.html')


@app.route('/experiments')
def experiments_page():
    """模型实验结果页"""
    return render_template('experiments.html')


# ==================== API 路由 ====================

@app.route('/api/search_disease', methods=['POST'])
def search_disease():
    """搜索疾病信息"""
    data = request.get_json()
    raw_disease_name = data.get('disease', '').strip()
    disease_name = normalize_search_name(raw_disease_name)
    
    if not disease_name:
        return jsonify({'error': '请输入疾病名称'}), 400
    
    if subgraph_df is None:
        return jsonify({'error': '数据未加载'}), 500
    
    # 搜索相关疾病
    disease_edges = subgraph_df[
        subgraph_df['x_name'].str.lower().str.contains(disease_name, na=False, regex=False) |
        subgraph_df['y_name'].str.lower().str.contains(disease_name, na=False, regex=False)
    ]
    
    # 获取相关信息
    symptoms = []
    drugs = []
    related_diseases = []
    phenotypes = []
    
    for _, row in disease_edges.iterrows():
        if disease_name in row['x_name'].lower():
            target = row['y_name']
            target_type = row['y_type']
            relation = row.get('display_relation', row.get('relation', 'related'))
            
            if target_type in ['phenotype', 'symptom', 'effect']:
                symptoms.append(format_entity(target, relation, target_type))
            elif target_type == 'drug':
                drugs.append(format_entity(target, relation, target_type))
            elif target_type == 'disease':
                related_diseases.append(format_entity(target, relation, target_type))
            else:
                item = format_entity(target, relation, target_type)
                item['type'] = target_type
                phenotypes.append(item)
        elif disease_name in row['y_name'].lower():
            source = row['x_name']
            source_type = row['x_type']
            relation = row.get('display_relation', row.get('relation', 'related'))
            
            if source_type in ['phenotype', 'symptom', 'effect']:
                symptoms.append(format_entity(source, relation, source_type))
            elif source_type == 'drug':
                drugs.append(format_entity(source, relation, source_type))
            elif source_type == 'disease':
                related_diseases.append(format_entity(source, relation, source_type))
    
    # 去重
    symptoms = list({s['name']: s for s in symptoms}.values())[:20]
    drugs = list({d['name']: d for d in drugs}.values())[:20]
    related_diseases = list({d['name']: d for d in related_diseases}.values())[:20]
    phenotypes = list({p['name']: p for p in phenotypes}.values())[:20]
    
    # 构建网络图数据
    network_data = build_network_data(disease_name, symptoms, drugs, related_diseases)
    
    return jsonify({
        'disease': disease_name.title(),
        'symptoms': symptoms,
        'drugs': drugs,
        'related_diseases': related_diseases,
        'phenotypes': phenotypes,
        'network': network_data,
        'total_symptoms': len(symptoms),
        'total_drugs': len(drugs),
        'total_related': len(related_diseases)
    })


def build_network_data(disease_name, symptoms, drugs, related_diseases):
    """构建网络图数据"""
    nodes = []
    edges = []
    node_ids = {}
    
    # 添加中心疾病节点
    disease_id = 'disease_0'
    nodes.append({
        'id': disease_id,
        'name': disease_name.title(),
        'type': 'disease',
        'level': 0
    })
    node_ids[disease_name.lower()] = disease_id
    
    # 添加症状节点
    for i, symptom in enumerate(symptoms[:15]):
        node_id = f'symptom_{i}'
        nodes.append({
            'id': node_id,
            'name': symptom['name'],
            'type': 'symptom',
            'level': 1
        })
        edges.append({
            'source': disease_id,
            'target': node_id,
            'relation': symptom['relation']
        })
    
    # 添加药物节点
    for i, drug in enumerate(drugs[:15]):
        node_id = f'drug_{i}'
        nodes.append({
            'id': node_id,
            'name': drug['name'],
            'type': 'drug',
            'level': 1
        })
        edges.append({
            'source': disease_id,
            'target': node_id,
            'relation': drug['relation']
        })
    
    # 添加相关疾病节点
    for i, related in enumerate(related_diseases[:10]):
        node_id = f'disease_{i+1}'
        nodes.append({
            'id': node_id,
            'name': related['name'],
            'type': 'disease',
            'level': 1
        })
        edges.append({
            'source': disease_id,
            'target': node_id,
            'relation': related['relation']
        })
    
    return {'nodes': nodes, 'edges': edges}


@app.route('/api/answer_question', methods=['POST'])
def answer_question():
    """回答问题"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': '请输入问题'}), 400
    
    # 模拟 RAG 回答（实际应用中调用 LLM + RAG）
    answer = generate_mock_answer(question)
    
    # 模拟三元组证据
    triplets = generate_mock_triplets(question)
    
    return jsonify({
        'question': question,
        'answer': answer,
        'triplets': triplets,
        'sources': [
            {'text': f'关于{question}的详细信息来自 PrimeKG 知识图谱。',
             'score': 0.95}
        ]
    })


def generate_mock_answer(question):
    """生成模拟回答"""
    question_lower = question.lower()
    
    if '抑郁' in question or 'depression' in question_lower:
        return (
            "抑郁症（Depression）是一种常见的情感性精神障碍，主要表现为持续的情绪低落、兴趣减退、 "
            "快感缺失、精力下降、注意力不集中、自我评价降低、睡眠和食欲改变等症状。\n\n"
            "根据 PrimeKG 知识图谱，抑郁症与多种症状相关，包括情绪低落、失眠、焦虑等。 "
            "常用的治疗药物包括 SSRIs（如氟西汀、舍曲林）和 SNRIs（如文拉法辛）。"
        )
    elif '焦虑' in question or 'anxiety' in question_lower:
        return (
            "焦虑障碍（Anxiety Disorder）是一类以过度的担忧、恐惧和回避行为为特征的精神障碍。\n\n"
            "主要类型包括：\n"
            "- 广泛性焦虑障碍（GAD）\n"
            "- 惊恐障碍\n"
            "- 社交焦虑障碍\n"
            "- 特定恐惧症\n\n"
            "常用治疗药物包括苯二氮卓类、SSRIs 和 SNRIs。"
        )
    elif '精神分裂' in question or 'schizophrenia' in question_lower:
        return (
            "精神分裂症（Schizophrenia）是一种严重的精神病性障碍，特征包括：\n\n"
            "阳性症状：幻觉、妄想、思维紊乱\n"
            "阴性症状：情感淡漠、社交退缩、意志减退\n"
            "认知症状：注意力、记忆、执行功能损害\n\n"
            "常用治疗药物为抗精神病药，如奥氮平、利培酮、氟哌啶醇等。"
        )
    elif '症状' in question or 'symptom' in question_lower:
        return (
            "精神障碍的常见症状包括：\n\n"
            "- 情绪症状：情绪低落、焦虑、情绪波动\n"
            "- 认知症状：注意力不集中、记忆力下降、思维紊乱\n"
            "- 行为症状：社交退缩、活动减少、刻板行为\n"
            "- 躯体症状：睡眠障碍、食欲改变、疼痛乏力"
        )
    elif '药物' in question or 'drug' in question_lower or '治疗' in question_lower:
        return (
            "精神障碍常用治疗药物包括：\n\n"
            "1. 抗抑郁药：SSRIs（氟西汀、舍曲林）、SNRIs（文拉法辛）\n"
            "2. 抗精神病药：奥氮平、利培酮、阿立哌唑\n"
            "3. 情感稳定剂：锂盐、丙戊酸盐、卡马西平\n"
            "4. 抗焦虑药：苯二氮卓类（短期使用）\n"
            "5. 兴奋剂：哌甲酯、安非他明（用于 ADHD）"
        )
    else:
        return (
            f"关于「{question}」的问题，我基于 PrimeKG 知识图谱为您解答：\n\n"
            "精神障碍是影响情绪、思维和行为的心理健康状况。\n"
            "主要类型包括情感障碍、焦虑障碍、精神病性障碍等。\n\n"
            "建议您提供更具体的问题，以便获得更准确的回答。"
        )


def generate_mock_triplets(question):
    """生成模拟三元组证据"""
    question_lower = question.lower()
    
    if '抑郁' in question or 'depression' in question_lower:
        return [
            {'head': 'Depression', 'relation': 'phenotype', 'tail': 'Depressed mood'},
            {'head': 'Depression', 'relation': 'treats', 'tail': 'Fluoxetine'},
            {'head': 'Depression', 'relation': 'associated_phenotype', 'tail': 'Insomnia'},
            {'head': 'Depression', 'relation': 'associated_phenotype', 'tail': 'Anxiety'},
            {'head': 'Depression', 'relation': 'similar_to', 'tail': 'Bipolar Disorder'}
        ]
    elif '焦虑' in question or 'anxiety' in question_lower:
        return [
            {'head': 'Anxiety Disorder', 'relation': 'phenotype', 'tail': 'Excessive worry'},
            {'head': 'Anxiety Disorder', 'relation': 'treats', 'tail': 'Sertraline'},
            {'head': 'Anxiety Disorder', 'relation': 'comorbidity', 'tail': 'Depression'},
            {'head': 'Panic Disorder', 'relation': 'phenotype', 'tail': 'Panic attacks'}
        ]
    else:
        return [
            {'head': 'Mental Disorder', 'relation': 'associated_phenotype', 'tail': 'Symptom'},
            {'head': 'Psychiatric Drug', 'relation': 'treats', 'tail': 'Mental Disorder'}
        ]


@app.route('/api/hallucination_test', methods=['POST'])
def hallucination_test():
    """幻觉检测测试"""
    data = request.get_json()
    relation_type = data.get('relation_type', 'all')
    
    # 模拟幻觉检测结果
    results = generate_hallucination_results(relation_type)
    
    return jsonify(results)


def generate_hallucination_results(relation_type):
    """生成幻觉检测结果"""
    # 模拟测试数据
    real_triplets = [
        {'head': 'Depression', 'relation': 'phenotype', 'tail': 'Sad mood', 'is_real': True},
        {'head': 'Anxiety Disorder', 'relation': 'treats', 'tail': 'Sertraline', 'is_real': True},
        {'head': 'Schizophrenia', 'relation': 'phenotype', 'tail': 'Hallucination', 'is_real': True},
        {'head': 'Bipolar Disorder', 'relation': 'similar_to', 'tail': 'Depression', 'is_real': True},
        {'head': 'PTSD', 'relation': 'phenotype', 'tail': 'Flashback', 'is_real': True},
    ]
    
    false_triplets = [
        {'head': 'Depression', 'relation': 'treats', 'tail': 'Aspirin', 'is_real': False},
        {'head': 'Schizophrenia', 'relation': 'treats', 'tail': 'Antibiotic', 'is_real': False},
        {'head': 'Anxiety', 'relation': 'phenotype', 'tail': 'High fever', 'is_real': False},
        {'head': 'Bipolar Disorder', 'relation': 'comorbidity', 'tail': 'Common cold', 'is_real': False},
    ]
    
    # 模拟 LLM 判断
    np.random.seed(datetime.now().microsecond)
    all_results = []
    
    for triplet in real_triplets + false_triplets:
        # 模拟 LLM 有 75% 准确率
        is_correct = np.random.random() < 0.75
        
        if is_correct:
            llm_answer = 'true' if triplet['is_real'] else 'false'
        else:
            llm_answer = 'false' if triplet['is_real'] else 'true'
        
        result = {
            **triplet,
            'question': f"({triplet['head']}) --[{triplet['relation']}]--> ({triplet['tail']})，这个说法正确吗？",
            'llm_answer': llm_answer,
            'llm_reason': "根据知识图谱数据，" + ("这个关系是正确的。" if llm_answer == 'true' else "这个关系是错误的或不存在。"),
            'is_correct': is_correct
        }
        all_results.append(result)
    
    # 计算统计
    total = len(all_results)
    correct = sum(1 for r in all_results if r['is_correct'])
    
    real_results = [r for r in all_results if r['is_real']]
    false_results = [r for r in all_results if not r['is_real']]
    
    real_correct = sum(1 for r in real_results if r['is_correct'])
    false_correct = sum(1 for r in false_results if r['is_correct'])
    
    return {
        'results': all_results,
        'statistics': {
            'total': total,
            'correct': correct,
            'accuracy': round(correct / total, 2) if total > 0 else 0,
            'real_triplets': {
                'total': len(real_results),
                'correct': real_correct,
                'miss_rate': round((len(real_results) - real_correct) / len(real_results), 2) if real_results else 0
            },
            'false_triplets': {
                'total': len(false_results),
                'correct': false_correct,
                'detection_rate': round(false_correct / len(false_results), 2) if false_results else 0
            }
        }
    }


@app.route('/api/model_results', methods=['GET'])
def get_model_results():
    """获取模型实验结果"""
    # 模拟实验结果数据
    results = {
        'transE': {
            'MRR': 0.35,
            'Hits@1': 0.22,
            'Hits@3': 0.38,
            'Hits@5': 0.45,
            'Hits@10': 0.52
        },
        'RotatE': {
            'MRR': 0.38,
            'Hits@1': 0.25,
            'Hits@3': 0.42,
            'Hits@5': 0.48,
            'Hits@10': 0.55
        },
        'disease_classification': {
            'Random Forest': 0.82,
            'Gradient Boosting': 0.85,
            'SVM': 0.79
        }
    }
    
    return jsonify(results)


@app.route('/api/graph_stats', methods=['GET'])
def get_graph_stats():
    """获取图统计信息"""
    if graph_data:
        return jsonify(graph_data)
    
    # 返回默认统计
    return jsonify({
        'basic_stats': {
            'num_nodes': 0,
            'num_edges': 0
        },
        'node_types': {},
        'relation_types': {}
    })


@app.route('/api/all_diseases', methods=['GET'])
def get_all_diseases():
    """获取所有疾病列表"""
    if subgraph_df is not None:
        diseases = subgraph_df[
            (subgraph_df['x_type'] == 'disease') | 
            (subgraph_df['y_type'] == 'disease')
        ]
        
        disease_set = set(diseases['x_name'].unique()) | set(diseases['y_name'].unique())
        disease_list = sorted([d for d in disease_set if isinstance(d, str)])[:100]
        
        return jsonify({'diseases': disease_list})
    
    # 返回默认列表
    default_diseases = [
        'Depression', 'Anxiety Disorder', 'Schizophrenia', 'Bipolar Disorder',
        'PTSD', 'OCD', 'Autism Spectrum Disorder', 'ADHD', 'Panic Disorder',
        'Social Anxiety Disorder', 'Borderline Personality Disorder',
        'Anorexia Nervosa', 'Bulimia Nervosa', "Alzheimer's Disease",
        "Parkinson's Disease", 'Insomnia Disorder', 'Post-Traumatic Stress Disorder'
    ]
    
    return jsonify({'diseases': default_diseases})


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', error='页面不存在'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', error='服务器错误'), 500


# ==================== 启动应用 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("精神障碍知识图谱助手 Web 应用")
    print("=" * 60)
    print("访问地址: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
