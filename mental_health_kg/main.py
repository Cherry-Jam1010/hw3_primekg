"""
精神障碍知识图谱分析与可信问答系统
主入口脚本 - 整合所有模块
"""

import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入各模块
from analysis.subgraph_extraction import download_primekg, load_primekg, extract_mental_health_subgraph
from analysis.graph_statistics import run_full_analysis, GraphStatistics
from models.train_transE import run_experiment as run_transe
from models.train_rotateE import run_experiment as run_rotate
from models.evaluation import ModelComparison
from models.disease_classification import run_classification_experiment, plot_classification_results
from hallucination.generate_triplets import TripletGenerator, HallucinationDataset, generate_synthetic_results
from hallucination.eval_hallucination import HallucinationEvaluator
from rag.vector_store import build_vector_store
from rag.rag_chain import MentalHealthQASystem


def print_banner():
    """打印横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     精神障碍知识图谱分析与可信问答系统                    ║
    ║     Mental Health Knowledge Graph Analysis System         ║
    ║                                                          ║
    ║     基于 PrimeKG 构建的精神障碍知识图谱系统               ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def step_1_download_data():
    """步骤1：下载数据"""
    print("\n" + "=" * 60)
    print("步骤 1: 下载 PrimeKG 数据")
    print("=" * 60)
    
    csv_path = download_primekg('data/primekg_raw.csv')
    
    if csv_path:
        print("数据下载完成！")
    else:
        print("数据下载失败，请手动下载。")
    
    return csv_path


def step_2_extract_subgraph():
    """步骤2：提取子图"""
    print("\n" + "=" * 60)
    print("步骤 2: 提取精神障碍子图")
    print("=" * 60)
    
    try:
        df = load_primekg('data/primekg_raw.csv')
        mental_edges, mental_nodes = extract_mental_health_subgraph(
            df,
            output_path='data/mental_subgraph.csv'
        )
        return mental_edges
    except Exception as e:
        print(f"子图提取失败: {e}")
        return None


def step_3_analyze_graph(edges_df):
    """步骤3：图分析"""
    print("\n" + "=" * 60)
    print("步骤 3: 图结构分析")
    print("=" * 60)
    
    if edges_df is None:
        print("没有可用的边数据")
        return
    
    import networkx as nx
    G = nx.DiGraph()
    
    for _, row in edges_df.iterrows():
        G.add_edge(row['x_name'], row['y_name'])
    
    stats = run_full_analysis(G, edges_df, output_dir='results')
    print("图分析完成！")


def step_4_train_models():
    """步骤4：训练模型"""
    print("\n" + "=" * 60)
    print("步骤 4: 训练知识图谱嵌入模型")
    print("=" * 60)
    
    # 训练 TransE
    print("\n>>> 训练 TransE 模型...")
    transe_result, transe_metrics = run_transe(
        output_dir='models',
        csv_path='data/mental_subgraph.csv'
    )
    
    # 训练 RotatE
    print("\n>>> 训练 RotatE 模型...")
    rotate_result, rotate_metrics = run_rotate(
        output_dir='models',
        csv_path='data/mental_subgraph.csv'
    )
    
    print("\n模型训练完成！")


def step_5_compare_models():
    """步骤5：对比模型"""
    print("\n" + "=" * 60)
    print("步骤 5: 模型对比评估")
    print("=" * 60)
    
    comparison = ModelComparison()
    comparison.load_results()
    
    if comparison.results:
        comparison.compare_metrics()
        comparison.create_comparison_chart()
        comparison.generate_report()
        print("模型对比完成！")
    else:
        print("没有可用的模型结果，请先训练模型。")


def step_6_disease_classification():
    """步骤6：疾病分类"""
    print("\n" + "=" * 60)
    print("步骤 6: 疾病大类预测")
    print("=" * 60)
    
    try:
        from models.disease_classification import load_embeddings_for_classification
        embeddings = load_embeddings_for_classification('transe', 'models')
        
        if embeddings:
            results = run_classification_experiment(embeddings, 'results')
            if results:
                plot_classification_results(results)
                print("疾病分类完成！")
        else:
            print("没有可用的嵌入数据")
    except Exception as e:
        print(f"疾病分类失败: {e}")


def step_7_hallucination_detection():
    """步骤7：幻觉检测"""
    print("\n" + "=" * 60)
    print("步骤 7: LLM 幻觉检测")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        df = pd.read_csv('data/mental_subgraph.csv')
        generator = TripletGenerator(df)
        dataset_gen = HallucinationDataset(generator)
        
        # 生成数据集
        dataset = dataset_gen.generate_dataset(real_count=30, false_count=30)
        
        # 生成模拟结果
        results = generate_synthetic_results(dataset, llm_accuracy=0.75)
        
        # 评估
        evaluator = HallucinationEvaluator(results)
        eval_result = evaluator.evaluate()
        
        evaluator.save_results('results/hallucination_evaluation.json')
        evaluator.plot_results()
        
        report = evaluator.generate_report()
        print(report)
        
        print("\n幻觉检测完成！")
    except Exception as e:
        print(f"幻觉检测失败: {e}")


def step_8_build_rag():
    """步骤8：构建RAG"""
    print("\n" + "=" * 60)
    print("步骤 8: 构建 RAG 向量数据库")
    print("=" * 60)
    
    try:
        import pandas as pd
        
        df = pd.read_csv('data/mental_subgraph.csv')
        vector_store = build_vector_store(df, 'data', 'simple')
        
        print("RAG 向量数据库构建完成！")
        return vector_store
    except Exception as e:
        print(f"RAG 构建失败: {e}")
        return None


def step_9_start_webapp():
    """步骤9：启动Web应用"""
    print("\n" + "=" * 60)
    print("步骤 9: 启动 Web 应用")
    print("=" * 60)
    
    print("\n正在启动 Web 应用...")
    print("访问地址: http://localhost:5000")
    print("\n按 Ctrl+C 停止服务器\n")
    
    from web.app import app
    app.run(debug=True, host='0.0.0.0', port=5000)


def main():
    """主函数"""
    print_banner()
    
    parser = argparse.ArgumentParser(description='精神障碍知识图谱分析与可信问答系统')
    parser.add_argument('--step', type=int, choices=range(1, 10),
                       help='运行特定步骤 (1-9)')
    parser.add_argument('--all', action='store_true',
                       help='运行所有步骤')
    parser.add_argument('--web', action='store_true',
                       help='启动 Web 应用')
    
    args = parser.parse_args()
    
    if args.web:
        step_9_start_webapp()
        return
    
    if args.all:
        # 运行所有步骤
        step_1_download_data()
        edges_df = step_2_extract_subgraph()
        step_3_analyze_graph(edges_df)
        step_4_train_models()
        step_5_compare_models()
        step_6_disease_classification()
        step_7_hallucination_detection()
        step_8_build_rag()
        
        print("\n" + "=" * 60)
        print("所有步骤完成！启动 Web 应用？")
        print("=" * 60)
        
        response = input("输入 'y' 启动 Web 应用: ")
        if response.lower() == 'y':
            step_9_start_webapp()
        
        return
    
    if args.step:
        # 运行特定步骤
        if args.step == 1:
            step_1_download_data()
        elif args.step == 2:
            step_2_extract_subgraph()
        elif args.step == 3:
            edges_df = load_primekg('data/mental_subgraph.csv')
            step_3_analyze_graph(edges_df)
        elif args.step == 4:
            step_4_train_models()
        elif args.step == 5:
            step_5_compare_models()
        elif args.step == 6:
            step_6_disease_classification()
        elif args.step == 7:
            step_7_hallucination_detection()
        elif args.step == 8:
            step_8_build_rag()
        elif args.step == 9:
            step_9_start_webapp()
        
        return
    
    # 默认：显示帮助信息
    print("\n使用说明：")
    print("  python main.py --all          运行所有步骤")
    print("  python main.py --step 1      运行步骤1（下载数据）")
    print("  python main.py --step 2      运行步骤2（提取子图）")
    print("  python main.py --step 3      运行步骤3（图分析）")
    print("  python main.py --step 4      运行步骤4（训练模型）")
    print("  python main.py --step 5      运行步骤5（模型对比）")
    print("  python main.py --step 6      运行步骤6（疾病分类）")
    print("  python main.py --step 7      运行步骤7（幻觉检测）")
    print("  python main.py --step 8      运行步骤8（构建RAG）")
    print("  python main.py --step 9      运行步骤9（启动Web）")
    print("  python main.py --web         直接启动 Web 应用")


if __name__ == '__main__':
    main()
