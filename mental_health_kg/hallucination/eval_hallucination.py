"""
幻觉检测评估模块
对 LLM 在精神障碍知识上的幻觉进行量化评估
"""
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict

# 设置中文显示
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HallucinationEvaluator:
    """幻觉检测评估器"""
    
    def __init__(self, results: List[Dict]):
        self.results = results
        self.evaluation = {}
    
    def evaluate(self) -> Dict:
        """执行完整评估"""
        self.evaluation = self._compute_basic_metrics()
        self.evaluation['by_relation_type'] = self._compute_by_relation_type()
        self.evaluation['by_entity_type'] = self._compute_by_entity_type()
        self.evaluation['detailed_analysis'] = self._compute_detailed_analysis()
        
        return self.evaluation
    
    def _compute_basic_metrics(self) -> Dict:
        """计算基础指标"""
        total = len(self.results)
        if total == 0:
            return {}
        
        correct = sum(1 for r in self.results if r.get('is_correct', False))
        incorrect = total - correct
        
        # 区分真实和伪造
        real_triplets = [r for r in self.results if r.get('is_real', True)]
        false_triplets = [r for r in self.results if not r.get('is_real', True)]
        
        real_correct = sum(1 for r in real_triplets if r.get('is_correct', False))
        false_correct = sum(1 for r in false_triplets if r.get('is_correct', False))
        
        metrics = {
            'total_samples': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': round(correct / total, 4),
            'hallucination_rate': round(1 - correct / total, 4),
            'real_triplets': {
                'total': len(real_triplets),
                'correct': real_correct,
                'accuracy': round(real_correct / len(real_triplets), 4) if real_triplets else 0,
                'miss_rate': round(1 - real_correct / len(real_triplets), 4) if real_triplets else 0
            },
            'false_triplets': {
                'total': len(false_triplets),
                'correct': false_correct,
                'accuracy': round(false_correct / len(false_triplets), 4) if false_triplets else 0,
                'detection_rate': round(false_correct / len(false_triplets), 4) if false_triplets else 0
            }
        }
        
        return metrics
    
    def _compute_by_relation_type(self) -> Dict:
        """按关系类型分析"""
        by_rel = defaultdict(lambda: {'correct': 0, 'total': 0, 'correct_real': 0, 
                                      'correct_false': 0, 'total_real': 0, 'total_false': 0})
        
        for r in self.results:
            rel_type = r.get('relation_type', 'other')
            is_correct = r.get('is_correct', False)
            is_real = r.get('is_real', True)
            
            by_rel[rel_type]['total'] += 1
            if is_correct:
                by_rel[rel_type]['correct'] += 1
            
            if is_real:
                by_rel[rel_type]['total_real'] += 1
                if is_correct:
                    by_rel[rel_type]['correct_real'] += 1
            else:
                by_rel[rel_type]['total_false'] += 1
                if is_correct:
                    by_rel[rel_type]['correct_false'] += 1
        
        result = {}
        for rel_type, stats in by_rel.items():
            result[rel_type] = {
                'total_samples': stats['total'],
                'accuracy': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0,
                'real_triplets': {
                    'correct': stats['correct_real'],
                    'total': stats['total_real'],
                    'miss_rate': round(1 - stats['correct_real'] / stats['total_real'], 4) if stats['total_real'] > 0 else 0
                },
                'false_triplets': {
                    'correct': stats['correct_false'],
                    'total': stats['total_false'],
                    'detection_rate': round(stats['correct_false'] / stats['total_false'], 4) if stats['total_false'] > 0 else 0
                }
            }
        
        return result
    
    def _compute_by_entity_type(self) -> Dict:
        """按实体类型分析"""
        by_entity = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for r in self.results:
            # 从三元组中提取头实体类型
            head = r.get('head', '')
            
            # 简单的类型推断
            entity_type = self._infer_entity_type(head)
            
            is_correct = r.get('is_correct', False)
            by_entity[entity_type]['total'] += 1
            if is_correct:
                by_entity[entity_type]['correct'] += 1
        
        result = {}
        for entity_type, stats in by_entity.items():
            result[entity_type] = {
                'total_samples': stats['total'],
                'accuracy': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0,
                'hallucination_rate': round(1 - stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0
            }
        
        return result
    
    def _infer_entity_type(self, entity_name: str) -> str:
        """推断实体类型"""
        name_lower = entity_name.lower()
        
        mental_keywords = ['depression', 'anxiety', 'schizophrenia', 'bipolar', 'ptsd', 
                         'ocd', 'autism', 'adhd', 'dementia', 'disorder']
        symptom_keywords = ['symptom', 'pain', 'fatigue', 'insomnia', 'headache']
        drug_keywords = ['drug', 'medication', 'fluoxetine', 'sertraline', 'lithium']
        
        if any(kw in name_lower for kw in mental_keywords):
            return 'mental_disorder'
        elif any(kw in name_lower for kw in symptom_keywords):
            return 'symptom'
        elif any(kw in name_lower for kw in drug_keywords):
            return 'drug'
        else:
            return 'other'
    
    def _compute_detailed_analysis(self) -> Dict:
        """详细分析"""
        analysis = {
            'most_confused_samples': [],
            'patterns': []
        }
        
        # 找出 LLM 容易混淆的样本
        incorrect_samples = [r for r in self.results if not r.get('is_correct', False)]
        
        for sample in incorrect_samples[:10]:
            analysis['most_confused_samples'].append({
                'question': sample.get('question', '')[:100],
                'triplet': sample.get('triplet', ()),
                'llm_answer': sample.get('llm_answer', ''),
                'llm_reason': sample.get('llm_reason', '')[:100]
            })
        
        # 分析错误模式
        error_patterns = defaultdict(int)
        for sample in incorrect_samples:
            rel_type = sample.get('relation_type', 'unknown')
            error_patterns[rel_type] += 1
        
        analysis['error_patterns'] = dict(error_patterns)
        
        return analysis
    
    def generate_report(self) -> str:
        """生成评估报告"""
        if not self.evaluation:
            self.evaluate()
        
        report = "# LLM 幻觉检测评估报告\n\n"
        
        report += "## 整体性能\n\n"
        report += f"| 指标 | 数值 |\n"
        report += f"|------|------|\n"
        report += f"| 总样本数 | {self.evaluation.get('total_samples', 0)} |\n"
        report += f"| 正确数 | {self.evaluation.get('correct', 0)} |\n"
        report += f"| 准确率 | {self.evaluation.get('accuracy', 0)*100:.2f}% |\n"
        report += f"| 幻觉率 | {self.evaluation.get('hallucination_rate', 0)*100:.2f}% |\n\n"
        
        report += "## 真实三元组检测\n\n"
        real = self.evaluation.get('real_triplets', {})
        report += f"- 真实三元组总数: {real.get('total', 0)}\n"
        report += f"- 正确识别数: {real.get('correct', 0)}\n"
        report += f"- 遗漏率 (假阴性): {real.get('miss_rate', 0)*100:.2f}%\n\n"
        
        report += "## 伪造三元组检测\n\n"
        false = self.evaluation.get('false_triplets', {})
        report += f"- 伪造三元组总数: {false.get('total', 0)}\n"
        report += f"- 正确拒绝数: {false.get('correct', 0)}\n"
        report += f"- 检测率: {false.get('detection_rate', 0)*100:.2f}%\n\n"
        
        report += "## 按关系类型分析\n\n"
        report += f"| 关系类型 | 样本数 | 准确率 | 幻觉率 |\n"
        report += f"|----------|--------|--------|--------|\n"
        
        by_rel = self.evaluation.get('by_relation_type', {})
        for rel_type, stats in by_rel.items():
            report += f"| {rel_type} | {stats['total_samples']} | {stats['accuracy']*100:.2f}% | {100-stats['accuracy']*100:.2f}% |\n"
        
        report += "\n## 错误模式分析\n\n"
        patterns = self.evaluation.get('detailed_analysis', {}).get('error_patterns', {})
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            report += f"- {pattern}: {count} 次\n"
        
        return report
    
    def save_results(self, output_path='results/hallucination_evaluation.json'):
        """保存评估结果"""
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'results', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存: {output_path}")
    
    def plot_results(self, output_path='results/hallucination_analysis.png'):
        """可视化评估结果"""
        if not self.evaluation:
            self.evaluate()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 整体准确率
        ax1 = axes[0, 0]
        labels = ['Correct', 'Incorrect']
        sizes = [self.evaluation.get('correct', 0), self.evaluation.get('incorrect', 0)]
        colors = ['#2ecc71', '#e74c3c']
        if sum(sizes) > 0:
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('Overall Accuracy')
        
        # 2. 真实 vs 伪造检测率
        ax2 = axes[0, 1]
        real_rate = self.evaluation.get('real_triplets', {}).get('accuracy', 0) * 100
        false_rate = self.evaluation.get('false_triplets', {}).get('detection_rate', 0) * 100
        bars = ax2.bar(['Real Triplets\nCorrect Rate', 'False Triplets\nDetection Rate'], 
                      [real_rate, false_rate], color=['#3498db', '#9b59b6'])
        ax2.set_ylabel('Rate (%)')
        ax2.set_title('Detection Performance')
        ax2.set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # 3. 按关系类型的幻觉率
        ax3 = axes[1, 0]
        by_rel = self.evaluation.get('by_relation_type', {})
        if by_rel:
            rel_types = list(by_rel.keys())
            hallucination_rates = [(1 - by_rel[rt]['accuracy']) * 100 for rt in rel_types]
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(rel_types)))
            ax3.barh(list(rel_types), hallucination_rates, color=colors)
            ax3.set_xlabel('Hallucination Rate (%)')
            ax3.set_title('Hallucination Rate by Relation Type')
            ax3.set_xlim(0, 100)
        
        # 4. 按关系类型的准确率
        ax4 = axes[1, 1]
        if by_rel:
            rel_types_for_ax4 = list(by_rel.keys())
            accuracies = [by_rel[rt]['accuracy'] * 100 for rt in rel_types_for_ax4]
            ax4.barh(rel_types_for_ax4, accuracies, color='#3498db')
            ax4.set_xlabel('Accuracy (%)')
            ax4.set_title('Accuracy by Relation Type')
            ax4.set_xlim(0, 100)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'results', exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"可视化结果已保存: {output_path}")


def compare_llm_models(results_dict: Dict[str, List[Dict]]) -> pd.DataFrame:
    """对比多个 LLM 模型的幻觉检测性能"""
    comparison_data = []
    
    for model_name, results in results_dict.items():
        evaluator = HallucinationEvaluator(results)
        eval_result = evaluator.evaluate()
        
        by_rel = eval_result.get('by_relation_type', {})
        row = {
            'Model': model_name,
            'Accuracy': eval_result.get('accuracy', 0) * 100,
            'Hallucination Rate': eval_result.get('hallucination_rate', 0) * 100,
            'Real Triplets Detection': eval_result.get('real_triplets', {}).get('accuracy', 0) * 100,
            'False Triplets Detection': eval_result.get('false_triplets', {}).get('detection_rate', 0) * 100,
        }
        
        # 动态添加各关系类型的准确率，避免硬编码键名
        for rel_type, stats in by_rel.items():
            col_name = f'{rel_type} Accuracy'
            row[col_name] = stats.get('accuracy', 0) * 100
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


if __name__ == '__main__':
    print("幻觉检测评估模块")
    print("请先运行幻觉检测实验以获取结果")
