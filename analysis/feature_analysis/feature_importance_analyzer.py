"""
特徴量重要度分析器
================

IRL学習後の重みファイルを読み込み、特徴量重要度を計算・分析する機能を提供します。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns  # Optional dependency
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """特徴量重要度分析器
    
    IRL学習後の重みファイルを読み込み、特徴量重要度を計算する機能を実装。
    重要度ランキング、カテゴリ別重要度比較、統計的有意性検証メソッドを提供。
    """
    
    def __init__(self, irl_weights_path: str, feature_names: List[str]):
        """
        Args:
            irl_weights_path: IRL重みファイルのパス
            feature_names: 特徴量名のリスト
        """
        self.irl_weights_path = Path(irl_weights_path)
        self.feature_names = feature_names
        self.weights = None
        self.feature_categories = None
        
        # 重みファイルを読み込み
        self._load_weights()
        
        # 特徴量カテゴリを定義
        self._define_feature_categories()
        
        logger.info(f"FeatureImportanceAnalyzer初期化完了: {len(self.feature_names)}特徴量")
    
    def _load_weights(self) -> None:
        """IRL重みファイルを読み込み"""
        try:
            if not self.irl_weights_path.exists():
                raise FileNotFoundError(f"重みファイルが見つかりません: {self.irl_weights_path}")
            
            self.weights = np.load(self.irl_weights_path)
            
            if len(self.weights) != len(self.feature_names):
                logger.warning(
                    f"重み次元数({len(self.weights)})と特徴量数({len(self.feature_names)})が一致しません"
                )
                # 次元数を合わせる
                min_dim = min(len(self.weights), len(self.feature_names))
                self.weights = self.weights[:min_dim]
                self.feature_names = self.feature_names[:min_dim]
            
            logger.info(f"IRL重み読み込み成功: {len(self.weights)}次元")
            
        except Exception as e:
            logger.error(f"重みファイル読み込みエラー: {e}")
            raise
    
    def _define_feature_categories(self) -> None:
        """特徴量カテゴリを定義"""
        self.feature_categories = {}
        
        for i, name in enumerate(self.feature_names):
            if name.startswith("task_"):
                category = "タスク特徴量"
            elif name.startswith("dev_"):
                category = "開発者特徴量"
            elif name.startswith("match_"):
                category = "マッチング特徴量"
            elif name.startswith("gat_") and "emb_" not in name:
                category = "GAT統計特徴量"
            elif "gat_" in name or name.startswith("feature_"):
                category = "GAT埋め込み特徴量"
            else:
                category = "その他特徴量"
            
            if category not in self.feature_categories:
                self.feature_categories[category] = []
            self.feature_categories[category].append(i)
    
    def analyze_importance(self) -> Dict[str, Any]:
        """重要度分析を実行
        
        Returns:
            分析結果を含む辞書
        """
        results = {
            'importance_ranking': self._rank_by_importance(),
            'category_importance': self._analyze_by_category(),
            'statistical_significance': self._test_significance(),
            'basic_vs_gat_comparison': self._compare_basic_vs_gat()
        }
        
        logger.info("特徴量重要度分析完了")
        return results
    
    def _rank_by_importance(self) -> List[Tuple[str, float, float]]:
        """重要度ランキング作成
        
        Returns:
            (特徴量名, IRL重み, 重要度)のタプルリスト（重要度降順）
        """
        importance_data = []
        
        for i, (name, weight) in enumerate(zip(self.feature_names, self.weights)):
            importance = abs(weight)
            importance_data.append((name, float(weight), float(importance)))
        
        # 重要度で降順ソート
        importance_data.sort(key=lambda x: x[2], reverse=True)
        
        return importance_data
    
    def _analyze_by_category(self) -> Dict[str, Dict[str, float]]:
        """カテゴリ別重要度分析
        
        Returns:
            カテゴリ別の統計情報
        """
        category_stats = {}
        
        for category, indices in self.feature_categories.items():
            if not indices:
                continue
                
            category_weights = self.weights[indices]
            category_importance = np.abs(category_weights)
            
            stats_dict = {
                'count': len(indices),
                'mean_weight': float(np.mean(category_weights)),
                'std_weight': float(np.std(category_weights)),
                'mean_importance': float(np.mean(category_importance)),
                'max_importance': float(np.max(category_importance)),
                'min_importance': float(np.min(category_importance)),
                'positive_weights': int(np.sum(category_weights > 0)),
                'negative_weights': int(np.sum(category_weights < 0)),
                'zero_weights': int(np.sum(category_weights == 0)),
                'weight_range': [float(np.min(category_weights)), float(np.max(category_weights))]
            }
            
            category_stats[category] = stats_dict
        
        return category_stats
    
    def _test_significance(self) -> Dict[str, float]:
        """統計的有意性検証
        
        Returns:
            統計的有意性テストの結果
        """
        significance_results = {}
        
        # 重みがゼロと有意に異なるかのt検定
        t_stat, p_value = stats.ttest_1samp(self.weights, 0)
        significance_results['weights_vs_zero_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
        
        # 重みの正規性検定
        shapiro_stat, shapiro_p = stats.shapiro(self.weights)
        significance_results['normality_test'] = {
            'shapiro_statistic': float(shapiro_stat),
            'shapiro_p_value': float(shapiro_p),
            'is_normal': shapiro_p > 0.05
        }
        
        # カテゴリ間の重み分布の差の検定（ANOVA）
        category_weights_lists = []
        category_names = []
        
        for category, indices in self.feature_categories.items():
            if len(indices) > 1:  # 最低2個以上の特徴量があるカテゴリのみ
                category_weights_lists.append(self.weights[indices])
                category_names.append(category)
        
        if len(category_weights_lists) >= 2:
            try:
                f_stat, anova_p = stats.f_oneway(*category_weights_lists)
                significance_results['category_anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(anova_p),
                    'significant_difference': anova_p < 0.05,
                    'tested_categories': category_names
                }
            except Exception as e:
                logger.warning(f"ANOVA検定でエラー: {e}")
                significance_results['category_anova'] = {'error': str(e)}
        
        return significance_results
    
    def _compare_basic_vs_gat(self) -> Dict[str, Any]:
        """基本特徴量とGAT特徴量の重要度比較
        
        Returns:
            基本特徴量とGAT特徴量の比較結果
        """
        # 基本特徴量のインデックス
        basic_indices = []
        gat_indices = []
        
        for category, indices in self.feature_categories.items():
            if "GAT" in category:
                gat_indices.extend(indices)
            else:
                basic_indices.extend(indices)
        
        if not basic_indices or not gat_indices:
            return {'error': '基本特徴量またはGAT特徴量が見つかりません'}
        
        basic_weights = self.weights[basic_indices]
        gat_weights = self.weights[gat_indices]
        
        basic_importance = np.abs(basic_weights)
        gat_importance = np.abs(gat_weights)
        
        comparison = {
            'basic_features': {
                'count': len(basic_indices),
                'mean_importance': float(np.mean(basic_importance)),
                'std_importance': float(np.std(basic_importance)),
                'max_importance': float(np.max(basic_importance)),
                'top_features': self._get_top_features_in_indices(basic_indices, 5)
            },
            'gat_features': {
                'count': len(gat_indices),
                'mean_importance': float(np.mean(gat_importance)),
                'std_importance': float(np.std(gat_importance)),
                'max_importance': float(np.max(gat_importance)),
                'top_features': self._get_top_features_in_indices(gat_indices, 5)
            }
        }
        
        # 統計的比較（Mann-Whitney U検定）
        try:
            u_stat, u_p = stats.mannwhitneyu(basic_importance, gat_importance, alternative='two-sided')
            comparison['statistical_test'] = {
                'test': 'Mann-Whitney U',
                'u_statistic': float(u_stat),
                'p_value': float(u_p),
                'significant_difference': u_p < 0.05
            }
        except Exception as e:
            logger.warning(f"Mann-Whitney U検定でエラー: {e}")
            comparison['statistical_test'] = {'error': str(e)}
        
        return comparison
    
    def _get_top_features_in_indices(self, indices: List[int], top_k: int) -> List[Dict[str, Any]]:
        """指定されたインデックス内でのトップ特徴量を取得
        
        Args:
            indices: 対象インデックスのリスト
            top_k: 上位何個を取得するか
            
        Returns:
            トップ特徴量の情報リスト
        """
        feature_data = []
        
        for idx in indices:
            if idx < len(self.feature_names) and idx < len(self.weights):
                feature_data.append({
                    'name': self.feature_names[idx],
                    'weight': float(self.weights[idx]),
                    'importance': float(abs(self.weights[idx])),
                    'index': idx
                })
        
        # 重要度で降順ソート
        feature_data.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_data[:top_k]
    
    def generate_importance_report(self, output_path: Optional[str] = None) -> str:
        """重要度分析レポートを生成
        
        Args:
            output_path: 出力ファイルパス（Noneの場合は自動生成）
            
        Returns:
            生成されたレポートファイルのパス
        """
        analysis_results = self.analyze_importance()
        
        if output_path is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"feature_importance_report_{timestamp}.txt"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("特徴量重要度分析レポート\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本統計
            f.write("【基本統計】\n")
            f.write(f"特徴量数: {len(self.feature_names)}\n")
            f.write(f"重み平均: {np.mean(self.weights):.6f}\n")
            f.write(f"重み標準偏差: {np.std(self.weights):.6f}\n")
            f.write(f"重み範囲: [{np.min(self.weights):.6f}, {np.max(self.weights):.6f}]\n")
            f.write(f"正の重み: {np.sum(self.weights > 0)}個\n")
            f.write(f"負の重み: {np.sum(self.weights < 0)}個\n")
            f.write(f"ゼロの重み: {np.sum(self.weights == 0)}個\n\n")
            
            # 重要度ランキング
            f.write("【重要度ランキング TOP20】\n")
            ranking = analysis_results['importance_ranking']
            for i, (name, weight, importance) in enumerate(ranking[:20], 1):
                f.write(f"{i:2d}. {name:40s} | 重み:{weight:8.5f} | 重要度:{importance:8.5f}\n")
            f.write("\n")
            
            # カテゴリ別分析
            f.write("【カテゴリ別分析】\n")
            category_stats = analysis_results['category_importance']
            for category, stats in category_stats.items():
                f.write(f"\n{category}:\n")
                f.write(f"  特徴量数: {stats['count']}\n")
                f.write(f"  平均重み: {stats['mean_weight']:.6f}\n")
                f.write(f"  平均重要度: {stats['mean_importance']:.6f}\n")
                f.write(f"  最大重要度: {stats['max_importance']:.6f}\n")
                f.write(f"  正の重み: {stats['positive_weights']}個\n")
                f.write(f"  負の重み: {stats['negative_weights']}個\n")
            
            # 基本特徴量 vs GAT特徴量比較
            f.write("\n【基本特徴量 vs GAT特徴量比較】\n")
            comparison = analysis_results['basic_vs_gat_comparison']
            if 'error' not in comparison:
                f.write("基本特徴量:\n")
                basic = comparison['basic_features']
                f.write(f"  特徴量数: {basic['count']}\n")
                f.write(f"  平均重要度: {basic['mean_importance']:.6f}\n")
                f.write(f"  最大重要度: {basic['max_importance']:.6f}\n")
                
                f.write("\nGAT特徴量:\n")
                gat = comparison['gat_features']
                f.write(f"  特徴量数: {gat['count']}\n")
                f.write(f"  平均重要度: {gat['mean_importance']:.6f}\n")
                f.write(f"  最大重要度: {gat['max_importance']:.6f}\n")
                
                if 'statistical_test' in comparison and 'error' not in comparison['statistical_test']:
                    test = comparison['statistical_test']
                    f.write(f"\n統計的比較 ({test['test']}):\n")
                    f.write(f"  p値: {test['p_value']:.6f}\n")
                    f.write(f"  有意差: {'あり' if test['significant_difference'] else 'なし'}\n")
            
            # 統計的有意性
            f.write("\n【統計的有意性検定】\n")
            significance = analysis_results['statistical_significance']
            
            if 'weights_vs_zero_ttest' in significance:
                ttest = significance['weights_vs_zero_ttest']
                f.write(f"重みゼロとの比較 (t検定):\n")
                f.write(f"  t統計量: {ttest['t_statistic']:.6f}\n")
                f.write(f"  p値: {ttest['p_value']:.6f}\n")
                f.write(f"  有意: {'はい' if ttest['significant'] else 'いいえ'}\n\n")
            
            if 'normality_test' in significance:
                norm = significance['normality_test']
                f.write(f"正規性検定 (Shapiro-Wilk):\n")
                f.write(f"  統計量: {norm['shapiro_statistic']:.6f}\n")
                f.write(f"  p値: {norm['shapiro_p_value']:.6f}\n")
                f.write(f"  正規分布: {'はい' if norm['is_normal'] else 'いいえ'}\n\n")
        
        logger.info(f"重要度分析レポート生成完了: {output_path}")
        return str(output_path)
    
    def visualize_importance(self, output_dir: Optional[str] = None, show_plot: bool = True) -> str:
        """重要度分析の可視化
        
        Args:
            output_dir: 出力ディレクトリ（Noneの場合は自動生成）
            show_plot: プロットを表示するかどうか
            
        Returns:
            生成された図のファイルパス
        """
        analysis_results = self.analyze_importance()
        
        if output_dir is None:
            output_dir = Path("outputs")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 図を作成
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('特徴量重要度分析', fontsize=16, fontweight='bold')
        
        # 1. 重み分布ヒストグラム
        axes[0, 0].hist(self.weights, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('IRL重み')
        axes[0, 0].set_ylabel('頻度')
        axes[0, 0].set_title('IRL重みの分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 重要度ランキング TOP15
        ranking = analysis_results['importance_ranking'][:15]
        names = [item[0][:20] + '...' if len(item[0]) > 20 else item[0] for item in ranking]
        weights = [item[1] for item in ranking]
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        y_pos = np.arange(len(names))
        axes[0, 1].barh(y_pos, weights, color=colors, alpha=0.7)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(names, fontsize=8)
        axes[0, 1].set_xlabel('IRL重み')
        axes[0, 1].set_title('重要特徴量 TOP15')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. カテゴリ別平均重要度
        category_stats = analysis_results['category_importance']
        categories = list(category_stats.keys())
        mean_importance = [category_stats[cat]['mean_importance'] for cat in categories]
        
        axes[0, 2].bar(range(len(categories)), mean_importance, alpha=0.7, color='lightcoral')
        axes[0, 2].set_xticks(range(len(categories)))
        axes[0, 2].set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
        axes[0, 2].set_ylabel('平均重要度')
        axes[0, 2].set_title('カテゴリ別平均重要度')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 重み vs 重要度散布図
        all_weights = [item[1] for item in ranking]
        all_importance = [item[2] for item in ranking]
        
        axes[1, 0].scatter(all_weights, all_importance, alpha=0.6, color='purple')
        axes[1, 0].set_xlabel('IRL重み')
        axes[1, 0].set_ylabel('重要度（絶対値）')
        axes[1, 0].set_title('重み vs 重要度')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. カテゴリ別重み分布（箱ひげ図）
        category_weights_data = []
        category_labels = []
        
        for category, indices in self.feature_categories.items():
            if len(indices) > 0:
                category_weights_data.append(self.weights[indices])
                category_labels.append(category[:10])  # ラベルを短縮
        
        if category_weights_data:
            axes[1, 1].boxplot(category_weights_data, labels=category_labels)
            axes[1, 1].set_xticklabels(category_labels, rotation=45, ha='right', fontsize=8)
            axes[1, 1].set_ylabel('IRL重み')
            axes[1, 1].set_title('カテゴリ別重み分布')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 基本特徴量 vs GAT特徴量比較
        comparison = analysis_results['basic_vs_gat_comparison']
        if 'error' not in comparison:
            basic_importance = comparison['basic_features']['mean_importance']
            gat_importance = comparison['gat_features']['mean_importance']
            
            axes[1, 2].bar(['基本特徴量', 'GAT特徴量'], 
                          [basic_importance, gat_importance],
                          color=['lightblue', 'lightgreen'], alpha=0.7)
            axes[1, 2].set_ylabel('平均重要度')
            axes[1, 2].set_title('基本特徴量 vs GAT特徴量')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = output_dir / f"feature_importance_analysis_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        logger.info(f"重要度分析可視化完了: {fig_path}")
        return str(fig_path)