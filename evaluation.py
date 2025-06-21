import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RecommendationEvaluator:
    """Evaluator for recommendation systems using MAP@12 and other metrics"""
    
    def __init__(self, transactions_df):
        self.transactions = transactions_df.copy()
        
    def create_validation_split(self, test_days=7, validation_days=7):
        """Create validation split based on time"""
        print("Creating time-based validation split...")
        
        max_date = self.transactions['t_dat'].max()
        test_start = max_date - timedelta(days=test_days)
        validation_start = test_start - timedelta(days=validation_days)
        
        # Train set: everything before validation period
        train_data = self.transactions[self.transactions['t_dat'] < validation_start]
        
        # Validation set: middle period (for hyperparameter tuning)
        validation_data = self.transactions[
            (self.transactions['t_dat'] >= validation_start) & 
            (self.transactions['t_dat'] < test_start)
        ]
        
        # Test set: last period (for final evaluation)
        test_data = self.transactions[self.transactions['t_dat'] >= test_start]
        
        print(f"Train period: {train_data['t_dat'].min()} to {train_data['t_dat'].max()}")
        print(f"Validation period: {validation_data['t_dat'].min()} to {validation_data['t_dat'].max()}")
        print(f"Test period: {test_data['t_dat'].min()} to {test_data['t_dat'].max()}")
        
        print(f"Train transactions: {len(train_data):,}")
        print(f"Validation transactions: {len(validation_data):,}")
        print(f"Test transactions: {len(test_data):,}")
        
        return train_data, validation_data, test_data
    
    def prepare_ground_truth(self, test_data):
        """Prepare ground truth from test data"""
        print("Preparing ground truth...")
        
        # Group by customer to get actual purchases
        actual_purchases = test_data.groupby('customer_id')['article_id'].apply(list).to_dict()
        
        print(f"Ground truth prepared for {len(actual_purchases)} customers")
        return actual_purchases
    
    def calculate_precision_at_k(self, recommended, actual, k=12):
        """Calculate Precision@K for a single customer"""
        if len(actual) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_items = set(actual)
        recommended_set = set(recommended_k)
        
        intersection = len(recommended_set & relevant_items)
        precision = intersection / len(recommended_k) if len(recommended_k) > 0 else 0.0
        
        return precision
    
    def calculate_recall_at_k(self, recommended, actual, k=12):
        """Calculate Recall@K for a single customer"""
        if len(actual) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_items = set(actual)
        recommended_set = set(recommended_k)
        
        intersection = len(recommended_set & relevant_items)
        recall = intersection / len(relevant_items)
        
        return recall
    
    def calculate_average_precision_at_k(self, recommended, actual, k=12):
        """Calculate Average Precision@K for a single customer (AP@K)"""
        if len(actual) == 0:
            return 0.0
        
        recommended_k = recommended[:k]
        relevant_items = set(actual)
        
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(recommended_k):
            if item in relevant_items:
                num_hits += 1.0
                precision_at_i = num_hits / (i + 1.0)
                score += precision_at_i
        
        # Normalize by minimum of k and number of relevant items
        norm_factor = min(len(relevant_items), k)
        if norm_factor == 0:
            return 0.0
        
        return score / norm_factor
    
    def calculate_map_at_k(self, recommendations_dict, actual_purchases_dict, k=12):
        """Calculate Mean Average Precision@K (MAP@K)"""
        print(f"Calculating MAP@{k}...")
        
        average_precisions = []
        customers_with_purchases = 0
        
        for customer_id, actual_items in actual_purchases_dict.items():
            if customer_id in recommendations_dict:
                recommended_items = recommendations_dict[customer_id]
                ap = self.calculate_average_precision_at_k(recommended_items, actual_items, k)
                average_precisions.append(ap)
                
                if len(actual_items) > 0:
                    customers_with_purchases += 1
        
        if len(average_precisions) == 0:
            return 0.0
        
        map_score = np.mean(average_precisions)
        
        print(f"MAP@{k}: {map_score:.6f}")
        print(f"Evaluated customers: {len(average_precisions)}")
        print(f"Customers with purchases: {customers_with_purchases}")
        
        return map_score
    
    def calculate_coverage_metrics(self, recommendations_dict, all_items_set):
        """Calculate coverage metrics"""
        print("Calculating coverage metrics...")
        
        # Catalog coverage: percentage of items that appear in recommendations
        recommended_items = set()
        for recs in recommendations_dict.values():
            recommended_items.update(recs)
        
        catalog_coverage = len(recommended_items) / len(all_items_set)
        
        # User coverage: percentage of users that get personalized recommendations
        user_coverage = len(recommendations_dict) / len(recommendations_dict)  # Always 1.0 in our case
        
        print(f"Catalog coverage: {catalog_coverage:.4f}")
        print(f"Unique items recommended: {len(recommended_items):,}")
        print(f"Total items in catalog: {len(all_items_set):,}")
        
        return {
            'catalog_coverage': catalog_coverage,
            'user_coverage': user_coverage,
            'unique_items_recommended': len(recommended_items)
        }
    
    def calculate_diversity_metrics(self, recommendations_dict, articles_df):
        """Calculate diversity metrics"""
        print("Calculating diversity metrics...")
        
        # Category diversity: how many different categories are recommended
        all_categories = set()
        category_counts = {}
        
        for customer_id, recs in recommendations_dict.items():
            customer_categories = articles_df[
                articles_df['article_id'].isin(recs)
            ]['product_type_name'].value_counts().to_dict()
            
            for category, count in customer_categories.items():
                if pd.notna(category):
                    all_categories.add(category)
                    category_counts[category] = category_counts.get(category, 0) + count
        
        # Calculate entropy for category distribution
        total_recommendations = sum(category_counts.values())
        category_probs = [count / total_recommendations for count in category_counts.values()]
        
        if len(category_probs) > 1:
            entropy = -sum(p * np.log2(p) for p in category_probs if p > 0)
        else:
            entropy = 0.0
        
        print(f"Categories in recommendations: {len(all_categories)}")
        print(f"Category entropy: {entropy:.4f}")
        
        return {
            'category_count': len(all_categories),
            'category_entropy': entropy,
            'category_distribution': category_counts
        }
    
    def calculate_novelty_metrics(self, recommendations_dict, popularity_scores):
        """Calculate novelty metrics (how non-popular are the recommendations)"""
        print("Calculating novelty metrics...")
        
        novelty_scores = []
        
        for customer_id, recs in recommendations_dict.items():
            customer_novelty = []
            for item in recs:
                if item in popularity_scores:
                    # Novelty = 1 - popularity (normalized)
                    novelty = 1 - popularity_scores[item]
                    customer_novelty.append(novelty)
            
            if customer_novelty:
                avg_novelty = np.mean(customer_novelty)
                novelty_scores.append(avg_novelty)
        
        overall_novelty = np.mean(novelty_scores) if novelty_scores else 0.0
        
        print(f"Average novelty: {overall_novelty:.4f}")
        
        return overall_novelty
    
    def comprehensive_evaluation(self, recommendations_dict, actual_purchases_dict, 
                                articles_df, popularity_scores=None):
        """Perform comprehensive evaluation"""
        print("=== COMPREHENSIVE EVALUATION ===")
        
        results = {}
        
        # MAP@12 (main metric)
        results['map_12'] = self.calculate_map_at_k(recommendations_dict, actual_purchases_dict, k=12)
        
        # Additional MAP scores
        for k in [1, 3, 5, 10]:
            results[f'map_{k}'] = self.calculate_map_at_k(recommendations_dict, actual_purchases_dict, k=k)
        
        # Precision and Recall
        precisions = []
        recalls = []
        
        for customer_id, actual_items in actual_purchases_dict.items():
            if customer_id in recommendations_dict:
                recommended_items = recommendations_dict[customer_id]
                prec = self.calculate_precision_at_k(recommended_items, actual_items, k=12)
                rec = self.calculate_recall_at_k(recommended_items, actual_items, k=12)
                precisions.append(prec)
                recalls.append(rec)
        
        results['precision_12'] = np.mean(precisions) if precisions else 0.0
        results['recall_12'] = np.mean(recalls) if recalls else 0.0
        
        # Coverage metrics
        all_items = set(articles_df['article_id'].unique())
        coverage_metrics = self.calculate_coverage_metrics(recommendations_dict, all_items)
        results.update(coverage_metrics)
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(recommendations_dict, articles_df)
        results.update(diversity_metrics)
        
        # Novelty metrics (if popularity scores available)
        if popularity_scores is not None:
            results['novelty'] = self.calculate_novelty_metrics(recommendations_dict, popularity_scores)
        
        print("\n=== EVALUATION SUMMARY ===")
        print(f"MAP@12: {results['map_12']:.6f}")
        print(f"Precision@12: {results['precision_12']:.6f}")
        print(f"Recall@12: {results['recall_12']:.6f}")
        print(f"Catalog Coverage: {results['catalog_coverage']:.4f}")
        print(f"Category Count: {results['category_count']}")
        
        return results
    
    def compare_models(self, model_results_dict):
        """Compare multiple models"""
        print("=== MODEL COMPARISON ===")
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, results in model_results_dict.items():
            row = {'model': model_name}
            row.update(results)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by MAP@12
        comparison_df = comparison_df.sort_values('map_12', ascending=False)
        
        print("\nModel Rankings (by MAP@12):")
        for i, row in comparison_df.iterrows():
            print(f"{row['model']:20s} | MAP@12: {row['map_12']:.6f} | "
                  f"Precision@12: {row['precision_12']:.6f} | "
                  f"Coverage: {row['catalog_coverage']:.4f}")
        
        return comparison_df

class ValidationFramework:
    """Framework for validating recommendation models"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.evaluator = RecommendationEvaluator(data_loader.transactions)
        
    def validate_model(self, model_class, model_params=None, test_days=7):
        """Validate a model using time-based split"""
        print("Starting model validation...")
        
        # Create validation split
        train_data, validation_data, test_data = self.evaluator.create_validation_split(test_days=test_days)
        
        # Prepare ground truth
        actual_purchases = self.evaluator.prepare_ground_truth(test_data)
        
        # Create model with training data
        train_loader = self.data_loader.__class__()
        train_loader.transactions = train_data
        train_loader.articles = self.data_loader.articles
        train_loader.customers = self.data_loader.customers
        train_loader.sample_submission = self.data_loader.sample_submission
        
        # Initialize and train model
        if model_params is None:
            model_params = {}
        
        model = model_class(train_loader, **model_params)
        
        # Generate recommendations for test customers
        test_customers = list(actual_purchases.keys())
        model.build_customer_history()
        
        if hasattr(model, 'set_fallback_items'):
            model.set_fallback_items()
        
        recommendations = model.generate_recommendations_for_all_customers(test_customers)
        
        # Evaluate
        results = self.evaluator.comprehensive_evaluation(
            recommendations, 
            actual_purchases,
            self.data_loader.articles
        )
        
        return results

if __name__ == "__main__":
    # Test evaluation
    from data_loader import HMDataLoader
    
    loader = HMDataLoader()
    loader.load_data()
    loader.filter_recent_data(months_back=6)
    
    evaluator = RecommendationEvaluator(loader.transactions)
    
    # Test validation split
    train_data, val_data, test_data = evaluator.create_validation_split()
    actual_purchases = evaluator.prepare_ground_truth(test_data)
    
    print(f"\nTest evaluation setup completed!")
    print(f"Test customers: {len(actual_purchases)}")
    
    # Create dummy recommendations for testing
    test_customers = list(actual_purchases.keys())[:100]
    dummy_recs = {}
    for customer in test_customers:
        dummy_recs[customer] = ['0706016001', '0706016002', '0372860001'] * 4  # 12 items
    
    # Test evaluation
    results = evaluator.comprehensive_evaluation(
        dummy_recs, 
        {k: v for k, v in actual_purchases.items() if k in test_customers},
        loader.articles
    )
    
    print("Evaluation test completed!") 