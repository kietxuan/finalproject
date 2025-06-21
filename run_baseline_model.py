#!/usr/bin/env python3
"""
H&M Personalized Fashion Recommendations - Popularity-Based Baseline Model

This script implements a comprehensive popularity-based recommendation system
for the H&M Kaggle competition using multiple popularity signals and strategies.
"""

import time
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import HMDataLoader
from popularity_calculator import PopularityCalculator
from baseline_model import PopularityBaselineModel, AdvancedPopularityModel
from evaluation import RecommendationEvaluator, ValidationFramework

def main():
    parser = argparse.ArgumentParser(description='Run H&M Popularity-Based Baseline Model')
    parser.add_argument('--data_path', default='./', help='Path to data files')
    parser.add_argument('--months_back', type=int, default=6, help='Months of data to use')
    parser.add_argument('--model_type', choices=['basic', 'advanced'], default='advanced', 
                       help='Type of popularity model to use')
    parser.add_argument('--output_file', default='baseline_submission.csv', 
                       help='Output submission filename')
    parser.add_argument('--validate', action='store_true', help='Run validation before final submission')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with subset of data')
    
    args = parser.parse_args()
    
    print("="*60)
    print("H&M POPULARITY-BASED BASELINE MODEL")
    print("="*60)
    print(f"Data path: {args.data_path}")
    print(f"Months back: {args.months_back}")
    print(f"Model type: {args.model_type}")
    print(f"Output file: {args.output_file}")
    print(f"Validation: {args.validate}")
    print(f"Test mode: {args.test_mode}")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\nSTEP 1: Loading and preprocessing data...")
    start_time = time.time()
    
    loader = HMDataLoader(data_path=args.data_path)
    loader.load_data()
    
    if args.test_mode:
        print("Running in test mode - using 3 months of data")
        loader.filter_recent_data(months_back=3)
    else:
        loader.filter_recent_data(months_back=args.months_back)
    
    loader.analyze_time_patterns()
    loader.get_category_info()
    loader.create_mappings()
    
    print(f"Data loading completed in {time.time() - start_time:.1f} seconds")
    
    # Step 2: Calculate popularity scores
    print("\nSTEP 2: Calculating popularity scores...")
    start_time = time.time()
    
    calc = PopularityCalculator(loader.transactions, loader.articles)
    calc.calculate_all_popularity_types(loader.customers)
    
    print(f"Popularity calculation completed in {time.time() - start_time:.1f} seconds")
    
    # Step 3: Model validation (if requested)
    if args.validate:
        print("\nSTEP 3: Model validation...")
        start_time = time.time()
        
        validation_framework = ValidationFramework(loader)
        
        # Test both model types
        if args.model_type == 'basic':
            # Validate basic model
            def create_basic_model(train_loader):
                train_calc = PopularityCalculator(train_loader.transactions, train_loader.articles)
                train_calc.calculate_all_popularity_types(train_loader.customers)
                return PopularityBaselineModel(train_loader, train_calc)
            
            basic_results = validation_framework.validate_model(
                lambda train_loader: create_basic_model(train_loader)
            )
            
            print(f"Basic model validation: MAP@12 = {basic_results['map_12']:.6f}")
            
        else:
            # Validate advanced model  
            def create_advanced_model(train_loader):
                train_calc = PopularityCalculator(train_loader.transactions, train_loader.articles)
                train_calc.calculate_all_popularity_types(train_loader.customers)
                model = AdvancedPopularityModel(train_loader, train_calc)
                model.segment_customers()
                return model
            
            advanced_results = validation_framework.validate_model(
                lambda train_loader: create_advanced_model(train_loader)
            )
            
            print(f"Advanced model validation: MAP@12 = {advanced_results['map_12']:.6f}")
        
        print(f"Validation completed in {time.time() - start_time:.1f} seconds")
    
    # Step 4: Create final model and generate submission
    print("\nSTEP 4: Creating final model and generating submission...")
    start_time = time.time()
    
    if args.model_type == 'advanced':
        model = AdvancedPopularityModel(loader, calc)
        model.segment_customers()
        print("Using Advanced Popularity Model with customer segmentation")
    else:
        model = PopularityBaselineModel(loader, calc)
        print("Using Basic Popularity Model")
    
    # Generate submission file
    if args.test_mode:
        # Test with subset of customers
        test_customers = loader.get_submission_customers()[:1000]
        print(f"Test mode: generating recommendations for {len(test_customers)} customers")
        
        model.build_customer_history()
        model.set_fallback_items(model_type='hybrid', n_items=12)
        
        recommendations = model.generate_recommendations_for_all_customers(test_customers)
        
        # Create mini submission file
        submission_data = []
        for customer_id, recs in recommendations.items():
            prediction = ' '.join(recs)
            submission_data.append({
                'customer_id': customer_id,
                'prediction': prediction
            })
        
        import pandas as pd
        submission_df = pd.DataFrame(submission_data)
        test_filename = f"test_{args.output_file}"
        submission_df.to_csv(test_filename, index=False)
        
        print(f"Test submission saved to {test_filename}")
        print(f"Test submission shape: {submission_df.shape}")
        
        # Analyze recommendations
        model.analyze_recommendations(recommendations)
        
    else:
        # Full submission
        submission_df = model.create_submission_file(args.output_file)
        
        print(f"Full submission created: {args.output_file}")
        print(f"Submission shape: {submission_df.shape}")
    
    print(f"Submission generation completed in {time.time() - start_time:.1f} seconds")
    
    # Step 5: Final summary
    print("\nSTEP 5: Final summary...")
    
    # Display top popular items
    print("\nTop 12 Global Popular Items:")
    top_global = calc.get_top_n_items('global', n=12)
    for i, item in enumerate(top_global, 1):
        print(f"  {i:2d}. {item}")
    
    print("\nTop 12 Hybrid Popular Items:")
    top_hybrid = calc.get_top_n_items('hybrid', n=12)
    for i, item in enumerate(top_hybrid, 1):
        print(f"  {i:2d}. {item}")
    
    # Display model statistics
    if hasattr(model, 'customer_segments'):
        print(f"\nCustomer segments: {len(model.customer_segments)}")
    
    print(f"\nCustomer history built for: {len(model.customer_history)} customers")
    print(f"Fallback items: {len(model.fallback_items)}")
    
    print("\n" + "="*60)
    print("BASELINE MODEL COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    if args.validate:
        print("\nVALIDATION RESULTS:")
        if args.model_type == 'basic':
            print(f"- Basic model MAP@12: {basic_results['map_12']:.6f}")
        else:
            print(f"- Advanced model MAP@12: {advanced_results['map_12']:.6f}")

def run_quick_analysis():
    """Run a quick analysis of the data and model performance"""
    print("RUNNING QUICK ANALYSIS...")
    
    # Load small subset
    loader = HMDataLoader()
    loader.load_data()
    loader.filter_recent_data(months_back=3)
    
    # Quick popularity calculation
    calc = PopularityCalculator(loader.transactions, loader.articles)
    calc.calculate_global_popularity()
    calc.calculate_time_weighted_popularity()
    calc.calculate_hybrid_popularity()
    
    # Display insights
    print("\nQUICK INSIGHTS:")
    print(f"Total transactions: {len(loader.transactions):,}")
    print(f"Unique customers: {loader.transactions['customer_id'].nunique():,}")
    print(f"Unique articles: {loader.transactions['article_id'].nunique():,}")
    
    # Top categories
    trans_with_cat = loader.transactions.merge(
        loader.articles[['article_id', 'product_type_name']], 
        on='article_id', how='left'
    )
    top_categories = trans_with_cat['product_type_name'].value_counts().head(5)
    print("\nTop 5 categories:")
    for cat, count in top_categories.items():
        print(f"  {cat}: {count:,}")
    
    # Sample recommendations
    model = PopularityBaselineModel(loader, calc)
    model.build_customer_history()
    model.set_fallback_items()
    
    sample_customers = loader.get_submission_customers()[:5]
    print(f"\nSample recommendations for 5 customers:")
    for customer in sample_customers:
        recs = model.get_personalized_recommendations(customer)
        print(f"  {customer}: {recs[:3]}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 2 and sys.argv[1] == '--quick':
        run_quick_analysis()
    else:
        main() 