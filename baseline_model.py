import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from data_loader import HMDataLoader
from popularity_calculator import PopularityCalculator

class PopularityBaselineModel:
    """Popularity-based baseline recommendation model for H&M"""
    
    def __init__(self, data_loader, popularity_calculator):
        self.data_loader = data_loader
        self.popularity_calc = popularity_calculator
        self.customer_history = {}
        self.model_type = 'hybrid'  # Default to hybrid popularity
        self.fallback_items = []
        
    def build_customer_history(self):
        """Build customer purchase history for personalization"""
        print("Building customer purchase history...")
        
        customer_purchases = self.data_loader.transactions.groupby('customer_id')['article_id'].apply(list)
        self.customer_history = customer_purchases.to_dict()
        
        print(f"Built history for {len(self.customer_history)} customers")
        
    def set_fallback_items(self, model_type='global', n_items=12):
        """Set fallback items for cold start customers"""
        self.fallback_items = self.popularity_calc.get_top_n_items(model_type, n_items)
        print(f"Set {len(self.fallback_items)} fallback items using {model_type} popularity")
        
    def get_customer_preferences(self, customer_id):
        """Analyze customer preferences based on purchase history"""
        if customer_id not in self.customer_history:
            return None
        
        purchased_items = self.customer_history[customer_id]
        
        # Get categories of purchased items
        customer_articles = self.data_loader.articles[
            self.data_loader.articles['article_id'].isin(purchased_items)
        ]
        
        preferences = {
            'categories': customer_articles['product_type_name'].value_counts().to_dict(),
            'groups': customer_articles['product_group_name'].value_counts().to_dict(),
            'total_purchases': len(purchased_items),
            'unique_items': len(set(purchased_items))
        }
        
        return preferences
    
    def get_personalized_recommendations(self, customer_id, n_recommendations=12):
        """Get personalized recommendations for a customer"""
        recommendations = []
        
        # Get customer preferences
        preferences = self.get_customer_preferences(customer_id)
        
        if preferences is None or preferences['total_purchases'] < 2:
            # Cold start: return fallback items
            return self.fallback_items[:n_recommendations]
        
        # Get customer's purchased items to avoid re-recommending
        purchased_items = set(self.customer_history.get(customer_id, []))
        
        # Strategy 1: Get recommendations from preferred categories
        category_recs = []
        if 'category' in self.popularity_calc.popularity_scores:
            top_categories = list(preferences['categories'].keys())[:3]  # Top 3 categories
            
            for category in top_categories:
                category_items = self.popularity_calc.get_category_top_items(category, n=6)
                # Filter out already purchased items
                new_items = [item for item in category_items if item not in purchased_items]
                category_recs.extend(new_items[:4])  # Max 4 items per category
        
        # Strategy 2: Get hybrid popularity recommendations
        hybrid_recs = []
        if self.model_type in self.popularity_calc.popularity_scores:
            all_hybrid_items = self.popularity_calc.get_top_n_items(self.model_type, n=50)
            # Filter out purchased items
            new_hybrid_items = [item for item in all_hybrid_items if item not in purchased_items]
            hybrid_recs = new_hybrid_items[:8]
        
        # Combine recommendations
        recommendations.extend(category_recs[:6])  # Max 6 from categories
        
        # Fill remaining slots with hybrid recommendations
        for item in hybrid_recs:
            if item not in recommendations and len(recommendations) < n_recommendations:
                recommendations.append(item)
        
        # Fill any remaining slots with fallback items
        for item in self.fallback_items:
            if item not in recommendations and item not in purchased_items:
                recommendations.append(item)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations[:n_recommendations]
    
    def generate_recommendations_for_all_customers(self, customer_list=None):
        """Generate recommendations for all customers in the submission file"""
        print("Generating recommendations for all customers...")
        
        if customer_list is None:
            customer_list = self.data_loader.get_submission_customers()
        
        recommendations = {}
        total_customers = len(customer_list)
        
        for i, customer_id in enumerate(customer_list):
            if i % 10000 == 0:
                print(f"Progress: {i}/{total_customers} ({i/total_customers*100:.1f}%)")
            
            customer_recs = self.get_personalized_recommendations(customer_id)
            recommendations[customer_id] = customer_recs
        
        print(f"Generated recommendations for {len(recommendations)} customers")
        return recommendations
    
    def create_submission_file(self, output_filename='baseline_submission.csv'):
        """Create submission file in the required format"""
        print("Creating submission file...")
        
        # Build customer history
        self.build_customer_history()
        
        # Set fallback items
        self.set_fallback_items(model_type='hybrid', n_items=12)
        
        # Generate recommendations
        all_recommendations = self.generate_recommendations_for_all_customers()
        
        # Create submission dataframe
        submission_data = []
        for customer_id, recs in all_recommendations.items():
            # Convert recommendations to space-separated string
            prediction = ' '.join(recs)
            submission_data.append({
                'customer_id': customer_id,
                'prediction': prediction
            })
        
        submission_df = pd.DataFrame(submission_data)
        
        # Save to file
        submission_df.to_csv(output_filename, index=False)
        print(f"Submission saved to {output_filename}")
        print(f"Submission shape: {submission_df.shape}")
        
        # Verify format
        print("\nSample predictions:")
        print(submission_df.head(3))
        
        return submission_df
    
    def analyze_recommendations(self, recommendations_dict, sample_size=1000):
        """Analyze the quality of recommendations"""
        print("Analyzing recommendation quality...")
        
        # Sample customers for analysis
        customer_sample = list(recommendations_dict.keys())[:sample_size]
        
        analysis = {
            'total_customers': len(recommendations_dict),
            'avg_recommendations_per_customer': 0,
            'unique_items_recommended': set(),
            'category_diversity': defaultdict(int),
            'cold_start_customers': 0,
            'personalized_customers': 0
        }
        
        for customer_id in customer_sample:
            recs = recommendations_dict[customer_id]
            analysis['avg_recommendations_per_customer'] += len(recs)
            analysis['unique_items_recommended'].update(recs)
            
            # Check if customer is cold start
            if customer_id in self.customer_history:
                analysis['personalized_customers'] += 1
            else:
                analysis['cold_start_customers'] += 1
            
            # Analyze category diversity
            rec_articles = self.data_loader.articles[
                self.data_loader.articles['article_id'].isin(recs)
            ]
            for category in rec_articles['product_type_name']:
                if pd.notna(category):
                    analysis['category_diversity'][category] += 1
        
        analysis['avg_recommendations_per_customer'] /= len(customer_sample)
        analysis['unique_items_recommended'] = len(analysis['unique_items_recommended'])
        
        print("\n=== RECOMMENDATION ANALYSIS ===")
        print(f"Total customers: {analysis['total_customers']:,}")
        print(f"Average recommendations per customer: {analysis['avg_recommendations_per_customer']:.1f}")
        print(f"Unique items recommended: {analysis['unique_items_recommended']:,}")
        print(f"Personalized customers: {analysis['personalized_customers']:,}")
        print(f"Cold start customers: {analysis['cold_start_customers']:,}")
        
        print("\nTop 10 categories in recommendations:")
        sorted_categories = sorted(analysis['category_diversity'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            print(f"  {category}: {count}")
        
        return analysis

class AdvancedPopularityModel(PopularityBaselineModel):
    """Advanced popularity model with additional features"""
    
    def __init__(self, data_loader, popularity_calculator):
        super().__init__(data_loader, popularity_calculator)
        self.customer_segments = {}
        
    def segment_customers(self):
        """Segment customers based on their behavior"""
        print("Segmenting customers...")
        
        # Merge customer data
        customer_purchases = self.data_loader.transactions.groupby('customer_id').agg({
            'article_id': 'count',
            'price': ['mean', 'sum'],
            't_dat': ['min', 'max']
        }).reset_index()
        
        customer_purchases.columns = ['customer_id', 'total_purchases', 'avg_price', 
                                    'total_spent', 'first_purchase', 'last_purchase']
        
        # Add customer metadata
        customer_data = self.data_loader.customers[['customer_id', 'age', 'club_member_status']]
        customer_features = customer_purchases.merge(customer_data, on='customer_id', how='left')
        
        # Simple segmentation rules
        segments = {}
        for _, row in customer_features.iterrows():
            customer_id = row['customer_id']
            
            # Segment based on purchase frequency and value
            if row['total_purchases'] >= 10 and row['total_spent'] >= 100:
                segment = 'high_value'
            elif row['total_purchases'] >= 5:
                segment = 'regular'
            elif row['total_purchases'] >= 2:
                segment = 'occasional'
            else:
                segment = 'new'
            
            segments[customer_id] = segment
        
        self.customer_segments = segments
        print(f"Segmented {len(segments)} customers")
        
        # Print segment distribution
        segment_counts = defaultdict(int)
        for segment in segments.values():
            segment_counts[segment] += 1
        
        print("Segment distribution:")
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count:,}")
    
    def get_segment_based_recommendations(self, customer_id, n_recommendations=12):
        """Get recommendations based on customer segment"""
        if customer_id not in self.customer_segments:
            return self.get_personalized_recommendations(customer_id, n_recommendations)
        
        segment = self.customer_segments[customer_id]
        purchased_items = set(self.customer_history.get(customer_id, []))
        
        # Different strategies for different segments
        if segment == 'high_value':
            # Mix of trending and diverse items
            trending_items = self.popularity_calc.get_top_n_items('recent_trend', n=8)
            hybrid_items = self.popularity_calc.get_top_n_items('hybrid', n=8)
            candidate_items = trending_items + hybrid_items
        elif segment == 'regular':
            # Balanced approach
            candidate_items = self.popularity_calc.get_top_n_items('hybrid', n=20)
        elif segment == 'occasional':
            # Popular and safe choices
            candidate_items = self.popularity_calc.get_top_n_items('global', n=20)
        else:  # new customers
            # Most popular items
            candidate_items = self.fallback_items
        
        # Filter out purchased items and return top N
        recommendations = []
        for item in candidate_items:
            if item not in purchased_items and item not in recommendations:
                recommendations.append(item)
                if len(recommendations) >= n_recommendations:
                    break
        
        # Fill with fallback if needed
        for item in self.fallback_items:
            if item not in recommendations and item not in purchased_items:
                recommendations.append(item)
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations[:n_recommendations]

if __name__ == "__main__":
    # Test the baseline model
    print("Testing Popularity Baseline Model...")
    
    # Load data
    loader = HMDataLoader()
    loader.load_data()
    loader.filter_recent_data(months_back=6)
    
    # Calculate popularity scores
    calc = PopularityCalculator(loader.transactions, loader.articles)
    calc.calculate_all_popularity_types(loader.customers)
    
    # Create and test basic model
    model = PopularityBaselineModel(loader, calc)
    
    # Test with a small subset first
    print("\nTesting with small subset...")
    test_customers = loader.get_submission_customers()[:100]
    
    model.build_customer_history()
    model.set_fallback_items()
    
    test_recommendations = model.generate_recommendations_for_all_customers(test_customers)
    model.analyze_recommendations(test_recommendations)
    
    print("\nBaseline model test completed!") 