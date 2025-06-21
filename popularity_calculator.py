import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PopularityCalculator:
    """Calculate different types of popularity scores for articles"""
    
    def __init__(self, transactions_df, articles_df):
        self.transactions = transactions_df.copy()
        self.articles = articles_df.copy()
        self.popularity_scores = {}
        
    def calculate_global_popularity(self):
        """Calculate global popularity based on purchase count"""
        print("Calculating global popularity...")
        
        popularity = self.transactions.groupby('article_id').size().sort_values(ascending=False)
        
        # Normalize scores to 0-1 range
        max_count = popularity.max()
        normalized_popularity = popularity / max_count
        
        self.popularity_scores['global'] = normalized_popularity
        print(f"Global popularity calculated for {len(popularity)} articles")
        
        return normalized_popularity
    
    def calculate_time_weighted_popularity(self, decay_factor=0.1):
        """Calculate time-weighted popularity (recent purchases have higher weight)"""
        print("Calculating time-weighted popularity...")
        
        # Get max date and calculate days from max date
        max_date = self.transactions['t_dat'].max()
        self.transactions['days_from_max'] = (max_date - self.transactions['t_dat']).dt.days
        
        # Apply exponential decay: weight = exp(-decay_factor * days_from_max)
        self.transactions['time_weight'] = np.exp(-decay_factor * self.transactions['days_from_max'])
        
        # Calculate weighted popularity
        weighted_popularity = self.transactions.groupby('article_id')['time_weight'].sum().sort_values(ascending=False)
        
        # Normalize
        max_weight = weighted_popularity.max()
        normalized_weighted = weighted_popularity / max_weight
        
        self.popularity_scores['time_weighted'] = normalized_weighted
        print(f"Time-weighted popularity calculated for {len(weighted_popularity)} articles")
        
        return normalized_weighted
    
    def calculate_category_popularity(self):
        """Calculate popularity within each product category"""
        print("Calculating category-based popularity...")
        
        # Merge with articles to get categories
        trans_with_cat = self.transactions.merge(
            self.articles[['article_id', 'product_type_name', 'product_group_name']], 
            on='article_id', 
            how='left'
        )
        
        category_popularity = {}
        
        # Product type popularity
        for category in trans_with_cat['product_type_name'].unique():
            if pd.isna(category):
                continue
            cat_trans = trans_with_cat[trans_with_cat['product_type_name'] == category]
            cat_pop = cat_trans.groupby('article_id').size().sort_values(ascending=False)
            
            # Normalize within category
            if len(cat_pop) > 0:
                max_count = cat_pop.max()
                category_popularity[category] = cat_pop / max_count
        
        self.popularity_scores['category'] = category_popularity
        print(f"Category popularity calculated for {len(category_popularity)} categories")
        
        return category_popularity
    
    def calculate_price_tier_popularity(self, n_tiers=5):
        """Calculate popularity within price tiers"""
        print("Calculating price-tier popularity...")
        
        # Calculate price quantiles
        price_quantiles = self.transactions['price'].quantile(np.linspace(0, 1, n_tiers + 1))
        
        # Assign price tiers
        self.transactions['price_tier'] = pd.cut(
            self.transactions['price'], 
            bins=price_quantiles, 
            labels=[f'tier_{i}' for i in range(n_tiers)],
            include_lowest=True
        )
        
        price_tier_popularity = {}
        
        for tier in self.transactions['price_tier'].unique():
            if pd.isna(tier):
                continue
            tier_trans = self.transactions[self.transactions['price_tier'] == tier]
            tier_pop = tier_trans.groupby('article_id').size().sort_values(ascending=False)
            
            # Normalize within tier
            if len(tier_pop) > 0:
                max_count = tier_pop.max()
                price_tier_popularity[tier] = tier_pop / max_count
        
        self.popularity_scores['price_tier'] = price_tier_popularity
        print(f"Price-tier popularity calculated for {n_tiers} tiers")
        
        return price_tier_popularity
    
    def calculate_recent_trend_popularity(self, recent_days=30):
        """Calculate popularity based on recent trend"""
        print(f"Calculating recent trend popularity (last {recent_days} days)...")
        
        max_date = self.transactions['t_dat'].max()
        recent_cutoff = max_date - timedelta(days=recent_days)
        
        recent_trans = self.transactions[self.transactions['t_dat'] >= recent_cutoff]
        
        if len(recent_trans) == 0:
            print("No recent transactions found!")
            return pd.Series()
        
        recent_popularity = recent_trans.groupby('article_id').size().sort_values(ascending=False)
        
        # Normalize
        max_count = recent_popularity.max()
        normalized_recent = recent_popularity / max_count
        
        self.popularity_scores['recent_trend'] = normalized_recent
        print(f"Recent trend popularity calculated for {len(recent_popularity)} articles")
        
        return normalized_recent
    
    def calculate_customer_segment_popularity(self, customer_df):
        """Calculate popularity for different customer segments"""
        print("Calculating customer segment popularity...")
        
        # Merge transactions with customer data
        trans_with_customers = self.transactions.merge(
            customer_df[['customer_id', 'age', 'club_member_status']], 
            on='customer_id', 
            how='left'
        )
        
        segment_popularity = {}
        
        # Age-based segments
        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ['young', 'adult', 'middle', 'mature', 'senior']
        trans_with_customers['age_segment'] = pd.cut(
            trans_with_customers['age'], 
            bins=age_bins, 
            labels=age_labels
        )
        
        for segment in trans_with_customers['age_segment'].unique():
            if pd.isna(segment):
                continue
            seg_trans = trans_with_customers[trans_with_customers['age_segment'] == segment]
            seg_pop = seg_trans.groupby('article_id').size().sort_values(ascending=False)
            
            if len(seg_pop) > 0:
                max_count = seg_pop.max()
                segment_popularity[f'age_{segment}'] = seg_pop / max_count
        
        # Club member status segments
        for status in trans_with_customers['club_member_status'].unique():
            if pd.isna(status):
                continue
            status_trans = trans_with_customers[trans_with_customers['club_member_status'] == status]
            status_pop = status_trans.groupby('article_id').size().sort_values(ascending=False)
            
            if len(status_pop) > 0:
                max_count = status_pop.max()
                segment_popularity[f'club_{status}'] = status_pop / max_count
        
        self.popularity_scores['customer_segment'] = segment_popularity
        print(f"Customer segment popularity calculated for {len(segment_popularity)} segments")
        
        return segment_popularity
    
    def calculate_hybrid_popularity(self, weights=None):
        """Calculate hybrid popularity score combining multiple signals"""
        print("Calculating hybrid popularity...")
        
        if weights is None:
            weights = {
                'global': 0.4,
                'time_weighted': 0.3,
                'recent_trend': 0.2,
                'diversity_boost': 0.1
            }
        
        # Get all articles
        all_articles = set()
        for score_type, scores in self.popularity_scores.items():
            if isinstance(scores, pd.Series):
                all_articles.update(scores.index)
            elif isinstance(scores, dict):
                for subscore in scores.values():
                    all_articles.update(subscore.index)
        
        # Initialize hybrid scores
        hybrid_scores = pd.Series(0.0, index=list(all_articles))
        
        # Combine scores
        if 'global' in self.popularity_scores and 'global' in weights:
            global_scores = self.popularity_scores['global']
            hybrid_scores = hybrid_scores.add(global_scores * weights['global'], fill_value=0)
        
        if 'time_weighted' in self.popularity_scores and 'time_weighted' in weights:
            time_scores = self.popularity_scores['time_weighted']
            hybrid_scores = hybrid_scores.add(time_scores * weights['time_weighted'], fill_value=0)
        
        if 'recent_trend' in self.popularity_scores and 'recent_trend' in weights:
            recent_scores = self.popularity_scores['recent_trend']
            hybrid_scores = hybrid_scores.add(recent_scores * weights['recent_trend'], fill_value=0)
        
        # Add diversity boost (favor less popular items slightly)
        if 'diversity_boost' in weights and 'global' in self.popularity_scores:
            # Inverse of global popularity (boosting less popular items)
            global_scores = self.popularity_scores['global']
            diversity_boost = 1 - global_scores
            diversity_boost = diversity_boost / diversity_boost.max()  # Normalize
            hybrid_scores = hybrid_scores.add(diversity_boost * weights['diversity_boost'], fill_value=0)
        
        # Sort by final hybrid score
        hybrid_scores = hybrid_scores.sort_values(ascending=False)
        
        self.popularity_scores['hybrid'] = hybrid_scores
        print(f"Hybrid popularity calculated for {len(hybrid_scores)} articles")
        
        return hybrid_scores
    
    def get_top_n_items(self, score_type='global', n=12):
        """Get top N items for a specific popularity type"""
        if score_type not in self.popularity_scores:
            print(f"Score type '{score_type}' not found. Available: {list(self.popularity_scores.keys())}")
            return []
        
        scores = self.popularity_scores[score_type]
        if isinstance(scores, pd.Series):
            return scores.head(n).index.tolist()
        else:
            print(f"Score type '{score_type}' is not a simple series")
            return []
    
    def get_category_top_items(self, category, n=12):
        """Get top N items for a specific category"""
        if 'category' not in self.popularity_scores:
            print("Category popularity not calculated")
            return []
        
        category_scores = self.popularity_scores['category']
        if category in category_scores:
            return category_scores[category].head(n).index.tolist()
        else:
            print(f"Category '{category}' not found")
            return []
    
    def calculate_all_popularity_types(self, customer_df=None):
        """Calculate all types of popularity scores"""
        print("=== CALCULATING ALL POPULARITY TYPES ===")
        
        self.calculate_global_popularity()
        self.calculate_time_weighted_popularity()
        self.calculate_category_popularity()
        self.calculate_price_tier_popularity()
        self.calculate_recent_trend_popularity()
        
        if customer_df is not None:
            self.calculate_customer_segment_popularity(customer_df)
        
        self.calculate_hybrid_popularity()
        
        print("=== ALL POPULARITY CALCULATIONS COMPLETED ===")
        
        return self.popularity_scores

if __name__ == "__main__":
    # Test the popularity calculator
    from data_loader import HMDataLoader
    
    loader = HMDataLoader()
    loader.load_data()
    loader.filter_recent_data(months_back=3)  # Use less data for testing
    
    calc = PopularityCalculator(loader.transactions, loader.articles)
    calc.calculate_all_popularity_types(loader.customers)
    
    # Print top 10 items for each popularity type
    for score_type in ['global', 'time_weighted', 'recent_trend', 'hybrid']:
        top_items = calc.get_top_n_items(score_type, n=10)
        print(f"\nTop 10 {score_type} items: {top_items}") 