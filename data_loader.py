import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HMDataLoader:
    """Data loader and preprocessor for H&M recommendation system"""
    
    def __init__(self, data_path="./"):
        self.data_path = data_path
        self.transactions = None
        self.articles = None
        self.customers = None
        self.sample_submission = None
        
    def load_data(self):
        """Load all necessary data files"""
        print("Loading data files...")
        
        # Load transactions with proper data types
        print("Loading transactions_train.csv...")
        self.transactions = pd.read_csv(
            f"{self.data_path}transactions_train.csv",
            dtype={'article_id': str, 'customer_id': str},
            parse_dates=['t_dat']
        )
        
        # Load articles
        print("Loading articles.csv...")
        self.articles = pd.read_csv(
            f"{self.data_path}articles.csv",
            dtype={'article_id': str}
        )
        
        # Load customers
        print("Loading customers.csv...")
        self.customers = pd.read_csv(
            f"{self.data_path}customers.csv",
            dtype={'customer_id': str}
        )
        
        # Load sample submission
        print("Loading sample_submission.csv...")
        self.sample_submission = pd.read_csv(
            f"{self.data_path}sample_submission.csv",
            dtype={'customer_id': str}
        )
        
        print("Data loaded successfully!")
        self._print_data_info()
        
    def _print_data_info(self):
        """Print basic information about loaded data"""
        print("\n=== DATA OVERVIEW ===")
        print(f"Transactions: {len(self.transactions):,} rows")
        print(f"Unique customers: {self.transactions['customer_id'].nunique():,}")
        print(f"Unique articles: {self.transactions['article_id'].nunique():,}")
        print(f"Date range: {self.transactions['t_dat'].min()} to {self.transactions['t_dat'].max()}")
        print(f"Articles metadata: {len(self.articles):,} rows")
        print(f"Customers metadata: {len(self.customers):,} rows")
        print(f"Customers for prediction: {len(self.sample_submission):,}")
        
    def filter_recent_data(self, months_back=6):
        """Filter transactions to keep only recent data"""
        if self.transactions is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        max_date = self.transactions['t_dat'].max()
        cutoff_date = max_date - timedelta(days=30 * months_back)
        
        print(f"\nFiltering data from {cutoff_date} to {max_date}")
        original_size = len(self.transactions)
        
        self.transactions = self.transactions[self.transactions['t_dat'] >= cutoff_date]
        
        print(f"Transactions reduced from {original_size:,} to {len(self.transactions):,}")
        print(f"Unique customers: {self.transactions['customer_id'].nunique():,}")
        print(f"Unique articles: {self.transactions['article_id'].nunique():,}")
        
    def get_active_customers(self, min_purchases=1):
        """Get customers with minimum number of purchases"""
        customer_purchases = self.transactions.groupby('customer_id').size()
        active_customers = customer_purchases[customer_purchases >= min_purchases].index
        
        print(f"\nActive customers (min {min_purchases} purchases): {len(active_customers):,}")
        return active_customers.tolist()
        
    def get_popular_articles(self, top_n=1000):
        """Get most popular articles by purchase count"""
        article_popularity = self.transactions.groupby('article_id').size().sort_values(ascending=False)
        popular_articles = article_popularity.head(top_n)
        
        print(f"\nTop {top_n} popular articles account for {popular_articles.sum():,} purchases")
        return popular_articles
        
    def create_mappings(self):
        """Create customer and article ID mappings"""
        # Get all unique customers and articles
        all_customers = self.customers['customer_id'].unique()
        all_articles = self.articles['article_id'].unique()
        
        # Create mappings
        self.customer_to_idx = {cid: idx for idx, cid in enumerate(all_customers)}
        self.idx_to_customer = {idx: cid for cid, idx in self.customer_to_idx.items()}
        
        self.article_to_idx = {aid: idx for idx, aid in enumerate(all_articles)}
        self.idx_to_article = {idx: aid for aid, idx in self.article_to_idx.items()}
        
        print(f"\nMappings created:")
        print(f"Customer mapping: {len(self.customer_to_idx):,} entries")
        print(f"Article mapping: {len(self.article_to_idx):,} entries")
        
    def get_submission_customers(self):
        """Get list of customers that need predictions"""
        return self.sample_submission['customer_id'].tolist()
        
    def analyze_time_patterns(self):
        """Analyze time patterns in the data"""
        print("\n=== TIME PATTERN ANALYSIS ===")
        
        # Weekly patterns
        self.transactions['day_of_week'] = self.transactions['t_dat'].dt.day_name()
        weekly_sales = self.transactions.groupby('day_of_week').size()
        print("Sales by day of week:")
        print(weekly_sales.sort_values(ascending=False))
        
        # Monthly patterns
        self.transactions['month'] = self.transactions['t_dat'].dt.month
        monthly_sales = self.transactions.groupby('month').size()
        print("\nSales by month:")
        print(monthly_sales.sort_values(ascending=False))
        
        # Recent trend
        last_30_days = self.transactions[
            self.transactions['t_dat'] >= (self.transactions['t_dat'].max() - timedelta(days=30))
        ]
        daily_sales = last_30_days.groupby('t_dat').size()
        print(f"\nDaily sales in last 30 days: avg={daily_sales.mean():.0f}, std={daily_sales.std():.0f}")
        
    def get_category_info(self):
        """Analyze product categories"""
        print("\n=== CATEGORY ANALYSIS ===")
        
        # Merge with articles to get category info
        trans_with_categories = self.transactions.merge(
            self.articles[['article_id', 'product_type_name', 'product_group_name']], 
            on='article_id', 
            how='left'
        )
        
        # Top product types
        product_type_sales = trans_with_categories.groupby('product_type_name').size().sort_values(ascending=False)
        print("Top 10 product types:")
        print(product_type_sales.head(10))
        
        # Top product groups
        product_group_sales = trans_with_categories.groupby('product_group_name').size().sort_values(ascending=False)
        print("\nTop 10 product groups:")
        print(product_group_sales.head(10))
        
        return trans_with_categories

if __name__ == "__main__":
    # Test the data loader
    loader = HMDataLoader()
    loader.load_data()
    loader.filter_recent_data(months_back=6)
    loader.analyze_time_patterns()
    loader.get_category_info()
    loader.create_mappings() 