# H&M Personalized Fashion Recommendations - Popularity-Based Baseline

Đây là implementation của một **popularity-based baseline model** cho cuộc thi [H&M Personalized Fashion Recommendations](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations) trên Kaggle.

## 🎯 Mục tiêu

Tạo một baseline model mạnh mẽ sử dụng multiple popularity signals để recommend 12 sản phẩm cho mỗi khách hàng, được đánh giá bằng MAP@12.

## 📁 Cấu trúc Project

```
h&m-baseline/
├── data_loader.py              # Load và preprocess data
├── popularity_calculator.py    # Tính toán popularity scores
├── baseline_model.py          # Main recommendation models  
├── evaluation.py              # Evaluation metrics và validation
├── run_baseline_model.py      # Main execution script
├── README_baseline.md         # Documentation
└── test_baseline_submission.csv  # Sample output
```

## 🚀 Cách sử dụng

### 1. Quick Analysis
```bash
python run_baseline_model.py --quick
```

### 2. Test Mode (1000 customers)
```bash
python run_baseline_model.py --test_mode --model_type advanced
```

### 3. Full Production Run
```bash
python run_baseline_model.py --model_type advanced --output_file my_submission.csv
```

### 4. Validation Mode
```bash
python run_baseline_model.py --test_mode --validate --model_type advanced
```

## 🧩 Các Components

### 1. Data Loader (`data_loader.py`)
- Load transactions, articles, customers data
- Filter recent data (default: 6 months)
- Analyze time patterns và categories
- Create mappings và preprocessing

### 2. Popularity Calculator (`popularity_calculator.py`)
- **Global Popularity**: Based on purchase count
- **Time-weighted Popularity**: Recent purchases có weight cao hơn
- **Category Popularity**: Popular items trong từng category
- **Price-tier Popularity**: Popular items trong từng price range
- **Recent Trend**: Items trending trong 30 ngày gần nhất
- **Customer Segment**: Popularity cho từng age/club member segment
- **Hybrid Popularity**: Combination of multiple signals

### 3. Baseline Models (`baseline_model.py`)

#### PopularityBaselineModel
- Cold start: Global fallback items
- Personalized: Category-based + Hybrid recommendations
- Avoid re-recommending purchased items

#### AdvancedPopularityModel
- Customer segmentation (new, occasional, regular, high_value)
- Segment-specific recommendation strategies
- Enhanced personalization logic

### 4. Evaluation (`evaluation.py`)
- MAP@12 calculation (main metric)
- Time-based validation split
- Coverage, diversity, novelty metrics
- Model comparison framework

## 📊 Results từ Test Run

```
=== RECOMMENDATION ANALYSIS ===
Total customers: 1,000
Average recommendations per customer: 12.0
Unique items recommended: 203
Personalized customers: 328
Cold start customers: 672

Top 10 categories in recommendations:
  Trousers: 4772
  Sweater: 2647
  Leggings/Tights: 1762
  Dress: 949
  Shirt: 868
```

## 🔧 Tuning Parameters

### Popularity Calculator
```python
# Time decay factor for time-weighted popularity
decay_factor = 0.1  # Higher = more recent bias

# Recent trend window
recent_days = 30

# Number of price tiers
n_tiers = 5

# Hybrid weights
weights = {
    'global': 0.4,
    'time_weighted': 0.3, 
    'recent_trend': 0.2,
    'diversity_boost': 0.1
}
```

### Model Parameters
```python
# Customer segmentation thresholds
high_value_threshold = (10, 100)  # (purchases, spending)
regular_threshold = 5
occasional_threshold = 2

# Recommendation mix
category_items = 6  # Max from preferred categories
hybrid_items = 6   # Max from hybrid popularity
```

## 📈 Actual Performance Results

- **Achieved MAP@12**: 0.003363 (validated baseline)
- **Processing time**: 615.3 seconds for full pipeline
- **Memory usage**: Handles 31M+ transactions
- **Cold start coverage**: 100% customers get recommendations

## 🛠 Advanced Usage

### Custom Popularity Weights
```python
from popularity_calculator import PopularityCalculator

calc = PopularityCalculator(transactions, articles)
calc.calculate_all_popularity_types(customers)

# Custom hybrid weights
custom_weights = {
    'global': 0.5,
    'time_weighted': 0.3,
    'recent_trend': 0.2
}
hybrid_scores = calc.calculate_hybrid_popularity(custom_weights)
```

### Custom Model
```python
from baseline_model import AdvancedPopularityModel

class CustomModel(AdvancedPopularityModel):
    def get_personalized_recommendations(self, customer_id, n_recommendations=12):
        # Your custom logic here
        return super().get_personalized_recommendations(customer_id, n_recommendations)
```

### Validation Framework
```python
from evaluation import ValidationFramework

framework = ValidationFramework(data_loader)
results = framework.validate_model(YourModelClass, test_days=7)
print(f"MAP@12: {results['map_12']:.6f}")
```

## 📝 Command Line Options

```bash
python run_baseline_model.py [OPTIONS]

Options:
  --data_path PATH           Path to data files (default: ./)
  --months_back INT          Months of data to use (default: 6)
  --model_type {basic,advanced}  Type of model (default: advanced)
  --output_file FILE         Output filename (default: baseline_submission.csv)
  --validate                 Run validation before submission
  --test_mode               Run with subset of data for testing
  --help                    Show help message
```

## 📊 Validation Results

Time-based validation with 75,481 customers:
- **MAP@12**: 0.003363
- **MAP@10**: 0.003209
- **MAP@5**: 0.002520
- **Precision@12**: 0.003092
- **Recall@12**: 0.012808
- **Catalog Coverage**: 0.0050 (530 unique items)
- **Category Count**: 98 categories

## 🔍 Debugging

### Common Issues

1. **Memory Error**: Reduce `months_back` parameter
2. **Slow Performance**: Use `--test_mode` for development
3. **Low Scores**: Try different popularity weights
4. **Missing Data**: Check file paths và data format

### Logging
All steps are logged với progress indicators:
```
STEP 1: Loading and preprocessing data...
STEP 2: Calculating popularity scores...
STEP 3: Model validation...
STEP 4: Creating final model and generating submission...
STEP 5: Final summary...
```

## 📊 Output Format

Submission file format (required by Kaggle):
```csv
customer_id,prediction
00000dbacae5e...,0751471001 0706016001 0448509014 0918292001 0915529003 0751471043 0915526001 0850917001 0924243001 0909370001 0866731001 0714790020
```

Each prediction contains exactly 12 article IDs separated by spaces.

---

**Ngày tạo**: 2025  
**Phiên bản**: 1.0  
**MAP@12 đã xác thực**: 0.003363 