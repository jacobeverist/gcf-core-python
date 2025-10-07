# ==============================================================================
# Test PatternPooler and PatternClassifier with block connections
# ==============================================================================
from gnomics import ScalarTransformer, PatternPooler, PatternClassifier

# Create a scalar encoder
encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

# Test PatternPooler
print("Testing PatternPooler with block connections...")
pooler = PatternPooler(
    num_s=50,
    num_as=5,
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.8,
    pct_conn=0.5,
    pct_learn=0.3,
    num_t=2,
    seed=42
)

# Connect encoder output to pooler input
pooler.input.add_child(encoder.output, 0)
pooler.init(num_i=100)
print("PatternPooler initialized successfully!")

# Test PatternClassifier
print("\nTesting PatternClassifier with block connections...")
classifier = PatternClassifier(
    num_l=3,
    num_s=60,
    num_as=6,
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.8,
    pct_conn=0.5,
    pct_learn=0.3,
    num_t=2,
    seed=42
)

# Connect encoder output to classifier input
classifier.input.add_child(encoder.output, 0)
classifier.init(num_i=100)
print("PatternClassifier initialized successfully!")

# Test chaining: encoder -> pooler -> classifier
print("\nTesting block chaining: encoder -> pooler -> classifier...")
classifier2 = PatternClassifier(
    num_l=3,
    num_s=60,
    num_as=6,
    perm_thr=20,
    perm_inc=2,
    perm_dec=1,
    pct_pool=0.8,
    pct_conn=0.5,
    pct_learn=0.3,
    num_t=2,
    seed=43
)

# Connect pooler output to classifier input
classifier2.input.add_child(pooler.output, 0)
classifier2.init(num_i=50)
print("Chained blocks initialized successfully!")
print("\nAll tests passed!")
