# ==============================================================================
# Test connecting block outputs to inputs
# ==============================================================================
from gnomics import ScalarTransformer, DiscreteTransformer, PatternClassifier, ContextLearner, SequenceLearner

# Scalar Transformer
transformer = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
d_transformer = DiscreteTransformer(num_v=10, num_s=100, num_t=2, seed=42)

slearner = SequenceLearner(num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15, perm_thr=20, perm_inc=2,
                           perm_dec=1,
                           num_t=3, seed=42, )

# Connect encoder to sequence learner
print("Connecting transformer output to sequence learner input...")
slearner.input.add_child(transformer.output, 0)
print("Initializing sequence learner...")
slearner.init(num_input_bits=100)
print("Sequence learner initialized successfully!")

# For ContextLearner, we need both input AND context
clearner = ContextLearner(num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15, perm_thr=20, perm_inc=2, perm_dec=1,
                          num_t=3, seed=42, )
print("\nConnecting transformer output to context learner input...")
clearner.input.add_child(transformer.output, 0)
print("Connecting transformer output to context learner context...")
clearner.context.add_child(d_transformer.output, 0)
print("Initializing context learner...")
clearner.init(num_input_bits=100, num_context_bits=100)
print("Context learner initialized successfully!")


