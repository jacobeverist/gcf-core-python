# ==============================================================================
# online_learning_scalar_sequence_anomalies.py
# ==============================================================================
from gnomics import BitArray, ScalarTransformer, SequenceLearner

# Scalar Transformer
transformer = ScalarTransformer(
        min_val=0.0, max_val=100.0, num_s=1024, num_as=40, num_t=2, seed=42)
transformer.set_value(50.0)
val = transformer.get_value() # 50.0
transformer.execute(False)
output = transformer.output()
assert isinstance(output, BitArray)
assert len(output) == 1024


learner = SequenceLearner(
        num_c=100,
        num_spc=8,
        num_dps=4,
        num_rpd=20,
        d_thresh=15,
        perm_thr=20,
        perm_inc=2,
        perm_dec=1,
        num_t=3,
        seed=42,
)

learner.init(num_input_bits=100)

input_pattern = BitArray(100)
input_pattern.set_acts([1, 5, 10, 15, 20])

learner.execute(input_pattern, learn_flag=True)
score = learner.get_anomaly_score()
output = learner.output()


# Create a simple sequence: A -> B -> C -> A
pattern_a = BitArray(50)
pattern_a.set_acts([1, 2, 3, 4, 5])

pattern_b = BitArray(50)
pattern_b.set_acts([10, 11, 12, 13, 14])

pattern_c = BitArray(50)
pattern_c.set_acts([20, 21, 22, 23, 24])

# Train the sequence multiple times
for _ in range(5):
    learner.execute(pattern_a, learn_flag=True)
    learner.execute(pattern_b, learn_flag=True)
    learner.execute(pattern_c, learn_flag=True)




values = [
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.2, 1.0, 1.0] # <-- anomaly is 0.2

scores = [0.0 for _ in range(len(values))]

# Setup blocks
st = ScalarTransformer(
    min_val=0.0, # minimum input value
    max_val=1.0, # maximum input value
    num_s=64,    # number of statelets
    num_as=8)    # number of active statelets

sl = SequenceLearner(
    num_spc=10,  # number of statelets per column
    num_dps=10,  # number of dendrites per statelet
    num_rpd=12,  # number of receptors per dendrite
    d_thresh=6,  # dendrite threshold
    perm_thr=20, # receptor permanence threshold
    perm_inc=2,  # receptor permanence increment
    perm_dec=1)  # receptor permanence decrement

# Connect blocks
sl.input.add_child(st.output, 0)

# Loop through the values
for i in range(len(values)):

    # Set scalar transformer value
    st.set_value(values[i])

    # Compute the scalar transformer
    st.feedforward()

    # Compute the sequence learner
    sl.feedforward(learn=True)

    # Get anomaly score
    scores[i] = sl.get_anomaly_score()

# Print output
print("val, scr")
for i in range(len(values)):
    print("%0.1f, %0.1f" % (values[i], scores[i]))
