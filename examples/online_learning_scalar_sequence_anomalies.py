# ==============================================================================
# online_learning_scalar_sequence_anomalies.py
# ==============================================================================
from gnomics import BitArray, ScalarTransformer, SequenceLearner

# Scalar Transformer
transformer = ScalarTransformer(
        min_val=0.0, # minimum input value
        max_val=1.0, # maximum input value
        num_s=64,    # number of statelets
        num_as=8,    # number of active statelets
        num_t=2,
        seed=42)


# Sequence Learner
learner = SequenceLearner(
        num_c=64,
        num_spc=10,  # number of statelets per column
        num_dps=10,  # number of dendrites per statelet
        num_rpd=12,  # number of receptors per dendrite
        d_thresh=6,  # dendrite threshold
        perm_thr=20, # receptor permanence threshold
        perm_inc=2,  # receptor permanence increment
        perm_dec=1,  # receptor permanence decrement
        num_t=3,
        seed=42)

# Connect encoder to pooler
learner.input.add_child(transformer.output, 0)
learner.init(num_input_bits=64)

values = [
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        0.0, 0.2, 0.4, 0.2, 0.8, 1.0,] # <-- anomaly is 0.2

scores = []
patterns = []

# Loop through the values
for i in range(len(values)):

    # Set scalar transformer value
    transformer.set_value(values[i])

    # Compute the scalar transformer
    transformer.execute(False)

    encoder_pattern = transformer.output.state()

    # Compute the sequence learner (input is pulled automatically from connected transformer)
    learner.execute(True)

    # view the computational state of the sequence learner
    learner_pattern = learner.output.state()

    if values[i] == 0.0:
        # print("encoder:", encoder_pattern)
        print("learner:", learner_pattern)

    # Get anomaly score
    score = learner.get_anomaly_score()
    print(values[i], score)

    scores.append(score)
    patterns.append(learner_pattern)

