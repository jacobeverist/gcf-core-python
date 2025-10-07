"""Test connections from transformer outputs to learner inputs."""
import pytest
from gnomics import (
    ScalarTransformer,
    DiscreteTransformer,
    PersistenceTransformer,
    PatternPooler,
    PatternClassifier,
    ContextLearner,
    SequenceLearner,
)


class TestScalarTransformerConnections:
    """Test ScalarTransformer output connections to various learners."""

    def test_scalar_to_pattern_pooler(self):
        """Test connecting ScalarTransformer to PatternPooler."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Connect and initialize
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Verify connection
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 100

    def test_scalar_to_pattern_classifier(self):
        """Test connecting ScalarTransformer to PatternClassifier."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Connect and initialize
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        # Verify connection
        assert classifier.input.num_children() == 1
        assert classifier.input.num_bits() == 100

    def test_scalar_to_context_learner(self):
        """Test connecting ScalarTransformer to ContextLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )

        # Connect to both input and context
        learner.input.add_child(encoder.output, 0)
        learner.context.add_child(encoder.output, 1)  # Use previous timestep
        learner.init(num_input_bits=100, num_context_bits=100)

        # Verify connections
        assert learner.input.num_children() == 1
        assert learner.context.num_children() == 1
        assert learner.input.num_bits() == 100
        assert learner.context.num_bits() == 100

    def test_scalar_to_sequence_learner(self):
        """Test connecting ScalarTransformer to SequenceLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )

        # Connect and initialize
        learner.input.add_child(encoder.output, 0)
        learner.init(num_input_bits=100)

        # Verify connection
        assert learner.input.num_children() == 1
        assert learner.input.num_bits() == 100


class TestDiscreteTransformerConnections:
    """Test DiscreteTransformer output connections to various learners."""

    def test_discrete_to_pattern_pooler(self):
        """Test connecting DiscreteTransformer to PatternPooler."""
        encoder = DiscreteTransformer(num_v=10, num_s=100, num_t=2, seed=42)
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Connect and initialize
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Verify connection
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 100

    def test_discrete_to_pattern_classifier(self):
        """Test connecting DiscreteTransformer to PatternClassifier."""
        encoder = DiscreteTransformer(num_v=10, num_s=100, num_t=2, seed=42)
        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Connect and initialize
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        # Verify connection
        assert classifier.input.num_children() == 1
        assert classifier.input.num_bits() == 100

    def test_discrete_to_context_learner(self):
        """Test connecting DiscreteTransformer to ContextLearner."""
        encoder = DiscreteTransformer(num_v=10, num_s=100, num_t=2, seed=42)
        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )

        # Connect to both input and context
        learner.input.add_child(encoder.output, 0)
        learner.context.add_child(encoder.output, 1)
        learner.init(num_input_bits=100, num_context_bits=100)

        # Verify connections
        assert learner.input.num_children() == 1
        assert learner.context.num_children() == 1

    def test_discrete_to_sequence_learner(self):
        """Test connecting DiscreteTransformer to SequenceLearner."""
        encoder = DiscreteTransformer(num_v=10, num_s=100, num_t=2, seed=42)
        learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )

        # Connect and initialize
        learner.input.add_child(encoder.output, 0)
        learner.init(num_input_bits=100)

        # Verify connection
        assert learner.input.num_children() == 1


class TestPersistenceTransformerConnections:
    """Test PersistenceTransformer output connections to various learners."""

    def test_persistence_to_pattern_pooler(self):
        """Test connecting PersistenceTransformer to PatternPooler."""
        encoder = PersistenceTransformer(
            min_val=0.0, max_val=100.0, num_s=100, num_as=10, max_step=50, num_t=2, seed=42
        )
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Connect and initialize
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Verify connection
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 100

    def test_persistence_to_pattern_classifier(self):
        """Test connecting PersistenceTransformer to PatternClassifier."""
        encoder = PersistenceTransformer(
            min_val=0.0, max_val=100.0, num_s=100, num_as=10, max_step=50, num_t=2, seed=42
        )
        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Connect and initialize
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        # Verify connection
        assert classifier.input.num_children() == 1

    def test_persistence_to_context_learner(self):
        """Test connecting PersistenceTransformer to ContextLearner."""
        encoder = PersistenceTransformer(
            min_val=0.0, max_val=100.0, num_s=100, num_as=10, max_step=50, num_t=2, seed=42
        )
        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )

        # Connect to both input and context
        learner.input.add_child(encoder.output, 0)
        learner.context.add_child(encoder.output, 1)
        learner.init(num_input_bits=100, num_context_bits=100)

        # Verify connections
        assert learner.input.num_children() == 1
        assert learner.context.num_children() == 1

    def test_persistence_to_sequence_learner(self):
        """Test connecting PersistenceTransformer to SequenceLearner."""
        encoder = PersistenceTransformer(
            min_val=0.0, max_val=100.0, num_s=100, num_as=10, max_step=50, num_t=2, seed=42
        )
        learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )

        # Connect and initialize
        learner.input.add_child(encoder.output, 0)
        learner.init(num_input_bits=100)

        # Verify connection
        assert learner.input.num_children() == 1


class TestMultipleTransformerConnections:
    """Test connecting multiple transformers to a single learner."""

    def test_multiple_transformers_to_pooler(self):
        """Test connecting multiple transformer outputs to PatternPooler."""
        scalar_enc = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=50, num_as=5, num_t=2, seed=42)
        discrete_enc = DiscreteTransformer(num_v=10, num_s=50, num_t=2, seed=43)

        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=44
        )

        # Connect both encoders
        pooler.input.add_child(scalar_enc.output, 0)
        pooler.input.add_child(discrete_enc.output, 0)
        pooler.init(num_i=100)  # 50 + 50 = 100 bits

        # Verify connections
        assert pooler.input.num_children() == 2
        assert pooler.input.num_bits() == 100

    def test_multiple_transformers_to_classifier(self):
        """Test connecting multiple transformer outputs to PatternClassifier."""
        scalar_enc = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=50, num_as=5, num_t=2, seed=42)
        discrete_enc = DiscreteTransformer(num_v=10, num_s=50, num_t=2, seed=43)

        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=44
        )

        # Connect both encoders
        classifier.input.add_child(scalar_enc.output, 0)
        classifier.input.add_child(discrete_enc.output, 0)
        classifier.init(num_i=100)

        # Verify connections
        assert classifier.input.num_children() == 2
        assert classifier.input.num_bits() == 100

    def test_multiple_transformers_to_context_learner(self):
        """Test connecting multiple transformers to ContextLearner input and context."""
        scalar_enc1 = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=50, num_as=5, num_t=2, seed=42)
        scalar_enc2 = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=50, num_as=5, num_t=2, seed=43)
        discrete_enc = DiscreteTransformer(num_v=10, num_s=50, num_t=2, seed=44)

        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=45
        )

        # Connect to input (scalar1 + scalar2)
        learner.input.add_child(scalar_enc1.output, 0)
        learner.input.add_child(scalar_enc2.output, 0)

        # Connect to context (discrete)
        learner.context.add_child(discrete_enc.output, 0)

        learner.init(num_input_bits=100, num_context_bits=50)

        # Verify connections
        assert learner.input.num_children() == 2
        assert learner.context.num_children() == 1
        assert learner.input.num_bits() == 100
        assert learner.context.num_bits() == 50

    def test_multiple_transformers_to_sequence_learner(self):
        """Test connecting multiple transformers to SequenceLearner."""
        scalar_enc = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=50, num_as=5, num_t=2, seed=42)
        discrete_enc = DiscreteTransformer(num_v=10, num_s=50, num_t=2, seed=43)

        learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=44
        )

        # Connect both encoders
        learner.input.add_child(scalar_enc.output, 0)
        learner.input.add_child(discrete_enc.output, 0)
        learner.init(num_input_bits=100)

        # Verify connections
        assert learner.input.num_children() == 2
        assert learner.input.num_bits() == 100
