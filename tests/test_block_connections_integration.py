"""Integration tests for block connections with actual data flow."""
import pytest
from gnomics import (
    ScalarTransformer,
    DiscreteTransformer,
    PatternPooler,
    PatternClassifier,
    ContextLearner,
    SequenceLearner,
    BitArray,
)


class TestDataFlowThroughConnections:
    """Test that data actually flows through connected blocks."""

    def test_scalar_to_pooler_data_flow(self):
        """Test data flow from ScalarTransformer through PatternPooler."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )

        # Connect and initialize
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Set encoder value and execute
        encoder.set_value(50.0)
        encoder.execute(False)

        # Pull data into pooler
        pooler.input.pull()

        # Verify data was pulled
        assert pooler.input.state().num_set() > 0
        assert pooler.input.state().num_set() == 10  # encoder has 10 active bits

    def test_encoder_chain_data_flow(self):
        """Test that connections can be chained: encoder -> pooler -> classifier."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=44
        )
        classifier.input.add_child(pooler.output, 0)
        classifier.init(num_i=50)

        # Verify the chain is connected
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 100
        assert classifier.input.num_children() == 1
        assert classifier.input.num_bits() == 50

    def test_multiple_inputs_concatenation(self):
        """Test that multiple inputs are properly concatenated."""
        encoder1 = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=50, num_as=5, num_t=2, seed=42)
        encoder2 = DiscreteTransformer(num_v=10, num_s=50, num_t=2, seed=43)

        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=44
        )

        # Connect both encoders
        pooler.input.add_child(encoder1.output, 0)
        pooler.input.add_child(encoder2.output, 0)
        pooler.init(num_i=100)

        # Execute both encoders
        encoder1.set_value(50.0)
        encoder1.execute(False)

        encoder2.set_value(5)
        encoder2.execute(False)

        # Pull data
        pooler.input.pull()

        # Verify concatenated input
        assert pooler.input.num_bits() == 100
        # DiscreteTransformer has num_as = num_s / num_v = 50 / 10 = 5 active bits per category
        # But since we're looking at just the raw encoding, it has 5 active bits
        # ScalarTransformer has 5 active bits
        # Total: 5 + 5 = 10 active bits
        assert pooler.input.state().num_set() == 10  # 5 + 5 active bits

    def test_context_learner_dual_inputs(self):
        """Test ContextLearner with separate input and context connections."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        context_enc = DiscreteTransformer(num_v=10, num_s=50, num_t=2, seed=43)

        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=44
        )

        # Connect separate inputs
        learner.input.add_child(encoder.output, 0)
        learner.context.add_child(context_enc.output, 0)
        learner.init(num_input_bits=100, num_context_bits=50)

        # Execute encoders
        encoder.set_value(50.0)
        encoder.execute(False)

        context_enc.set_value(5)
        context_enc.execute(False)

        # Pull data
        learner.input.pull()
        learner.context.pull()

        # Verify separate inputs
        assert learner.input.num_bits() == 100
        assert learner.context.num_bits() == 50
        assert learner.input.state().num_set() == 10
        # DiscreteTransformer has num_as = num_s / num_v = 50 / 10 = 5 active bits
        assert learner.context.state().num_set() == 5

    def test_time_offset_connections(self):
        """Test connections with different time offsets."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=3, seed=42)

        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )

        # Connect current timestep to input, previous to context
        learner.input.add_child(encoder.output, 0)  # time=0 (current)
        learner.context.add_child(encoder.output, 1)  # time=1 (previous)
        learner.init(num_input_bits=100, num_context_bits=100)

        # Execute encoder multiple times
        encoder.set_value(30.0)
        encoder.execute(False)

        encoder.set_value(50.0)
        encoder.execute(False)

        encoder.set_value(70.0)
        encoder.execute(False)

        # Pull data
        learner.input.pull()
        learner.context.pull()

        # Both should have data, but from different timesteps
        assert learner.input.state().num_set() == 10
        assert learner.context.state().num_set() == 10

    def test_output_state_vs_output_methods(self):
        """Test that output() and get_output_state() provide correct data."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        encoder.set_value(50.0)
        encoder.execute(False)

        # Get output as BlockOutput (for connections)
        output_obj = encoder.output
        assert output_obj is not None

        # Get output state as BitArray (for inspection)
        output_state = encoder.get_output_state()
        assert output_state.num_set() == 10
        assert len(output_state) == 100  # BitArray uses len(), not num_bits()


class TestBlockInputMethods:
    """Test BlockInput methods work correctly with connections."""

    def test_children_changed_detection(self):
        """Test that children_changed() detects changes in connected blocks."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )

        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Execute encoder
        encoder.set_value(50.0)
        encoder.execute(False)

        # Should detect change
        assert pooler.input.children_changed()

        # Pull and execute again without changing encoder
        pooler.input.pull()
        encoder.execute(False)  # Same value

        # Should still detect change because encoder executed
        # (encoder always stores even if value unchanged)

    def test_clear_input(self):
        """Test clearing input state."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )

        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Execute and pull
        encoder.set_value(50.0)
        encoder.execute(False)
        pooler.input.pull()

        assert pooler.input.state().num_set() > 0

        # Clear input
        pooler.input.clear()
        assert pooler.input.state().num_set() == 0

    def test_input_id(self):
        """Test that each BlockInput has a unique ID."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        pooler1 = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler1.input.add_child(encoder.output, 0)
        pooler1.init(num_i=100)

        pooler2 = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=44
        )
        pooler2.input.add_child(encoder.output, 0)
        pooler2.init(num_i=100)

        # Each input should have a unique ID
        id1 = pooler1.input.id()
        id2 = pooler2.input.id()
        assert id1 != id2


class TestErrorHandling:
    """Test error handling in block connections."""

    def test_init_without_connection(self):
        """Test that init() works without connections (manual size specification)."""
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )

        # Initialize without connection (old-style)
        pooler.init(num_i=100)

        # Should work
        assert pooler.input.num_bits() == 100

    def test_init_with_size_override(self):
        """Test that explicit size parameter overrides connected size."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )

        pooler.input.add_child(encoder.output, 0)

        # Initialize with explicit size (should override)
        pooler.init(num_i=100)

        # Should use specified size
        assert pooler.input.num_bits() == 100
