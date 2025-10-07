"""Tests for the Pythonic API layer."""

from gnomics.api import (
    BitArray,
    create_category_encoder,
    create_classifier,
    create_context_learner,
    create_pooler,
    create_scalar_encoder,
    create_sequence_learner,
)


class TestFactoryFunctions:
    """Test convenience factory functions."""

    def test_create_scalar_encoder(self) -> None:
        """Test create_scalar_encoder factory."""
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
        assert encoder.num_s() == 10
        assert encoder.num_as() == 5

        # Test with custom parameters
        encoder2 = create_scalar_encoder(
            min_value=10.0, max_value=50.0, num_segments=20, active_per_segment=10
        )
        assert encoder2.num_s() == 20
        assert encoder2.num_as() == 10

    def test_create_category_encoder(self) -> None:
        """Test create_category_encoder factory."""
        encoder = create_category_encoder(num_categories=5)
        assert encoder.num_v() == 5
        assert encoder.num_s() == 10

    def test_create_pooler(self) -> None:
        """Test create_pooler factory."""
        pooler = create_pooler(num_statelets=100, active_statelets=10)
        assert pooler.num_s() == 100
        assert pooler.num_as() == 10
        assert pooler.perm_thr() == 20

    def test_create_classifier(self) -> None:
        """Test create_classifier factory."""
        classifier = create_classifier(num_labels=3, num_statelets=90, active_statelets=10)
        assert classifier.num_l() == 3
        assert classifier.num_s() == 90
        assert classifier.num_as() == 10

    def test_create_context_learner(self) -> None:
        """Test create_context_learner factory."""
        learner = create_context_learner(num_columns=100)
        assert learner.num_c() == 100
        assert learner.num_spc() == 8
        assert learner.num_dps() == 4

    def test_create_sequence_learner(self) -> None:
        """Test create_sequence_learner factory."""
        learner = create_sequence_learner(num_columns=100)
        assert learner.num_c() == 100
        assert learner.num_spc() == 8
        assert learner.num_dps() == 4

    def test_factory_functions_work_correctly(self) -> None:
        """Test that factory-created objects are functional."""
        # Scalar encoder
        scalar = create_scalar_encoder(min_value=0.0, max_value=100.0)
        scalar.set_value(50.0)
        scalar.execute(learn_flag=False)
        output = scalar.output.state()
        assert isinstance(output, BitArray)
        assert output.num_set() > 0

        # Category encoder
        categorical = create_category_encoder(num_categories=5)
        categorical.set_value(2)
        categorical.execute(learn_flag=False)
        output = categorical.output.state()
        assert isinstance(output, BitArray)
        assert output.num_set() > 0

        # Pooler
        pooler = create_pooler(num_statelets=100, active_statelets=10)
        pooler.init(num_i=256)
        input_ba = BitArray(256)
        input_ba.set_acts([1, 5, 10, 15, 20])
        pooler.input.set_state(input_ba)
        pooler.execute(learn_flag=False)
        output = pooler.output.state()
        assert isinstance(output, BitArray)

        # Classifier
        classifier = create_classifier(num_labels=3, num_statelets=90, active_statelets=10)
        classifier.init(num_i=256)
        classifier.set_label(0)
        input_ba = BitArray(256)
        input_ba.set_acts([1, 5, 10])
        classifier.input.set_state(input_ba)
        classifier.execute(learn_flag=True)
        output = classifier.output.state()
        assert isinstance(output, BitArray)

        # Context learner
        learner = create_context_learner(num_columns=50)
        learner.init(num_input_bits=50, num_context_bits=100)
        input_ba = BitArray(50)
        input_ba.set_acts([1, 2, 3])
        context_ba = BitArray(100)
        context_ba.set_acts([10, 20, 30])
        learner.input.set_state(input_ba)
        learner.context.set_state(context_ba)
        learner.execute(learn_flag=False)
        output = learner.output.state()
        assert isinstance(output, BitArray)

        # Sequence learner
        seq_learner = create_sequence_learner(num_columns=50)
        seq_learner.init(num_input_bits=50)
        input_ba = BitArray(50)
        input_ba.set_acts([1, 2, 3])
        seq_learner.input.set_state(input_ba)
        seq_learner.execute(learn_flag=False)
        output = seq_learner.output.state()
        assert isinstance(output, BitArray)
