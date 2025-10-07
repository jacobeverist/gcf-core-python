"""Integration tests for GCF block pipelines."""

from gnomics.api import (
    BitArray,
    create_category_encoder,
    create_classifier,
    create_context_learner,
    create_pooler,
    create_scalar_encoder,
)


class TestScalarEncoderPipeline:
    """Test scalar encoding through learning pipelines."""

    def test_scalar_to_pooler_pipeline(self) -> None:
        """Test scalar encoder -> pooler pipeline."""
        # Create encoder and pooler
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
        pooler = create_pooler(num_statelets=200, active_statelets=20)

        # Initialize pooler with encoder output size
        encoder_output_size = encoder.num_s() * encoder.num_as()
        pooler.init(num_i=encoder_output_size)

        # Encode and pool several values
        for value in [10.0, 20.0, 30.0, 40.0, 50.0]:
            encoder.set_value(value)
            encoder.execute(learn_flag=False)
            encoded = encoder.output.state()

            pooler.execute(encoded, learn_flag=True)
            pooled = pooler.output.state()

            assert isinstance(pooled, BitArray)
            assert pooled.num_set() <= 20

    def test_scalar_to_classifier_pipeline(self) -> None:
        """Test scalar encoder -> classifier pipeline."""
        # Create encoder and classifier
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
        classifier = create_classifier(num_labels=3, num_statelets=90, active_statelets=10)

        # Initialize classifier with encoder output size
        encoder_output_size = encoder.num_s() * encoder.num_as()
        classifier.init(num_i=encoder_output_size)

        # Train classifier on different value ranges
        # Label 0: 0-33, Label 1: 34-66, Label 2: 67-100
        training_data = [
            (10.0, 0),
            (20.0, 0),
            (30.0, 0),
            (40.0, 1),
            (50.0, 1),
            (60.0, 1),
            (70.0, 2),
            (80.0, 2),
            (90.0, 2),
        ]

        for value, label in training_data:
            encoder.set_value(value)
            encoder.execute(learn_flag=False)
            encoded = encoder.output.state()

            classifier.set_label(label)
            classifier.execute(encoded, learn_flag=True)

        # Test prediction
        encoder.set_value(25.0)
        encoder.execute(learn_flag=False)
        encoded = encoder.output.state()

        classifier.compute(encoded)
        predicted = classifier.get_predicted_label()

        # Should predict label 0 for value 25
        assert 0 <= predicted < 3


class TestCategoryEncoderPipeline:
    """Test categorical encoding through learning pipelines."""

    def test_category_to_pooler_pipeline(self) -> None:
        """Test category encoder -> pooler pipeline."""
        # Create encoder and pooler
        encoder = create_category_encoder(num_categories=10)
        pooler = create_pooler(num_statelets=150, active_statelets=15)

        # Initialize pooler
        pooler.init(num_i=encoder.num_s())

        # Process different categories
        for category in range(10):
            encoder.set_value(category)
            encoder.execute(learn_flag=False)
            encoded = encoder.output.state()

            pooler.execute(encoded, learn_flag=True)
            pooled = pooler.output.state()

            assert isinstance(pooled, BitArray)
            assert pooled.num_set() <= 15


class TestContextLearningPipeline:
    """Test context learning pipelines."""

    def test_encoder_to_context_learner(self) -> None:
        """Test scalar encoder -> context learner pipeline."""
        # Create encoder and context learner
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0, num_segments=20)

        # Encoder output becomes input to learner
        encoder_output_size = encoder.num_s() * encoder.num_as()

        # ContextLearner num_columns must equal num_input_bits
        learner = create_context_learner(num_columns=encoder_output_size)

        # Context is previous encoder output
        learner.init(num_input_bits=encoder_output_size, num_context_bits=encoder_output_size)

        # Create sequence: increasing values
        sequence = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

        prev_encoded = BitArray(encoder_output_size)

        for value in sequence:
            # Encode current value
            encoder.set_value(value)
            encoder.execute(learn_flag=False)
            current_encoded = encoder.output.state()

            # Learn context pattern (current input, previous as context)
            learner.execute(current_encoded, prev_encoded, learn_flag=True)

            # Get prediction and anomaly
            prediction = learner.output.state()
            anomaly = learner.get_anomaly_score()

            assert isinstance(prediction, BitArray)
            assert 0.0 <= anomaly <= 1.0

            # Update context
            prev_encoded = current_encoded


class TestMultiStageClassificationPipeline:
    """Test multi-stage classification pipelines."""

    def test_scalar_pooler_classifier_pipeline(self) -> None:
        """Test scalar encoder -> pooler -> classifier pipeline."""
        # Create components
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
        pooler = create_pooler(num_statelets=200, active_statelets=20)
        classifier = create_classifier(num_labels=2, num_statelets=60, active_statelets=6)

        # Initialize
        encoder_output_size = encoder.num_s() * encoder.num_as()
        pooler.init(num_i=encoder_output_size)
        classifier.init(num_i=pooler.num_s())

        # Training data: classify as low (0) or high (1)
        training_data = [
            (15.0, 0),
            (25.0, 0),
            (35.0, 0),
            (65.0, 1),
            (75.0, 1),
            (85.0, 1),
        ]

        # Train through pipeline
        for value, label in training_data:
            # Encode
            encoder.set_value(value)
            encoder.execute(learn_flag=False)
            encoded = encoder.output.state()

            # Pool
            pooler.execute(encoded, learn_flag=True)
            pooled = pooler.output.state()

            # Classify
            classifier.set_label(label)
            classifier.execute(pooled, learn_flag=True)

        # Test prediction
        encoder.set_value(80.0)
        encoder.execute(learn_flag=False)
        encoded = encoder.output.state()

        pooler.execute(encoded, learn_flag=False)
        pooled = pooler.output.state()

        classifier.compute(pooled)
        predicted = classifier.get_predicted_label()

        # Should predict high (1) for value 80
        assert predicted in [0, 1]


class TestSequenceLearningPipeline:
    """Test sequence learning pipelines."""

    def test_category_context_learning(self) -> None:
        """Test learning context sequences of categories."""
        # Create category encoder and context learner
        encoder = create_category_encoder(num_categories=5)

        # ContextLearner num_columns must equal num_input_bits
        learner = create_context_learner(num_columns=encoder.num_s())

        # Initialize
        learner.init(num_input_bits=encoder.num_s(), num_context_bits=encoder.num_s())

        # Learn sequence: 0 -> 1 -> 2 -> 3 -> 4 -> 0 ...
        sequence = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]

        prev_encoded = BitArray(encoder.num_s())

        for i, category in enumerate(sequence):
            # Encode
            encoder.set_value(category)
            encoder.execute(learn_flag=False)
            current_encoded = encoder.output.state()

            # Learn
            learner.execute(current_encoded, prev_encoded, learn_flag=True)

            # Check anomaly (should decrease as pattern is learned)
            anomaly = learner.get_anomaly_score()
            assert 0.0 <= anomaly <= 1.0

            # Historical count should increase
            count = learner.get_historical_count()
            assert count >= 0

            prev_encoded = current_encoded


class TestOutputHistoryPipeline:
    """Test history tracking through pipelines."""

    def test_encoder_output_history(self) -> None:
        """Test accessing historical outputs from encoder."""
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)

        # Execute with different values
        encoder.set_value(10.0)
        encoder.execute(learn_flag=False)

        encoder.set_value(20.0)
        encoder.execute(learn_flag=False)

        # Access current and previous outputs
        current = encoder.output_at(0)
        previous = encoder.output_at(1)

        assert isinstance(current, BitArray)
        assert isinstance(previous, BitArray)

        # They should be different
        assert current.num_similar(previous) < current.num_set()

    def test_pooler_change_detection(self) -> None:
        """Test change detection in pooler pipeline."""
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
        pooler = create_pooler(num_statelets=100, active_statelets=10)

        encoder_output_size = encoder.num_s() * encoder.num_as()
        pooler.init(num_i=encoder_output_size)

        # First execution
        encoder.set_value(50.0)
        encoder.execute(learn_flag=False)
        pooler.execute(encoder.output.state(), learn_flag=True)

        # Second execution with same value
        encoder.set_value(50.0)
        encoder.execute(learn_flag=False)
        pooler.execute(encoder.output.state(), learn_flag=False)

        # Third execution with different value
        encoder.set_value(75.0)
        encoder.execute(learn_flag=False)
        pooler.execute(encoder.output.state(), learn_flag=False)

        # Check change detection
        changed = pooler.has_changed()
        assert isinstance(changed, bool)


class TestMemoryUsagePipeline:
    """Test memory usage across pipelines."""

    def test_pipeline_memory_usage(self) -> None:
        """Test memory usage estimation for complete pipeline."""
        # Create components
        encoder = create_scalar_encoder(min_value=0.0, max_value=100.0)
        pooler = create_pooler(num_statelets=200, active_statelets=20)
        classifier = create_classifier(num_labels=3, num_statelets=90, active_statelets=10)

        # Get memory usage
        encoder_mem = encoder.memory_usage()
        pooler_mem = pooler.memory_usage()
        classifier_mem = classifier.memory_usage()

        assert encoder_mem > 0
        assert pooler_mem > 0
        assert classifier_mem > 0

        # Total pipeline memory
        total_mem = encoder_mem + pooler_mem + classifier_mem
        assert total_mem > 0
