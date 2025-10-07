"""Test chaining learner outputs to other learner inputs."""
import pytest
from gnomics import (
    ScalarTransformer,
    DiscreteTransformer,
    PatternPooler,
    PatternClassifier,
    ContextLearner,
    SequenceLearner,
)


class TestPoolerChaining:
    """Test PatternPooler output connections to other learners."""

    def test_pooler_to_pooler(self):
        """Test connecting PatternPooler to another PatternPooler."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        pooler1 = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        pooler1.input.add_child(encoder.output, 0)
        pooler1.init(num_i=100)

        pooler2 = PatternPooler(
            num_s=25, num_as=3, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler2.input.add_child(pooler1.output, 0)
        pooler2.init(num_i=50)

        # Verify connections
        assert pooler2.input.num_children() == 1
        assert pooler2.input.num_bits() == 50

    def test_pooler_to_classifier(self):
        """Test connecting PatternPooler to PatternClassifier."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        classifier.input.add_child(pooler.output, 0)
        classifier.init(num_i=50)

        # Verify connection
        assert classifier.input.num_children() == 1
        assert classifier.input.num_bits() == 50

    def test_pooler_to_context_learner(self):
        """Test connecting PatternPooler to ContextLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        learner = ContextLearner(
            num_c=50, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        learner.input.add_child(pooler.output, 0)
        learner.context.add_child(pooler.output, 1)
        learner.init(num_input_bits=50, num_context_bits=50)

        # Verify connections
        assert learner.input.num_children() == 1
        assert learner.context.num_children() == 1

    def test_pooler_to_sequence_learner(self):
        """Test connecting PatternPooler to SequenceLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        learner = SequenceLearner(
            num_c=50, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        learner.input.add_child(pooler.output, 0)
        learner.init(num_input_bits=50)

        # Verify connection
        assert learner.input.num_children() == 1


class TestClassifierChaining:
    """Test PatternClassifier output connections to other learners."""

    def test_classifier_to_pooler(self):
        """Test connecting PatternClassifier to PatternPooler."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        pooler = PatternPooler(
            num_s=30, num_as=3, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler.input.add_child(classifier.output, 0)
        pooler.init(num_i=60)

        # Verify connection
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 60

    def test_classifier_to_classifier(self):
        """Test connecting PatternClassifier to another PatternClassifier."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        classifier1 = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        classifier1.input.add_child(encoder.output, 0)
        classifier1.init(num_i=100)

        classifier2 = PatternClassifier(
            num_l=5, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        classifier2.input.add_child(classifier1.output, 0)
        classifier2.init(num_i=60)

        # Verify connection
        assert classifier2.input.num_children() == 1

    def test_classifier_to_context_learner(self):
        """Test connecting PatternClassifier to ContextLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        learner = ContextLearner(
            num_c=60, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        learner.input.add_child(classifier.output, 0)
        learner.context.add_child(encoder.output, 0)
        learner.init(num_input_bits=60, num_context_bits=100)

        # Verify connections
        assert learner.input.num_children() == 1
        assert learner.context.num_children() == 1

    def test_classifier_to_sequence_learner(self):
        """Test connecting PatternClassifier to SequenceLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=42
        )
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        learner = SequenceLearner(
            num_c=60, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        learner.input.add_child(classifier.output, 0)
        learner.init(num_input_bits=60)

        # Verify connection
        assert learner.input.num_children() == 1


class TestContextLearnerChaining:
    """Test ContextLearner output connections to other learners."""

    def test_context_learner_to_pooler(self):
        """Test connecting ContextLearner to PatternPooler."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        learner.input.add_child(encoder.output, 0)
        learner.context.add_child(encoder.output, 1)
        learner.init(num_input_bits=100, num_context_bits=100)

        # ContextLearner has num_c * num_spc output bits = 100 * 8 = 800
        pooler = PatternPooler(
            num_s=100, num_as=10, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler.input.add_child(learner.output, 0)
        pooler.init(num_i=800)

        # Verify connection
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 800

    def test_context_learner_to_classifier(self):
        """Test connecting ContextLearner to PatternClassifier."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        learner.input.add_child(encoder.output, 0)
        learner.context.add_child(encoder.output, 1)
        learner.init(num_input_bits=100, num_context_bits=100)

        classifier = PatternClassifier(
            num_l=4, num_s=800, num_as=80, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        classifier.input.add_child(learner.output, 0)
        classifier.init(num_i=800)

        # Verify connection
        assert classifier.input.num_children() == 1

    def test_context_learner_to_context_learner(self):
        """Test connecting ContextLearner to another ContextLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        learner1 = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        learner1.input.add_child(encoder.output, 0)
        learner1.context.add_child(encoder.output, 1)
        learner1.init(num_input_bits=100, num_context_bits=100)

        learner2 = ContextLearner(
            num_c=800, num_spc=4, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        learner2.input.add_child(learner1.output, 0)
        learner2.context.add_child(encoder.output, 0)
        learner2.init(num_input_bits=800, num_context_bits=100)

        # Verify connections
        assert learner2.input.num_children() == 1
        assert learner2.context.num_children() == 1

    def test_context_learner_to_sequence_learner(self):
        """Test connecting ContextLearner to SequenceLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        context_learner = ContextLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        context_learner.input.add_child(encoder.output, 0)
        context_learner.context.add_child(encoder.output, 1)
        context_learner.init(num_input_bits=100, num_context_bits=100)

        seq_learner = SequenceLearner(
            num_c=800, num_spc=4, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        seq_learner.input.add_child(context_learner.output, 0)
        seq_learner.init(num_input_bits=800)

        # Verify connection
        assert seq_learner.input.num_children() == 1


class TestSequenceLearnerChaining:
    """Test SequenceLearner output connections to other learners."""

    def test_sequence_learner_to_pooler(self):
        """Test connecting SequenceLearner to PatternPooler."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        learner.input.add_child(encoder.output, 0)
        learner.init(num_input_bits=100)

        # SequenceLearner has num_c * num_spc output bits = 100 * 8 = 800
        pooler = PatternPooler(
            num_s=100, num_as=10, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler.input.add_child(learner.output, 0)
        pooler.init(num_i=800)

        # Verify connection
        assert pooler.input.num_children() == 1
        assert pooler.input.num_bits() == 800

    def test_sequence_learner_to_classifier(self):
        """Test connecting SequenceLearner to PatternClassifier."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        learner.input.add_child(encoder.output, 0)
        learner.init(num_input_bits=100)

        classifier = PatternClassifier(
            num_l=4, num_s=800, num_as=80, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        classifier.input.add_child(learner.output, 0)
        classifier.init(num_i=800)

        # Verify connection
        assert classifier.input.num_children() == 1

    def test_sequence_learner_to_context_learner(self):
        """Test connecting SequenceLearner to ContextLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        seq_learner = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        seq_learner.input.add_child(encoder.output, 0)
        seq_learner.init(num_input_bits=100)

        context_learner = ContextLearner(
            num_c=800, num_spc=4, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        context_learner.input.add_child(seq_learner.output, 0)
        context_learner.context.add_child(encoder.output, 0)
        context_learner.init(num_input_bits=800, num_context_bits=100)

        # Verify connections
        assert context_learner.input.num_children() == 1
        assert context_learner.context.num_children() == 1

    def test_sequence_learner_to_sequence_learner(self):
        """Test connecting SequenceLearner to another SequenceLearner."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        learner1 = SequenceLearner(
            num_c=100, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=42
        )
        learner1.input.add_child(encoder.output, 0)
        learner1.init(num_input_bits=100)

        learner2 = SequenceLearner(
            num_c=800, num_spc=4, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=43
        )
        learner2.input.add_child(learner1.output, 0)
        learner2.init(num_input_bits=800)

        # Verify connection
        assert learner2.input.num_children() == 1


class TestComplexChains:
    """Test complex multi-stage chains."""

    def test_three_stage_chain(self):
        """Test a three-stage chain: Encoder -> Pooler -> Classifier -> SequenceLearner."""
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

        seq_learner = SequenceLearner(
            num_c=60, num_spc=4, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=45
        )
        seq_learner.input.add_child(classifier.output, 0)
        seq_learner.init(num_input_bits=60)

        # Verify entire chain
        assert pooler.input.num_children() == 1
        assert classifier.input.num_children() == 1
        assert seq_learner.input.num_children() == 1

    def test_parallel_paths_with_merge(self):
        """Test parallel processing paths that merge."""
        encoder = ScalarTransformer(min_val=0.0, max_val=100.0, num_s=100, num_as=10, num_t=2, seed=42)

        # Path 1: Encoder -> Pooler
        pooler = PatternPooler(
            num_s=50, num_as=5, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=43
        )
        pooler.input.add_child(encoder.output, 0)
        pooler.init(num_i=100)

        # Path 2: Encoder -> Classifier
        classifier = PatternClassifier(
            num_l=3, num_s=60, num_as=6, perm_thr=20, perm_inc=2, perm_dec=1,
            pct_pool=0.8, pct_conn=0.5, pct_learn=0.3, num_t=2, seed=44
        )
        classifier.input.add_child(encoder.output, 0)
        classifier.init(num_i=100)

        # Merge: Both paths -> ContextLearner
        learner = ContextLearner(
            num_c=50, num_spc=8, num_dps=4, num_rpd=20, d_thresh=15,
            perm_thr=20, perm_inc=2, perm_dec=1, num_t=3, seed=45
        )
        learner.input.add_child(pooler.output, 0)
        learner.context.add_child(classifier.output, 0)
        learner.init(num_input_bits=50, num_context_bits=60)

        # Verify merge
        assert learner.input.num_children() == 1
        assert learner.context.num_children() == 1
