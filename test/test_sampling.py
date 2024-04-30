import numpy as np
import pytest
import torch
from jaxtyping import Float, Int

import bootstrap
from src.models import transformer
from src.utils import sampling


@pytest.fixture
def gpt2_model() -> transformer.Transformer:
    model = transformer.Transformer.from_pretrained("gpt2").eval()
    return model


class TestSoftmax:
    def test_softmax_temp(self):
        x = torch.randn(10)

        assert torch.allclose(
            sampling.softmax_temp(x, temperature=1),
            torch.softmax(x, dim=-1),
            atol=1e-4,
        )

        for temp in np.linspace(0.1, 1, 10):
            assert torch.allclose(
                sampling.softmax_temp(x, temperature=temp),
                torch.softmax(x / temp, dim=-1),
                atol=1e-4,
            )

    def test_log_softmax_temp(self):
        x = torch.randn(10)

        assert torch.allclose(
            sampling.log_softmax_temp(x, temperature=1),
            torch.log_softmax(x, dim=-1),
            atol=1e-4,
        )

        assert torch.allclose(
            sampling.log_softmax_temp(x, temperature=1),
            sampling.softmax_temp(x, temperature=1).log(),
            atol=1e-4,
        )

        for temp in np.linspace(0.1, 1, 10):
            assert torch.allclose(
                sampling.log_softmax_temp(x, temperature=temp),
                torch.log_softmax(x / temp, dim=-1),
                atol=1e-4,
            )

            assert torch.allclose(
                sampling.log_softmax_temp(x, temperature=temp),
                sampling.softmax_temp(x, temperature=temp).log(),
                atol=1e-4,
            )


class TestGeneration:
    def test_generate_greedy(self, gpt2_model: transformer.Transformer):
        y_det = bootstrap.generate_rnd(
            gpt2_model,
            prompt=42,
            n_tokens=10,
            temperature=0.0,
            strategy=sampling.GenerationStrategies.DETERMINIST,
            return_log_scores=True,
        )

        y_greedy = bootstrap.generate_rnd(
            gpt2_model,
            prompt=42,
            n_tokens=10,
            temperature=1.5,
            strategy=sampling.GenerationStrategies.SAMPLING,
            return_log_scores=True,
        )

        y_beam = bootstrap.generate_rnd(
            gpt2_model,
            prompt=42,
            n_tokens=10,
            n_beams=5,
            temperature=0.0,
            strategy=sampling.GenerationStrategies.BEAM_SEARCH,
            return_log_scores=True,
        )

        assert (
            y_det.tokens.size()
            == y_greedy.tokens.size()
            == y_beam.tokens.size()
            == (1, 11)
        )

        det_score = gpt2_model.score_sequences(y_det.tokens)
        greedy_score = gpt2_model.score_sequences(y_greedy.tokens)
        beam_score = gpt2_model.score_sequences(y_beam.tokens)

        assert beam_score >= det_score >= greedy_score
