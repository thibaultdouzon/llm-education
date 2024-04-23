import numpy as np
import pytest
import torch
from jaxtyping import Float, Int

from src.utils import sampling


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
