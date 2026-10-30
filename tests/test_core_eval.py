"""Capping CORE scores must preserve the prompts used by the full evaluation."""
import re

import pytest
import torch

from nanochat import core_eval


class CharacterTokenizer:
    def __init__(self):
        self.prompts = []

    def get_bos_token_id(self):
        return 0

    def __call__(self, prompts, prepend):
        self.prompts.append(prompts)
        return [[prepend, *map(ord, prompt)] for prompt in prompts]


@pytest.mark.parametrize('cap,expected_count', [(1, 1), (8, 8), (100, 30), (-1, 30)])
def test_score_cap_preserves_fewshot_prompts(monkeypatch, cap, expected_count):
    data = [dict(query=f'Question {i}:', choices=['A', 'B'], gold=i % 2) for i in range(30)]
    meta = dict(task_type='multiple_choice', num_fewshot=10, continuation_delimiter=' ')

    def prefer_first_choice(model, input_ids):
        # Stub only neural inference. Real sampling, rendering, tokenization,
        # continuation scoring and task aggregation run below.
        losses = torch.ones_like(input_ids, dtype=torch.float32)
        losses[0] = 0
        return losses, torch.zeros_like(input_ids)

    monkeypatch.setattr(core_eval, 'forward_model', prefer_first_choice)
    full = CharacterTokenizer()
    assert core_eval.evaluate_task(None, full, data, 'cpu', meta) == 0.5
    capped = CharacterTokenizer()
    score = core_eval.evaluate_task(None, capped, data, 'cpu', meta, max_examples=cap)
    assert len(capped.prompts) == expected_count
    assert capped.prompts == full.prompts[:expected_count]
    assert score == pytest.approx((expected_count + 1) // 2 / expected_count)
    for idx, prompts in enumerate(capped.prompts):
        questions = [int(x) for x in re.findall(r'Question (\d+):', prompts[0])]
        assert len(questions) == 11 and len(set(questions)) == 11
        assert questions[-1] == idx and idx not in questions[:-1]
