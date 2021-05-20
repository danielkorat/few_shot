from typing import Optional, Tuple
from transformers import DataCollatorForLanguageModeling
import torch
from dataclasses import dataclass

from transformers import RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss
from transformers.utils.dummy_tokenizers_objects import GPT2TokenizerFast
from typing import List

class RobertaForMLMWithCE(RobertaForMaskedLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mlm_labels=None,
        is_aspect=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        cross_entropy = CrossEntropyLoss()

        # LABEL LOSS
        len_label_list = 2
        label_prediction_scores = self.pvp.mlm_logits_to_cls_logits(mlm_labels, prediction_scores)
        label_loss = cross_entropy(label_prediction_scores.view(-1, len_label_list), is_aspect.view(-1))

        # MLM LOSS
        masked_lm_loss = cross_entropy(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # FINAL LOSS
        alpha = 1e-4
        loss = (1 - alpha) * label_loss + alpha * masked_lm_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=label_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

STRING_TO_MODEL_CLS = {'RobertaForMLMWithCE': RobertaForMLMWithCE}

@dataclass
class DataCollatorForPatternLanguageModeling(DataCollatorForLanguageModeling):
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

@dataclass
class DataCollatorForPatternLanguageModelingOld(DataCollatorForLanguageModeling):

    pattern: str = None

    def __post_init__(self):
        pattern_tokens = self.tokenizer.tokenize(self.pattern)
        idx = pattern_tokens.index(' ' + self.tokenizer.mask_token)
        self.pattern_mask_idx = idx - len(pattern_tokens)

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Pattern MLM: set pattern-mask token masking probability to 1.0:
        for label_vector, prob_vector in zip(labels.tolist(), probability_matrix):
            mask_idx = (label_vector + [1]).index(1) + self.pattern_mask_idx - 1
            prob_vector[mask_idx] = 1.0

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class BooleanPVP:
    VERBALIZER = {
        "1": ["Yes"],
        "0": ["No"]
    }

    def get_parts(self, example, word):
        text = self.shortenable(example)
        pattern = "Is there sentiment towards <aspect> in the previous sentence?"
        pattern = ' ' + pattern.replace("<aspect>", word)
        return [text, pattern, self.mask]

    def __init__(self, tokenizer, max_seq_length) -> None:
        self.label_list = BooleanPVP.VERBALIZER.keys()
        self.max_seq_length = max_seq_length
        self.max_num_verbalizers = 1
        self.tokenizer = tokenizer
        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.mask_token_id

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    def encode(self, text: str, word: str):
        """
        Encode an input example using this pattern-verbalizer pair.

        :param text: the input example to encode
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        parts = self.get_parts(text, word)
        parts = [x if isinstance(x, tuple) else (x, False) for x in parts]
        token_text = [(x, s) for x, s in parts if x]
        parts = [(self.tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in token_text]
        self.truncate(parts, max_length=self.max_seq_length)
        tokens = [token_id for part, _ in parts for token_id in part]
        input_ids = self.tokenizer.build_inputs_with_special_tokens(tokens, None)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(tokens, None)
        return input_ids, token_type_ids

    def truncate(self, parts, max_length: int):
        """Truncate text to a predefined total maximum length"""
        total_len = self._seq_length(parts) + self.tokenizer.num_special_tokens_to_add(False)
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts

        for _ in range(num_tokens_to_remove):
            self._remove_last(parts)

    def get_mask_positions(self, input_ids):
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def verbalize(self, label):
        return BooleanPVP.VERBALIZER[label]

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers: List[str] = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits


def get_verbalization_ids(word: str, tokenizer, force_single_token: bool):
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {} #{'add_prefix_space': True} if isinstance(tokenizer, GPT2TokenizerFast) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id