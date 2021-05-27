from typing import Optional, Tuple
from transformers import DataCollatorForLanguageModeling
import torch
from dataclasses import dataclass

from transformers import RobertaForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput
from torch.nn import CrossEntropyLoss

class P_B_13:
    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    } 

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.label_list)

    def __init__(self, tokenizer) -> None:
        self.label_list = ["0", "1"]
        self.tokenizer = tokenizer
        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def verbalize(self, label):
        return P_B_13.VERBALIZER[label]

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.label_list],
                                    dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

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
    # kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id

class RobertaForMLMWithCE(RobertaForMaskedLM):
    def set_pvp(self, pvp):
        self.pvp = pvp

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
        scoring_labels=None,
        mask_positions=None,
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

        loss_fct = CrossEntropyLoss()

        # LABEL LOSS
        len_label_list = 2
        scoring_logits = self.pvp.convert_mlm_logits_to_cls_logits(mask_positions, prediction_scores)
        label_loss = loss_fct(scoring_logits.view(-1, len_label_list), scoring_labels.view(-1))

        masked_lm_loss = None
        if labels is not None:
            # MLM LOSS
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # FINAL LOSS
        alpha = 1e-4
        loss = (1 - alpha) * label_loss + alpha * masked_lm_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

STR_TO_MODEL_CLS = {'RobertaForMLMWithCE': RobertaForMLMWithCE}

@dataclass
class DataCollatorForPatternLanguageModeling(DataCollatorForLanguageModeling):

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


# @dataclass
# class DataCollatorForLanguageModeling:
#     """
#     Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
#     are not all of the same length.
#     Args:
#         tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
#             The tokenizer used for encoding the data.
#         mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
#             Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
#             inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
#             non-masked tokens and the value to predict for the masked token.
#         mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
#             The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
#     .. note::
#         For best performance, this data collator should be used with a dataset having items that are dictionaries or
#         BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
#         :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
#         argument :obj:`return_special_tokens_mask=True`.
#     """

#     tokenizer: PreTrainedTokenizerBase
#     mlm: bool = True
#     mlm_probability: float = 0.15

#     def __post_init__(self):
#         if self.mlm and self.tokenizer.mask_token is None:
#             raise ValueError(
#                 "This tokenizer does not have a mask token which is necessary for masked language modeling. "
#                 "You should pass `mlm=False` to train on causal language modeling instead."
#             )

#     def __call__(
#         self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
#     ) -> Dict[str, torch.Tensor]:
#         # Handle dict or lists with proper padding and conversion to tensor.
#         if isinstance(examples[0], (dict, BatchEncoding)):
#             batch = self.tokenizer.pad(examples, return_tensors="pt")
#         else:
#             batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

#         # If special token mask has been preprocessed, pop it from the dict.
#         special_tokens_mask = batch.pop("special_tokens_mask", None)
#         if self.mlm:
#             batch["input_ids"], batch["labels"] = self.mask_tokens(
#                 batch["input_ids"], special_tokens_mask=special_tokens_mask
#             )
#         else:
#             labels = batch["input_ids"].clone()
#             if self.tokenizer.pad_token_id is not None:
#                 labels[labels == self.tokenizer.pad_token_id] = -100
#             batch["labels"] = labels
#         return batch

#     def mask_tokens(
#         self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
#         """
#         labels = inputs.clone()
#         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
#         probability_matrix = torch.full(labels.shape, self.mlm_probability)
#         if special_tokens_mask is None:
#             special_tokens_mask = [
#                 self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
#             ]
#             special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
#         else:
#             special_tokens_mask = special_tokens_mask.bool()

#         probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100  # We only compute loss on masked tokens

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#         random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
#         inputs[indices_random] = random_words[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#         return inputs, labels