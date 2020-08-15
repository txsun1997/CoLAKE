import os
import torch
import torch.nn as nn
import numpy as np
from fastNLP import seq_len_to_mask
from transformers import RobertaForMaskedLM, RobertaTokenizer, BertPreTrainedModel, RobertaModel, RobertaConfig
from transformers.modeling_roberta import RobertaLMHead


class Roberta(object):

    def __init__(self, args):
        # self.dict_file = "{}/{}".format(args.roberta_model_dir, args.roberta_vocab_name)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        if args.model_path is not None:
            print("Testing CoLAKE...")
            print('loading model parameters from {}...'.format(args.model_path))
            config = RobertaConfig.from_pretrained('roberta-base', type_vocab_size=3)
            self.model = RobertaForMaskedLM(config=config)
            states_dict = torch.load(os.path.join(args.model_path, 'model.bin'))
            self.model.load_state_dict(states_dict, strict=False)
        else:
            print("Testing RoBERTa baseline...")
            self.model = RobertaForMaskedLM.from_pretrained('roberta-base')

        self._build_vocab()
        self._init_inverse_vocab()
        self._model_device = 'cpu'
        self.max_sentence_length = args.max_sentence_length

    def _cuda(self):
        self.model.cuda()

    def _build_vocab(self):
        self.vocab = []
        for key in range(len(self.tokenizer)):
            value = self.tokenizer.decode([key])
            if value[0] == " ":  # if the token starts with a whitespace
                value = value.strip()
            else:
                # this is subword information
                value = "_{}_".format(value)

            if value in self.vocab:
                # print("WARNING: token '{}' is already in the vocab".format(value))
                value = "{}_{}".format(value, key)

            self.vocab.append(value)
        print("size of vocabulary: {}".format(len(self.vocab)))

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                self._cuda()
                self._model_device = 'cuda'
        else:
            print('No CUDA found')

    def init_indices_for_filter_logprobs(self, vocab_subset):
        index_list = []
        new_vocab_subset = []
        for word in vocab_subset:
            if word in self.inverse_vocab:
                inverse_id = self.inverse_vocab[word]
                index_list.append(inverse_id)
                new_vocab_subset.append(word)
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                print("WARNING: {}".format(msg))

        indices = torch.as_tensor(index_list)
        return indices, index_list

    def filter_logprobs(self, log_probs, indices):
        new_log_probs = log_probs.index_select(dim=2, index=indices)
        return new_log_probs

    def get_id(self, input_string):
        # Roberta predicts ' London' and not 'London'
        string = " " + str(input_string).strip()
        tokens = self.tokenizer.encode(string, add_special_tokens=False)
        # return [element.item() for element in tokens.long().flatten()]
        return tokens

    def get_batch_generation(self, samples_list, try_cuda=True):
        if not samples_list:
            return None
        if try_cuda:
            self.try_cuda()

        tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        seq_len = []
        for sample in samples_list:
            masked_inputs_list = sample["masked_sentences"]

            tokens_list = [self.tokenizer.bos_token_id]

            for idx, masked_input in enumerate(masked_inputs_list):
                tokens_list.extend(self.tokenizer.encode(" " + masked_input.strip(), add_special_tokens=False))
                tokens_list.append(self.tokenizer.eos_token_id)

            # tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            tokens = torch.tensor(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.long().cpu().numpy())

            seq_len.append(len(tokens))
            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.tokenizer.mask_token_id).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])
        tokens_list = []
        for tokens in tensor_list:
            pad_lenght = max_len - len(tokens)
            if pad_lenght > 0:
                pad_tensor = torch.full([pad_lenght], self.tokenizer.pad_token_id, dtype=torch.int)
                tokens = torch.cat((tokens, pad_tensor.long()))
            tokens_list.append(tokens)

        batch_tokens = torch.stack(tokens_list)
        seq_len = torch.LongTensor(seq_len)
        attn_mask = seq_len_to_mask(seq_len)

        with torch.no_grad():
            # with utils.eval(self.model.model):
            self.model.eval()
            outputs = self.model(
                batch_tokens.long().to(device=self._model_device),
                attention_mask=attn_mask.to(device=self._model_device)
            )
            log_probs = outputs[0]

        return log_probs.cpu(), output_tokens_list, masked_indices_list

