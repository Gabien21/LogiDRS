# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
from typing import List, Dict, Any, Tuple
from itertools import groupby
from operator import itemgetter
import copy
from transformers import BertPreTrainedModel, RobertaModel, BertModel
from attention import TransformerLayer

# Simple linear classification head
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class LogiDRS(BertPreTrainedModel):
    '''
    LogiDRS model customized for logical reasoning and discourse-aware modeling, using discourse relation sense information.
    '''

    def __init__(self,
                 config,
                 init_weights: bool,
                 max_rel_id,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 token_encoder_type: str = "roberta",
                 use_discourse: bool = False,
                 n_transformer_layer: int = 2
                ) -> None:
        super().__init__(config)

        self.token_encoder_type = token_encoder_type
        self.max_rel_id = max_rel_id
        self.use_discourse = use_discourse
        self.n_transformer_layer = n_transformer_layer

        # Load pretrained Roberta model
        self.roberta = RobertaModel(config)

        # for param in self.roberta.parameters():
        #     param.requires_grad = False

        # Discourse classification head from external pretrained layers
        if self.use_discourse:
            self._discourse_classification_head = ClassificationHead(input_dim=config.hidden_size, output_dim=23)
            state_dict = torch.load("classification_head/custom_layers.pth")
            self._discourse_dropout1 = nn.Dropout(config.hidden_dropout_prob)
            self._discourse_classification_head.linear.load_state_dict(state_dict["classification_head"]) 
            self._discourse_dropout2 = nn.Dropout(config.hidden_dropout_prob)

            for param in self._discourse_classification_head.parameters():
                param.requires_grad = False

            self._discourse_information_transform = nn.Linear(config.hidden_size + 23, config.hidden_size) # 23 is the number of classes in PDTB2.0

        # Custom transformer layer
        self.num_heads = 4
        self.ff_hidden_size = config.hidden_size * 2
        self.n_layers = self.n_transformer_layer
        self._transformer_block = TransformerLayer(self.n_layers,config.hidden_size, self.num_heads, self.ff_hidden_size,dropout = config.hidden_dropout_prob, max_len=64)
        
        # Linear Projection Layer and Classfication Head
        if self.use_discourse == False:
            self._transformer_linear_projection = nn.Linear(config.hidden_size * 2 , config.hidden_size)
        else:
            self._transformer_linear_projection = nn.Linear(config.hidden_size * 3 , config.hidden_size)
        self._transformer_dropout = nn.Dropout(config.hidden_dropout_prob)
        self._transformer_classifier = nn.Linear(config.hidden_size,1)
        if init_weights:
            self.init_weights()


    def split_into_spans_9(self, seq, seq_mask, split_bpe_ids):
        def _consecutive(seq: list, vals: np.array):
            groups_seq = []
            output_vals = copy.deepcopy(vals)
            for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
                groups_seq.append(list(map(itemgetter(1), g)))
            output_seq = []
            for i, ids in enumerate(groups_seq):
                output_seq.append(ids[0])
                if len(ids) > 1:
                    output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
            return groups_seq, output_seq, output_vals

        embed_size = seq.size(-1)
        device = seq.device
        encoded_spans = []
        encoded_edges = []
        span_masks = []
        edges = []
        node_in_seq_indices = []
        span_edge_sents = []
        encoded_sents = []
        for item_seq_mask, item_seq, item_split_ids in zip(seq_mask, seq, split_bpe_ids):
            item_seq_len = item_seq_mask.sum().item()
            item_seq = item_seq[:item_seq_len]
            item_split_ids = item_split_ids[:item_seq_len]
            item_split_ids = item_split_ids.cpu().numpy()
            if item_split_ids[-1] != 5:
                if item_seq_len < 256:
                    item_split_ids = np.concatenate([item_split_ids, np.array([5])])
                    item_seq_len += 1
                else:
                    item_split_ids[-1] = 5
            if item_split_ids[-2] != 5:
                if item_seq_len < 256:
                    item_split_ids = np.concatenate([item_split_ids, np.array([5])])
                    item_seq_len += 1
                else:
                    item_split_ids[-2] = 5
            if item_split_ids[-3] != 0:
                if item_seq_len < 256:
                    item_split_ids = np.concatenate([item_split_ids[:-2], np.array([0]), item_split_ids[-2:]])
                    item_seq_len += 1
                else:
                    item_split_ids[-3] = 0
                    
            split_ids_indices = np.where(item_split_ids > 0)[0].tolist()
            grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
                split_ids_indices, item_split_ids)
            n_split_ids = len(split_ids_indices)

            #--------------------------Discourse Info--------------------------
            edges_embedings = torch.zeros(len(grouped_split_ids_indices[1:-1]), embed_size, device=seq.device)
            for i, idxs in enumerate(grouped_split_ids_indices[1:-1]):
                span = item_seq[idxs]
                if len(span) != 0:
                    edges_embedings[i] = span.sum(0)

            if self.use_discourse:
                punct_idx_list = []
                connective_idx_list = []
                for i in grouped_split_ids_indices:
                    if item_split_ids[i[0]] == 4:
                        connective_idx_list.append(i)
                    else:
                        punct_idx_list.append(i)
    
                sequence_discourse_span = torch.zeros(len(grouped_split_ids_indices[1:-1]), embed_size, device=seq.device)

                for i in range(1, len(grouped_split_ids_indices)-1):
                    if grouped_split_ids_indices[i] in punct_idx_list:
                        start = punct_idx_list[punct_idx_list.index(grouped_split_ids_indices[i])-1]
                        end = punct_idx_list[punct_idx_list.index(grouped_split_ids_indices[i])+1]
                    else:
                        start = grouped_split_ids_indices[i-1]
                        end = grouped_split_ids_indices[i+1]
                    if i+1 < len(grouped_split_ids_indices) and  grouped_split_ids_indices[i+1] not in punct_idx_list:
                        span = item_seq[start[-1]+1:end[0]]  
                    else:
                        span = item_seq[start[-1]+1:end[-1]+1] 
                    if len(span) != 0:
                        sequence_discourse_span[i-1] = span.sum(0)
                discourse_info = self._discourse_classification_head(sequence_discourse_span)
                discourse_info = self._discourse_dropout1(discourse_info)
                discourse_info = self._discourse_information_transform(torch.concat([discourse_info, edges_embedings], dim=1))
                discourse_info = self._discourse_dropout2(discourse_info)
                encoded_edges.append(list(edges_embedings))
            else:
                encoded_edges.append(list(edges_embedings))
            #--------------------------End Discourse Info--------------------------

            
            item_spans, item_mask = [], []
            item_edges = []
            item_node_in_seq_indices = []
            item_edges.append(item_split_ids[split_ids_indices[0]])
            for i in range(n_split_ids):
                if i == n_split_ids - 1:
                    span = item_seq[grouped_split_ids_indices[i][-1] + 1:]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)

                else:
                    span = item_seq[grouped_split_ids_indices[i][-1] + 1:split_ids_indices[i + 1]]
                    if not len(span) == 0:
                        item_spans.append(span.sum(0))
                        item_mask.append(1)
                        item_edges.append(item_split_ids[split_ids_indices[i + 1]])
                        item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                                            grouped_split_ids_indices[i + 1][0])])
            


            encoded_spans.append(item_spans)
            span_masks.append(item_mask)
            edges.append(item_edges)
            node_in_seq_indices.append(item_node_in_seq_indices)


            # Sentence :EDUs and Connectives (Spans and Edges)
            span_edge_sent = [1] * n_split_ids
            span_sent = [span_edge_sent[i] if i == len(span_edge_sent) - 1 else (span_edge_sent[i], 0) for i in range(len(span_edge_sent))]

            span_sent = [item for sublist in span_sent for item in (sublist if isinstance(sublist, tuple) else [sublist])]
            
            span_sent = span_sent[1:-1]
            span_edge_sents.append(span_sent)
            sent_embedding=[]
            span_idx = 0
            edge_idx = 0
            for i in range(len(span_sent)) : 
                if span_sent[i] == 1 : 
                    sent_embedding.append((edges_embedings[edge_idx]))
                    edge_idx+=1
                if span_sent[i] == 0 : 
                    sent_embedding.append((item_spans[span_idx]))
                    span_idx+=1
            encoded_sents.append(sent_embedding)
        


        max_nodes = max(map(len, span_masks))
        span_masks = [spans + [0] * (max_nodes - len(spans)) for spans in span_masks]
        span_masks = torch.from_numpy(np.array(span_masks))
        span_masks = span_masks.to(device).long()

        pad_embed = torch.zeros(embed_size, dtype=seq.dtype, device=seq.device)
        encoded_spans = [spans + [pad_embed] * (max_nodes - len(spans)) for spans in encoded_spans]
        encoded_spans = [torch.stack(lst, dim=0) for lst in encoded_spans]
        encoded_spans = torch.stack(encoded_spans, dim=0)
        encoded_spans = encoded_spans.to(device).float()

        encoded_edges = [spans_edge + [pad_embed] * (max_nodes -1 - len(spans_edge)) for spans_edge in encoded_edges]
        encoded_edges = [torch.stack(lst, dim=0) for lst in encoded_edges]
        encoded_edges = torch.stack(encoded_edges, dim=0)
        encoded_edges = encoded_edges.to(device).float()

        max_nodes_sent = max(map(len, span_masks)) * 2
        encoded_sents = [spans_edge + [pad_embed] * (max_nodes_sent -1 - len(spans_edge)) for spans_edge in encoded_sents]
        encoded_sents = [torch.stack(lst, dim=0) for lst in encoded_sents]
        encoded_sents = torch.stack(encoded_sents, dim=0)
        encoded_sents = encoded_sents.to(device).float()
        truncated_edges = [item[1:-1] for item in edges]

        return encoded_spans, span_masks, truncated_edges, node_in_seq_indices, encoded_edges, encoded_sents

    def get_gcn_info_vector(self, indices, node, size, device):
        '''

        :param indices: list(len=bsz) of list(len=n_notes) of list(len=varied).
        :param node: (bsz, n_nodes, embed_size)
        :param size: value=(bsz, seq_len, embed_size)
        :param device:
        :return:
        '''

        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                gcn_info_vec[b, ids] = emb

        return gcn_info_vec


    def get_adjacency_matrices_2(self, edges:List[List[int]], n_nodes:int, device:torch.device):
        '''
        Convert the edge_value_list into adjacency matrices.
            * argument graph adjacency matrix. Asymmetric (directed graph).
            * punctuation graph adjacency matrix. Symmetric (undirected graph).

            : argument
                - edges:list[list[str]]. len_out=(bsz x n_choices), len_in=n_edges. value={-1, 0, 1, 2, 3, 4, 5}.

            Note: relation patterns
                1 - (relation, head, tail)  关键词在句首
                2 - (head, relation, tail)  关键词在句中，先因后果
                3 - (tail, relation, head)  关键词在句中，先果后因
                4 - (head, relation, tail) & (tail, relation, head)  (1) argument words 中的一些关系
                5 - (head, relation, tail) & (tail, relation, head)  (2) punctuations

        '''

        batch_size = len(edges)
        argument_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        punct_graph = torch.zeros(
            (batch_size, n_nodes, n_nodes))  # NOTE: the diagonal should be assigned 0 since is acyclic graph.
        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if edge_value == 1:  # (relation, head, tail)  关键词在句首. Note: not used in graph_version==4.0.
                    try:
                        argument_graph[b, i + 1, i + 2] = 1
                    except Exception:
                        pass
                elif edge_value == 2:  # (head, relation, tail)  关键词在句中，先因后果. Note: not used in graph_version==4.0.
                    argument_graph[b, i, i + 1] = 1
                elif edge_value == 3:  # (tail, relation, head)  关键词在句中，先果后因. Note: not used in graph_version==4.0.
                    argument_graph[b, i + 1, i] = 1
                elif edge_value == 4:  # (head, relation, tail) & (tail, relation, head) ON ARGUMENT GRAPH
                    argument_graph[b, i, i + 1] = 1
                    argument_graph[b, i + 1, i] = 1
                elif edge_value == 5:  # (head, relation, tail) & (tail, relation, head) ON PUNCTUATION GRAPH
                    try:
                        punct_graph[b, i, i + 1] = 1
                        punct_graph[b, i + 1, i] = 1
                    except Exception:
                        pass
        return argument_graph.to(device), punct_graph.to(device)


    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: torch.LongTensor,

                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,

                argument_bpe_ids: torch.LongTensor,
                domain_bpe_ids: torch.LongTensor,
                punct_bpe_ids: torch.LongTensor,

                labels: torch.LongTensor,
                token_type_ids: torch.LongTensor = None,
                ) -> Tuple:

        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        flat_passage_mask = passage_mask.view(-1, passage_mask.size(-1)) if passage_mask is not None else None
        flat_question_mask = question_mask.view(-1, question_mask.size(-1)) if question_mask is not None else None

        flat_argument_bpe_ids = argument_bpe_ids.view(-1, argument_bpe_ids.size(-1)) if argument_bpe_ids is not None else None
        flat_domain_bpe_ids = domain_bpe_ids.view(-1, domain_bpe_ids.size(-1)) if domain_bpe_ids is not None else None  
        flat_punct_bpe_ids = punct_bpe_ids.view(-1, punct_bpe_ids.size(-1)) if punct_bpe_ids is not None else None

        bert_outputs = self.roberta(flat_input_ids, attention_mask=flat_attention_mask, token_type_ids=None)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]  

         

        new_punct_id = self.max_rel_id + 1
        new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 4. for incorporating with argument_bpe_ids.
        _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-3: arg, 4:punct.
        overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()
        flat_all_bpe_ids = _flat_all_bpe_ids * (1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
        assert flat_argument_bpe_ids.max().item() <= new_punct_id

        
        # span_mask: (bsz x n_choices, n_nodes)
        # edges: list[list[int]]
        # node_in_seq_indices: list[list[list[int]]]
        encoded_spans, span_mask, edges, node_in_seq_indices, encoded_edges, encoded_sents = self.split_into_spans_9(sequence_output,
                                                                                                        flat_attention_mask,
                                                                                                        flat_all_bpe_ids)

        # Basic fusion of pooled output and sequence average (no discourse)
        if self.use_discourse == False:
            total_output = torch.cat((pooled_output,torch.mean(sequence_output,dim=1)),dim=1)

        # Apply transformer to span-level embeddings 
        attention_output = self._transformer_block(encoded_sents)
        atom_output = torch.mean(attention_output,dim=1)

        # Concatenate all representation vectors
        total_output = torch.cat((pooled_output, torch.mean(sequence_output,dim=1), atom_output),dim=1)
    
         # Final classification head
        linear_prj = self._transformer_linear_projection(total_output)
        linear_prj = self._transformer_dropout(linear_prj)
        logits = self._transformer_classifier(linear_prj)

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)  
        outputs = (reshaped_logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs
