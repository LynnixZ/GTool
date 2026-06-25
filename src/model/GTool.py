import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.gnn import load_gnn_model
from src.utils.mask import batch_mask
from torch_geometric.data import Batch

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
PAD = '<pad>'
IGNORE_INDEX = -100
PROMPT = '\nPlease use the provided tools to solve the problem. Just answer the tool in the order they are provided.'

# Per-family prompt wrapping. The 'llama' entry reproduces GTool's original
# hardcoded Llama-2 behaviour exactly (default for llama / vicuna). Mistral-Instruct
# shares the [INST]...[/INST] format; Qwen3 uses its own <|im_start|> chat markers.
PROMPT_FORMATS = {
    'llama':   dict(bos='<s>[INST]', eos_user='[/INST]', eos='</s>',
                    use_fast=False, pad_token='<pad>', pad_token_id=0),
    'mistral': dict(bos='<s>[INST]', eos_user='[/INST]', eos='</s>',
                    use_fast=True, pad_token=None, pad_token_id=None),
    'qwen':    dict(bos='<|im_start|>user\n',
                    eos_user='<|im_end|>\n<|im_start|>assistant\n', eos='<|im_end|>',
                    use_fast=True, pad_token=None, pad_token_id=None),
}


def resolve_prompt_format(llm_model_name):
    name = (llm_model_name or '').lower()
    if 'mistral' in name:
        return PROMPT_FORMATS['mistral']
    if 'qwen' in name:
        return PROMPT_FORMATS['qwen']
    return PROMPT_FORMATS['llama']  # llama / vicuna (unchanged default)


class GTool(torch.nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        self.mask_prob = args.mask_prob
        self.LLMP_dim = args.LLMP_dim
        self.alpha = args.alpha

        print('Loading LLM')
        # device_map="auto" lets accelerate auto-detect per-GPU memory, so this
        # works on 1..N GPUs of any size. (Original code hardcoded a 2x80GiB
        # max_memory, which crashed when the node had fewer/smaller GPUs.)
        kwargs = {
            "device_map": "auto",
            "revision": "main",
        }
        print(f'Visible CUDA devices: {torch.cuda.device_count()}')

        fmt = resolve_prompt_format(getattr(args, 'llm_model_name', 'llama'))
        self.bos_str = fmt['bos']
        self.eos_user_str = fmt['eos_user']
        self.eos_str = fmt['eos']

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=fmt['use_fast'], trust_remote_code=True, revision=kwargs["revision"])
        if fmt['pad_token'] is not None:
            # Llama/Vicuna: original GTool behaviour (add <pad>, force id 0 = <unk>).
            self.tokenizer.add_special_tokens({"pad_token": fmt['pad_token']})
            self.tokenizer.pad_token_id = fmt['pad_token_id']
        elif self.tokenizer.pad_token is None:
            # Mistral has no pad token; reuse eos so padded positions are harmless.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'

        
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **kwargs
        )

        print("Freezing LLM!")
        for name, param in model.named_parameters():
            param.requires_grad = False

        self.model = model
        print('Finish loading LLM!')

        self.word_embedding = self.model.model.get_input_embeddings()

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=self.word_embedding.embedding_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        
        self.eos_tokens = self.tokenizer(self.eos_str, add_special_tokens=False)
        self.eos_user_tokens = self.tokenizer(PROMPT + self.eos_user_str, add_special_tokens=False)
        self.bos_embeds = self.word_embedding(self.tokenizer(self.bos_str, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        self.pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        self.graph_token_embeds = nn.Parameter(torch.randn(self.word_embedding.embedding_dim)).to(self.model.device)
        self.node_token_embeds = nn.Parameter(torch.randn(self.word_embedding.embedding_dim)).to(self.model.device)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    def EARE_loss(self, x, mask_edge_index, neg_edge_index):

        edge_mask = (mask_edge_index[0], mask_edge_index[1])
        x_i, x_j = x[edge_mask[0]], x[edge_mask[1]]
        indices = torch.randperm(x_i.size(0))[:self.LLMP_dim]
        x_i_p = x_i[indices]
        x_j_p = x_j[indices]

        neg_edge = (neg_edge_index[0], neg_edge_index[1])
        x_i, x_j = x[neg_edge[0]], x[neg_edge[1]]
        indices = torch.randperm(x_i.size(0))[:self.LLMP_dim]
        x_i_n = x_i[indices]
        x_j_n = x_j[indices]

        and_token = self.tokenizer("and", add_special_tokens=False)
        questions = self.tokenizer("are two node vectors encoded by the graph neural network, and determines whether the two nodes have edge connections, and only answers yes or no.", add_special_tokens=False)
        yes_token = self.tokenizer("yes", add_special_tokens=False)
        no_token = self.tokenizer("no", add_special_tokens=False)

        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []  
        for i in range(x_i_p.shape[0]):
            label_input_ids = yes_token.input_ids + self.eos_tokens.input_ids
            input_ids = questions.input_ids + self.eos_user_tokens.input_ids + label_input_ids
            inputs_embeds2 = self.word_embedding(torch.tensor(and_token.input_ids).to(self.model.device))
            inputs_embeds3 = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([self.bos_embeds,self.node_token_embeds.unsqueeze(0), x_i_p[i].unsqueeze(0),self.node_token_embeds.unsqueeze(0), inputs_embeds2, self.node_token_embeds.unsqueeze(0), x_j_p[i].unsqueeze(0), self.node_token_embeds.unsqueeze(0), inputs_embeds3], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids_p = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids_p)        

        for i in range(x_i_n.shape[0]):
            label_input_ids = no_token.input_ids + self.eos_tokens.input_ids
            input_ids = questions.input_ids + self.eos_user_tokens.input_ids + label_input_ids
            inputs_embeds2 = self.word_embedding(torch.tensor(and_token.input_ids).to(self.model.device))
            inputs_embeds3 = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([self.bos_embeds,self.node_token_embeds.unsqueeze(0), x_i_n[i].unsqueeze(0),self.node_token_embeds.unsqueeze(0), inputs_embeds2, self.node_token_embeds.unsqueeze(0), x_j_n[i].unsqueeze(0), self.node_token_embeds.unsqueeze(0), inputs_embeds3], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids_p = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids_p) 

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        
        for i in range(x_i_p.shape[0] + x_i_n.shape[0]):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]


        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)



        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss
    


    def encode_graphs(self, graphs, use_mask=False):
        if use_mask:
            edge_index, mask_edge_index, neg_edge_index = batch_mask(graphs.edge_index, graphs.num_nodes, graphs.num_graphs, self.mask_prob)
            graphs.edge_index = edge_index
            graphs.num_edges = edge_index.shape[1]
            graphs.edge_attr = graphs.edge_attr[:graphs.num_edges]
            
        graphs = graphs.to(self.model.device)
        n_embeds, _ = self.graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr)

        eare_loss = None
        if use_mask:
            mask_edge_index = mask_edge_index.to(self.model.device)
            eare_loss = self.EARE_loss(n_embeds, mask_edge_index, neg_edge_index)

        g_embeds = [n_embeds[i] for i in range(graphs.num_nodes // graphs.num_graphs - 1, graphs.num_nodes, graphs.num_nodes // graphs.num_graphs)]
        g_embeds = torch.stack(g_embeds, dim=0)
        return g_embeds, eare_loss
    


    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        graph_embeds, eare_loss = self.encode_graphs(Batch.from_data_list(samples['graph']), True)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + self.eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + self.eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([self.bos_embeds, self.graph_token_embeds.unsqueeze(0), graph_embeds[i].unsqueeze(0),self.graph_token_embeds.unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids_p = [IGNORE_INDEX] * (inputs_embeds.shape[0]-len(label_input_ids))+label_input_ids
            batch_label_input_ids.append(label_input_ids_p)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i]


        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)


        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        
        return outputs.loss + self.alpha * eare_loss 
    



    def inference(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        eos_user_tokens = self.eos_user_tokens

        graph_embeds,_ = self.encode_graphs(Batch.from_data_list(samples['graph']))

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([self.bos_embeds, self.graph_token_embeds.unsqueeze(0), graph_embeds[i].unsqueeze(0),self.graph_token_embeds.unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([self.pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'], }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
