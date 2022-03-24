import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from skills.ensemble.criterion import batched_L_divergence
from skills.ensemble.attention import Attention
from skills.models.q_function import LinearQFunction, compute_value_loss


class ValueEnsemble():

    def __init__(self, 
        device,
        embedding_output_size=64, 
        gru_hidden_size=128,
        learning_rate=2.5e-4,
        discount_rate=0.9,
        num_modules=8, 
        num_output_classes=18,
        plot_dir=None,
        verbose=False,):
        
        self.num_modules = num_modules
        self.num_output_classes = num_output_classes
        self.device = device
        self.gamma = discount_rate
        self.verbose = verbose

        self.embedding = Attention(
            embedding_size=embedding_output_size, 
            num_attention_modules=self.num_modules, 
            plot_dir=plot_dir
        ).to(self.device)

        self.recurrent_memory = nn.GRU(
            input_size=embedding_output_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
        ).to(self.device)

        self.q_networks = nn.ModuleList(
            [LinearQFunction(in_features=gru_hidden_size, n_actions=num_output_classes) for _ in range(self.num_modules)]
        ).to(self.device)
        self.target_q_networks = deepcopy(self.q_networks)
        self.target_q_networks.eval()

        self.optimizer = optim.Adam(
            list(self.embedding.parameters()) + list(self.q_networks.parameters()),
            learning_rate,
        )

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.q_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))

    def load(self, path):
        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.q_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))

    def train(self, batch, update_target_network=False, plot_embedding=False):
        """
        update both the embedding network and the value network by backproping
        the sumed divergence and q learning loss
        """
        self.embedding.train()
        self.q_networks.train()

        batch_states = batch['state']
        batch_actions = batch['action']
        batch_rewards = batch['reward']
        batch_next_states = batch['next_state']
        batch_dones = batch['is_state_terminal']

        loss = 0

        # divergence loss
        state_embeddings = self.embedding(batch_states, return_attention_mask=False, plot=plot_embedding)  # (batch_size, num_modules, embedding_size)
        state_embeddings, _ = self.recurrent_memory(state_embeddings)  # (batch_size, num_modules, gru_out_size)
        l_div = batched_L_divergence(state_embeddings)
        loss += l_div

        # q learning loss
        td_losses = np.zeros((self.num_modules,))
        next_state_embeddings = self.embedding(batch_next_states, return_attention_mask=False)
        next_state_embeddings, _ = self.recurrent_memory(next_state_embeddings)

        for idx in range(self.num_modules):

            # predicted q values
            state_attention = state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
            batch_pred_q_all_actions = self.q_networks[idx](state_attention)  # (batch_size, num_actions)
            batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)  # (batch_size,)

            # target q values 
            with torch.no_grad():
                next_state_attention = next_state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
                batch_next_state_q_all_actions = self.target_q_networks[idx](next_state_attention)  # (batch_size, num_actions)
                next_state_values = batch_next_state_q_all_actions.max  # (batch_size,)
                batch_q_target = batch_rewards + self.gamma * (1-batch_dones) *  next_state_values # (batch_size,)
            
            # loss
            td_loss = compute_value_loss(batch_pred_q, batch_q_target, clip_delta=True, batch_accumulator="mean")
            loss += td_loss
            if self.verbose: td_losses[idx] = td_loss.item()
    
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if update_target_network:
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())
            print(f"updated target network by hard copy")

        # logging
        if self.verbose:
            # for idx in range(self.num_modules):
            #     print("\t - Value {}: loss {:.6f}".format(idx, td_losses[idx]))
            print(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")

        self.embedding.eval()
        self.q_networks.eval()
    
    def predict_actions(self, state):
        """
        given a state, each one in the ensemble predicts an action
        """
        self.embedding.eval()
        self.q_networks.eval()
        with torch.no_grad():
            embeddings = self.embedding(state, return_attention_mask=False).detach()
            embeddings, _ = self.recurrent_memory(embeddings)

            actions = np.zeros(self.num_modules, dtype=np.int)
            for idx in range(self.num_modules):
                attention = embeddings[:,idx,:]
                q_vals = self.q_networks[idx](attention)
                actions[idx] = q_vals.greedy_actions

        return actions

    def get_attention(self, x):
        self.embedding.eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, return_attention_mask=True).detach()
        return atts
