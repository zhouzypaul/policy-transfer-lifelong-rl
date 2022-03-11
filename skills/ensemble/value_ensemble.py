import os
from copy import deepcopy
from queue import PriorityQueue

import torch
import torch.nn.functional as F
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
        embedding_learning_rate=1e-4, 
        policy_learning_rate=1e-2, 
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

        self.q_networks = nn.ModuleList(
            [LinearQFunction(in_features=embedding_output_size, n_actions=num_output_classes) for _ in range(self.num_modules)]
        ).to(self.device)
        self.target_q_networks = deepcopy(self.q_networks)
        self.target_q_networks.eval()

        self.embedding_optimizer = optim.SGD(
            self.embedding.parameters(), 
            embedding_learning_rate, 
            momentum=0.95, 
            weight_decay=1e-4
        )
        self.policy_optimisers = []
        for i in range(self.num_modules):
            optimizer = optim.Adam(self.q_networks[i].parameters(), policy_learning_rate)
            self.policy_optimisers.append(optimizer)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.q_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))

    def load(self, path):
        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.q_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))

    def set_policy_train(self):
        self.q_networks.train()

    def set_policy_eval(self):
        self.q_networks.eval()

    def train_embedding(self, batch, epochs, plot_embedding=False):
        """
        update the embedding network using a batch of experiences
        """
        # dataset is a pytorch dataset
        self.embedding.train()
        self.set_policy_eval()

        for _ in range(epochs):
            batch_states = batch['state']
            embedding = self.embedding(batch_states, return_attention_mask=False, plot=plot_embedding)
            if embedding.size()[0] == 0:
                continue
            embedding = embedding.view(embedding.size(0), self.num_modules, -1)

            self.embedding_optimizer.zero_grad()
            l_div = batched_L_divergence(embedding)
            l_div.backward()
            self.embedding_optimizer.step()

            if self.verbose:
                loss_div = l_div.item()

            if self.verbose:
                print(f"LOSS div: {loss_div}")

        self.embedding.eval()
        self.set_policy_eval()

    def train_q_network(self, batch, epochs, update_target_network=False):
        """
        train the q network with a batch of experience
        """
        self.embedding.eval()
        self.set_policy_train()

        for _ in range(epochs):
            avg_loss = np.zeros(self.num_modules)
            batch_states = batch['state']
            batch_actions = batch['action']
            batch_rewards = batch['reward']
            batch_next_states = batch['next_state']
            batch_dones = batch['is_state_terminal']

            state_embeddings = self.embedding(batch_states, return_attention_mask=False)
            next_state_embeddings = self.embedding(batch_next_states, return_attention_mask=False)

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
                loss = compute_value_loss(batch_pred_q, batch_q_target, clip_delta=True, batch_accumulator="mean")
                self.policy_optimisers[idx].zero_grad()
                loss.backward(retain_graph=True)
                self.policy_optimisers[idx].step()
                if self.verbose: avg_loss[idx] += loss.item()

            # update target network
            if update_target_network:
                self.target_q_networks.load_state_dict(self.q_networks.state_dict())
                print(f"updated target network by hard copy")

            if self.verbose:
                for idx in range(self.num_modules):
                    print("\t - Policy {}: loss {:.6f}".format(idx, avg_loss[idx]))
                print("Average across policy: loss = {:.6f}".format(np.mean(avg_loss)))

        self.embedding.eval()
        self.set_policy_eval()
    
    def predict_actions(self, state):
        """
        given a state, each one in the ensemble predicts an action
        """
        self.embedding.eval()
        self.set_policy_eval()
        with torch.no_grad():
            embeddings = self.embedding(state, return_attention_mask=False).detach()

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
        
    def test_embedding(self, dataset, check_top=5):
        self.embedding.eval()
        embedding = np.zeros([])
        class_test = [0]*self.num_modules
        results = [PriorityQueue()]*self.num_modules

        for i, batch in enumerate(dataset):
            
            x, y = batch
            x = x.to(self.device)
            y = y.item()

            query = self.embedding(x, return_attention_mask=False)
            for x in range(self.num_modules):
                current_query = query[:, x, ...].detach().cpu().numpy()           

                if i == 0:
                    embedding = current_query
                    class_test[x] = y
                else:
                    dist = np.sum((embedding - current_query)**2)
                    results[x].put((dist, y))

        for x in range(self.num_modules):
            print("Testing Module: {}".format(x))
            print("Test image class: {}".format(class_test[x]))

            for i in range(check_top):
                _, result_class = results[x].get()
                print("{}) Class: {}".format(i, result_class))
