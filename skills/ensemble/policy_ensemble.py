import os
from copy import deepcopy
from queue import PriorityQueue

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

from skills.ensemble import criterion
from skills.ensemble.attention import Attention
from skills.models.mlp import MLP


class PolicyEnsemble():

    def __init__(self, 
        device,
        num_votes_needed=1,
        embedding_output_size=64, 
        embedding_learning_rate=1e-4, 
        policy_learning_rate=1e-2, 
        discount_rate=0.9,
        num_modules=8, 
        batch_k=4, 
        normalize=False, 
        num_output_classes=18):
        
        self.num_modules = num_modules
        self.batch_k = batch_k
        self.normalize = normalize
        self.num_output_classes = num_output_classes
        self.device = device
        self.num_votes_needed = num_votes_needed
        self.gamma = discount_rate

        self.embedding = Attention(
            embedding_size=embedding_output_size, 
            num_attention_modules=self.num_modules, 
            batch_k=self.batch_k, 
            normalize=self.normalize
        ).to(self.device)

        self.policy_networks = nn.ModuleList(
            [MLP(embedding_output_size, self.num_output_classes) for _ in range(self.num_modules)]
        ).to(self.device)
        self.policy_target_networks = deepcopy(self.policy_networks)
        for i in range(self.num_modules):
            self.policy_target_networks[i].eval()

        self.embedding_optimizer = optim.SGD(
            self.embedding.parameters(), 
            embedding_learning_rate, 
            momentum=0.95, 
            weight_decay=1e-4
        )
        self.policy_optimisers = []
        for i in range(self.num_modules):
            optimizer = optim.Adam(self.policy_networks[i].parameters(), policy_learning_rate)
            self.policy_optimisers.append(optimizer)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.policy_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))

    def load(self, path):
        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.policy_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))

    def set_policy_train(self):
        for i in range(self.num_modules):
            self.policy_networks[i].train()

    def set_policy_eval(self):
        for i in range(self.num_modules):
            self.policy_networks[i].eval()

    def train_embedding(self, dataset, epochs):
        # dataset is a pytorch dataset
        self.embedding.train()
        self.set_policy_eval()

        for epoch in range(epochs):
            loss_div, loss_homo, loss_heter = 0, 0, 0
            counter = 0
            for batch in dataset:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)
                batch_states = torch.from_numpy(np.array([np.array(s) for s in batch_states])).float().to(self.device)
                _, anchors, positives, negatives, _ = self.embedding(batch_states, sampling=True, return_attention_mask=False)
                if anchors.size()[0] == 0:
                    continue
                anchors = anchors.view(anchors.size(0), self.num_modules, -1)
                positives = positives.view(positives.size(0), self.num_modules, -1)
                negatives = negatives.view(negatives.size(0), self.num_modules, -1)

                self.embedding_optimizer.zero_grad()
                l_div, l_homo, l_heter = criterion.criterion(anchors, positives, negatives)
                l = l_div + l_homo + l_heter
                l.backward()
                self.embedding_optimizer.step()

                loss_homo += l_homo.item()
                loss_heter += l_heter.item()
                loss_div += l_div.item()

                counter += 1

            loss_homo /= (counter+1)
            loss_heter /= (counter+1)
            loss_div /= (counter+1)

            print('batches %d\tdiv:%.4f\thomo:%.4f\theter:%.4f'%(counter+1, loss_div, loss_homo, loss_heter))

        self.embedding.eval()
        self.set_policy_eval()

    def train_policy(self, dataset, epochs, update_target_network=False):
        self.embedding.eval()
        self.set_policy_train()

        for epoch in range(epochs):
            avg_loss = np.zeros(self.num_modules)
            count = 0
            for batch in dataset:
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*batch)

                batch_states = torch.from_numpy(np.array([np.array(s) for s in batch_states])).float().to(self.device)
                batch_actions = torch.from_numpy(np.array(batch_actions)).float().to(self.device)
                batch_next_states = torch.from_numpy(np.array([np.array(s) for s in batch_next_states])).float().to(self.device)
                batch_rewards = torch.from_numpy(np.array(batch_rewards)).float().to(self.device)
                batch_dones = torch.from_numpy(np.array(batch_dones)).float().to(self.device)

                state_embeddings = self.embedding(batch_states, sampling=False, return_attention_mask=False)
                next_state_embeddings = self.embedding(batch_next_states, sampling=False, return_attention_mask=False)

                for idx in range(self.num_modules):

                    def get_q_for_action(qvals, actions):
                        """
                        take the q-values for the given actions
                        qvals: (batch_size, num_actions)
                        actions: (batch_size,)
                        """
                        batch_size = len(qvals)
                        return qvals[np.arange(batch_size), actions.long()]

                    # predicted q values
                    state_attention = state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
                    batch_pred_q_all_actions = self.policy_networks[idx](state_attention)  # (batch_size, num_actions)
                    batch_pred_q = get_q_for_action(batch_pred_q_all_actions, batch_actions)  # (batch_size,)

                    # target q values 
                    next_state_attention = next_state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
                    next_state_values = get_q_for_action(self.policy_target_networks[idx](next_state_attention), batch_actions)  # (batch_size,)
                    batch_q_target = batch_rewards + self.gamma * (1-batch_dones) *  next_state_values # (batch_size,)
                    
                    # loss
                    loss = F.smooth_l1_loss(batch_pred_q, batch_q_target)
                    self.policy_optimisers[idx].zero_grad()
                    loss.backward(retain_graph=True)
                    self.policy_optimisers[idx].step()
                    avg_loss[idx] += loss.item()

                count += 1
            avg_loss = avg_loss/count

            # update target network
            if update_target_network:
                self.policy_networks.load_state_dict(self.policy_target_networks.state_dict())
                print(f"updated target network by hard copy")

            for idx in range(self.num_modules):
                print("\t - Policy {}: loss {:.4f}".format(idx, avg_loss[idx]))
            print("Average across policy: loss = {:.4f}".format(np.mean(avg_loss)))

        self.embedding.eval()
        self.set_policy_eval()

    def get_ensemble_vote(self, x):
        pred = self.get_all_votes(x)

        if sum(pred) >= self.num_votes_needed:
            return 1
        else:
            return 0
    
    def predict_actions(self, state):
        """
        given a state, each one in the ensemble predicts an action
        """
        self.embedding.eval()
        self.set_policy_eval()
        state = state.to(self.device).unsqueeze(0)  # add batch dimension
        embeddings = self.embedding(state, sampling=False, return_attention_mask=False).detach()

        actions = np.zeros(self.num_modules, dtype=np.int)
        for idx in range(self.num_modules):
            attention = embeddings[:,idx,:]
            q_vals = self.policy_networks[idx](attention)
            actions[idx] = torch.argmax(q_vals, dim=1).detach().item()
        return actions

    def get_attention(self, x):
        self.embedding.eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, sampling=False, return_attention_mask=True).detach()
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

            query = self.embedding(x, sampling=False, return_attention_mask=False)
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
