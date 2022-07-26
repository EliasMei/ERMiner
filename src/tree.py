'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-01-20 07:44:13
LastEditors: Yinan Mei
LastEditTime: 2022-07-26 16:01:53
'''

import numpy as np
from copy import deepcopy

from rule import EditingRule

class Node(object):
    def __init__(self, rule, state, mask, id_, parent=None, children=[]) -> None:
        """RuleNode

        Args:
            rule (EditingRule): the rule contained in the node
            state (np.array): state encoding
            mask (np.array): mask vector
            id_ (int): node id
            parent (Node, optional): its parent. Defaults to None.
            children (list, optional): its children. Defaults to [].
        """
        super().__init__()
        self.rule = rule
        self.state = state.astype(int)
        self.parent = parent
        self.children = children
        self.mask = mask
        self.id_ = id_
    
    def get_state(self):
        return self.state
    
    def get_rule(self):
        return self.rule
    
    def is_finished(self):
        flag = True
        for col, value in self.rule:
            if value == "" or value == "_":
                flag = False
                break
        return flag
    
    def store_cand_counts(self, cand_counts):
        self.cand_counts = cand_counts
    
    def get_cand_counts(self):
        return self.cand_counts
    
    def get_local_mask(self):
        return self.mask # np.array(). 0-1 vectors. 1 denotes that the action is allowed
    

class Tree(object):
    def __init__(self, x_attrs, y_attr, rule_parser) -> None:
        """Rule tree

        Args:
            x_attrs (list): attributes in X and their types
            y_attr (str): target attribute
            rule_parser (RuleParser): the rule parser
        """
        super().__init__()
        self.y_attr = y_attr
        self.depth = 1
        self.states = list()
        self.x_attrs = x_attrs #[("Zip Code", "Discrete"), ("Score", "Continuous"),...]
        self.parser = rule_parser
        init_state = np.zeros(self.parser.enc_dim)
        init_mask = self.parser.get_mask_from_state(init_state)
        self.root = Node(rule=EditingRule({},{},x_attrs), state=init_state, mask=init_mask, id_=0)
        
        self.current_node = self.root
        self.current_queue = []
        self.next_queue = []
        self.leaves = set()
        self.rules = set()
    

    def get_rules(self):
        return self.rules
    
    def get_leaves(self):
        return self.leaves
    
    def update(self, encoding, stop_flag, valid):
        """Given the new rule encoding to update the tree

        Args:
            encoding (np.array): new rule encoding
            stop_flag (bool): whether stop
            valid (bool): whether the rule is valid

        Returns:
            Node: the current node in the tree
        """
        rule = self.parser.encoding_to_rule(encoding)
        assert len(rule.lhs_attrs) > 0
        local_mask = self.parser.get_mask_from_state(encoding)
        node = Node(rule, encoding, local_mask, parent=self.current_node, children=[], id_=len(self.states))
        self.current_node.children.append(node)
        if stop_flag is False:
            self.rules.add(rule)
            self.next_queue.append(node)
        if valid:
            self.leaves.add(rule)
        if self.current_node.get_rule() in self.leaves and valid:
            self.leaves.remove(self.current_node.get_rule())
        self.states.append(encoding)
        if sum(self.get_action_mask(self.current_node)[:-1]) == 0:
            self.current_node = self.get_next_node()
        return self.current_node

    def get_action_mask(self, node):
        """Get the mask for the action according to the environment

        Args:
            node (Node): the current node

        Returns:
            np.array: mask vector
        """
        state = node.get_state()
        local_mask = node.get_local_mask()
        mask = deepcopy(local_mask)
        cond_num = sum(state)
        for s0 in self.states:
            if sum(s0) - cond_num == 1:
                diff = state ^ s0
                ix = np.where(diff==1)[0][0]
                mask[ix] = 0
        return mask

    def get_next_node(self):
        """Traverse to the next node.
        BFS - Level Order. Following lattice structure in rule mining.

        Returns:
            Node: the next node
        """
        if self.current_queue:
            self.current_node = self.current_queue.pop()
        else:
            if self.next_queue:
                self.current_queue = self.next_queue
                self.next_queue = []
                self.current_node = self.current_queue.pop()
            else:
                self.current_node = None
        return self.current_node



