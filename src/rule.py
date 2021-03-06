'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-02-22 08:40:56
LastEditors: Yinan Mei
LastEditTime: 2022-07-26 15:57:28
'''

import numpy as np

class EditingRule(object):
    """Class of Editing Rule"""
    def __init__(self, lhs_attrs: dict, pattern: dict, x_attrs:list) -> None:
        """Init Function

        Args:
            lhs_attrs (dict): Left Hand Side attributes
            pattern (dict): patterns
            x_attrs (list): input attributes in LHS or patterns
        """
        self.lhs_attrs = dict()
        self.pattern = dict()
        for attr in x_attrs:
            if attr in lhs_attrs:
                self.lhs_attrs[attr] = lhs_attrs[attr]
            if attr in pattern:
                self.pattern[attr] = pattern[attr]
        self.attr_set = set(lhs_attrs.items())
        self.wildcard_attrs = list(set(self.lhs_attrs.keys()).difference(set(self.pattern.keys())))
        self.wildcard_attrs_m = [lhs_attrs[attr] for attr in self.wildcard_attrs]
        # encode
        self.enc = self.encode()

    def encode(self):
        """encode the rule into str format

        Returns:
            str: editing rule str encoding
        """
        lhs_attr_str = str(self.lhs_attrs)
        pattern_str = str(self.pattern)
        return f"{lhs_attr_str} || {pattern_str}"

    def dominate(self, rule):
        """check whether it dominates the rule

        Args:
            rule (EditingRule): the to-check rule

        Returns:
            bool: if dominat the rule, then return True. Else, False
        """
        attr_contain = self.attr_set.issubset(rule.attr_set)
        pattern_contain = True
        for attr, value in self.pattern.items():
            v = rule.pattern.get(attr, "notFound")
            if value != v:
                pattern_contain = False
                break
        if attr_contain and pattern_contain:
            return True
        return False

    def __eq__(self, __o: object) -> bool:
        if self.lhs_attrs == __o.lhs_attrs and self.pattern == __o.pattern:
            return True
        return False

    def __hash__(self) -> int:
        return hash(self.enc)

    def __str__(self) -> str:
        return self.enc

class RuleParser(object):
    def __init__(self, encoders, match, x_attrs) -> None:
        """Parse rule encoding to EditingRule

        Args:
            encoders (OneHotEncoder): encoders from input data
            match (dict): schema match
            x_attrs (list): input attributes in LHS or patterns
        """
        super().__init__()
        self.encoders = encoders # dict: {col_name:OneHotEncoder}
        self.x_attrs = x_attrs
        self.cols = encoders.keys()
        self.ix_cond_dict = dict()
        self.lhs_ranges = dict()
        self.dom_ranges = dict()
        start_ix = 0
        # s_a
        for attr, attr_m_cands in match.items():
            attr_ix = start_ix
            for attr_m in attr_m_cands:
                self.ix_cond_dict[start_ix] = [(attr, attr_m), "lhs"]
                start_ix += 1
            self.lhs_ranges[attr] = (attr_ix, start_ix)
        # s_p
        for attr, encoder in self.encoders.items():
            dom_size = len(encoder.categories_[0])
            print(f"[Attr]={attr}, [Domain Size]={dom_size}")
            self.dom_ranges[attr] = (start_ix, start_ix+dom_size)
            for step, v in enumerate(encoder.categories_[0]):
                v = str(v)
                self.ix_cond_dict[start_ix] = [(attr, v), "pattern"]
                start_ix += 1
        self.enc_dim = start_ix
        self.enc_rule_dict = dict()
    
    def get_enc_dim(self):
        return self.enc_dim

    def encoding_to_rule(self, encoding):
        """transform the rule (state) encoding to the rule

        Args:
            encoding (np.array): rule (state) encoding 

        Returns:
            EditingRule: the transformed rule
        """
        enc_key = encoding.tobytes()
        if enc_key in self.enc_rule_dict:
            return self.enc_rule_dict[enc_key]
        ixs = np.where(encoding==1)[0]
        lhs_attrs, pattern = dict(), dict()
        for ix in ixs:
            cond, cond_type = self.ix_cond_dict[ix]
            if cond_type == "lhs":
                lhs_attrs[cond[0]] = cond[1]
            else:
                pattern[cond[0]] = cond[1]
        rule = EditingRule(lhs_attrs, pattern, self.x_attrs)
        self.enc_rule_dict[enc_key] = rule
        return rule

    def get_mask_from_state(self, state):
        """Get the mask according to the state encoding

        Args:
            state (np.array): rule (state) encoding 

        Returns:
            np.array: mask vector
        """
        # mask: [0,1,1,1,0,0,0,0,0,1]
        # the last dimension identify whether we stop rule refinement.
        # thus, the last dimension is never masked.
        mask = np.ones(self.enc_dim+1, dtype=int)
        if sum(state) == 0:
            for attr, (left, right) in self.dom_ranges.items():
                mask[left:right] = 0
        else:
            cond_ixs = np.where(state==1)[0]
            for ix in cond_ixs:
                cond, cond_type = self.ix_cond_dict[ix]
                if cond_type == "lhs":
                    left, right = self.lhs_ranges[cond[0]]
                    mask[left:right] = 0
                else:
                    left, right = self.dom_ranges[cond[0]]
                    mask[left:right] = 0
        return mask