'''
Descripttion: 
version: 
Author: Yinan Mei
Date: 2022-02-02 15:44:43
LastEditors: Yinan Mei
LastEditTime: 2022-07-26 14:23:20
'''

import random
import pandas as pd

class DirtyDataGenerator(object):
    def __init__(self, data, seed=42) -> None:    
        """Init Function

        Args:
            data (pd.DataFrame): inject errors to the data
            seed (int, optional): random seed for error generation. Defaults to 42.
        """
        super().__init__()
        random.seed(seed)
        self.cols = data.columns
        self.domains = dict()
        for col in self.cols:
            self.domains[col] = list(set(data[col]))

    def replace_value(self, value, col):
        """select a value to replace randomly

        Args:
            value (to-replace value): to-replace value
            col (obj): the column name of the value belong to

        Returns:
            obj: new value
        """
        # avoid endless loop
        for _ in range(10):
            tmp = random.choice(self.domains[col])
            if tmp != value:
                break
        return tmp
    
    def add_typo(self, text, prob=0.1):
        text = str(text)
        """Based On https://github.com/alexyorke/butter-fingers

        Args:
            text (str): input text
            prob (float, optional): Error Prob. Defaults to 0.1.
        
        Returns:
            str: erroneous text
		"""
        keyApprox = {}
        keyApprox['q'] = "qwas"
        keyApprox['w'] = "qweasd"
        keyApprox['e'] = "wersdf"
        keyApprox['r'] = "ertdfg"
        keyApprox['t'] = "rtyfgh"
        keyApprox['y'] = "tyughj"
        keyApprox['u'] = "yuihjk"
        keyApprox['i'] = "uiojkl"
        keyApprox['o'] = "iopkl"
        keyApprox['p'] = "opkl"        
        keyApprox['a'] = "qazds"
        keyApprox['s'] = "qweasdzx"
        keyApprox['d'] = "ersdfxc"
        keyApprox['f'] = "rtdfgcv"
        keyApprox['g'] = "tyfghvb"
        keyApprox['h'] = "yughjbn"
        keyApprox['j'] = "uihjknm"
        keyApprox['k'] = "iojklm"
        keyApprox['l'] = "iopkl"        
        keyApprox['z'] = "asdzx"
        keyApprox['x'] = "sdzxc"
        keyApprox['c'] = "dfxcv"
        keyApprox['v'] = "fgcvb"
        keyApprox['b'] = "ghvbn"
        keyApprox['n'] = "hjbnm"
        keyApprox['m'] = "jklnm"
        keyApprox[' '] = " "   
        for i in range(10):
            keyApprox[str(i)] = "0123456789"
        probOfTypo = int(prob * 100)    
        erroneous_text = ""
        for letter in text:
            lcletter = letter.lower()
            if not lcletter in keyApprox.keys():
                newletter = lcletter
            else:
                if random.choice(range(0, 100)) <= probOfTypo:
                    newletter = random.choice(keyApprox[lcletter])
                else:
                    newletter = lcletter
            # go back to original case
            if not lcletter == letter:
                newletter = newletter.upper()
            erroneous_text += newletter
        if erroneous_text == text:
            # switch position
            erroneous_text = list(erroneous_text)
            ix = random.choice(range(len(erroneous_text)-1))
            erroneous_text[ix], erroneous_text[ix-1] = erroneous_text[ix-1], erroneous_text[ix]
            erroneous_text = "".join(erroneous_text)
        return erroneous_text

    def add_text_noise(self, value, col, prob=0.1):
        """add text noise to the value

        Args:
            value (str): the original text value
            col (str): the corresponding column name
            prob (float, optional): Probability of injecting errors. Defaults to 0.1.

        Returns:
            str: updated value
        """
        if random.random() > prob:
            return value
        method = random.choice(["replace", "missing", "typo"])
        if method == "replace":
            error_value = self.replace_value(value, col)
        elif method == "missing":
            error_value = "Nan"
        else:
            error_value = self.add_typo(value)
        return error_value
    
    def add_category_noise(self, value, col, prob=0.1):
        """add category noise to the value

        Args:
            value (str): the original categorical value
            col (str): the corresponding column name
            prob (float, optional): Probability of injecting errors. Defaults to 0.1.

        Returns:
            str: updated value
        """
        if random.random() > prob:
            return value
        method = random.choice(["replace", "missing"])
        if method == "replace":
            error_value = self.replace_value(value, col)
        else:
            error_value = "Nan"
        return error_value
