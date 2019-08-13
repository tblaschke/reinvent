# coding=utf-8

import logging
import pickle
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import utils


class activity_ecfp6_count(object):
    """Scores based on an count ECFP6 classifier for activity"""

    def __init__(self, clf_path: utils.FilePath):
        logging.debug(clf_path)
        self.clf_path = clf_path
        with open(clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = activity_ecfp6_count.fingerprints_from_mols(valid_mols)
        activity_score = self.clf.predict_proba(fps)[:, 1]

        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, activity_score):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}

    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp

    def __reduce__(self):
        """
        :return: A tuple with the constructor and its arguments. Used to reinitialize the object for pickling
        """
        return activity_ecfp6_count, (self.clf_path,)
