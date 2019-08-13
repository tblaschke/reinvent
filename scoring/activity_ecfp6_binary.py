# coding=utf-8

import numpy as np
from rdkit.Chem import AllChem

from .activity_ecfp6_count import activity_ecfp6_count


class activity_ecfp6_binary(activity_ecfp6_count):
    """Scores based on an binary ECFP6 classifier for activity"""

    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in enumerate(fp):
                nfp[i, idx] = v
        return nfp
