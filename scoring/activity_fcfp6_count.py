# coding=utf-8

import numpy as np
from rdkit.Chem import AllChem

from .activity_ecfp6_count import activity_ecfp6_count


class activity_fcfp6_count(activity_ecfp6_count):
    """Scores based on an count FCFP6 classifier for activity."""

    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp
