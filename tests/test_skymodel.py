from unittest import TestCase
from bfore import SkyModel
import numpy as np

class test_SkyModel(TestCase):
    def setUp(self):
        components = ["cmb", "dustmbb", "syncpl"]
        self.skymodel = SkyModel(components)
        return

    def test_param_names(self):
        param_names = set(["nu_ref_d", "beta_d", "T_d", "nu_ref_s", "beta_s"])
        self.assertEqual(set(self.skymodel.get_param_names()), param_names)
        return

    def test_ncomps(self):
        self.assertEqual(self.skymodel.ncomps, 3)
        return

    def test_fnu(self):
        params = {
            'nu_ref_d': 353.,
            'beta_d': 1.6,
            'T_d': 20.,
            'nu_ref_s': 23.,
            'beta_s': -3.,
            }

        nfreqs = 10
        nus = np.array(np.logspace(1, 3, nfreqs))
        f_matrix = self.skymodel.fnu(nus, params)
        self.assertEqual(f_matrix.shape, (3, 3, 10))
        return
