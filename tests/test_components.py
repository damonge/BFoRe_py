from unittest import TestCase
from bfore.components import Component

class test_Component(TestCase):
    def setUp(self):
        self.component_cmb = Component("cmb")
        self.component_syncpl = Component("syncpl")
        self.component_dustmbb = Component("dustmbb")
        return

    def test_parameters(self):
        self.assertEqual(self.component_cmb.get_sed_parameters(), [])
        self.assertEqual(self.component_cmb.get_cl_parameters(), ["r_tensor","a_lens"])
        self.assertEqual(self.component_syncpl.get_sed_parameters(), ["nu_ref_s", "beta_s"])
        self.assertEqual(self.component_syncpl.get_cl_parameters(), ["att_s","aee_s","abb_s","ate_s","alpha_s","ell_ref_s"])
        self.assertEqual(self.component_dustmbb.get_sed_parameters(), ["nu_ref_d", "beta_d", "T_d"])
        self.assertEqual(self.component_dustmbb.get_cl_parameters(), ["att_d","aee_d","abb_d","ate_d","alpha_d","ell_ref_d"])
        return
