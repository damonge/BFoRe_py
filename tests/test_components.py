from unittest import TestCase
from bfore.components import Component, cmb, syncpl, dustmbb

class test_Component(TestCase):
    def setUp(self):
        self.component_cmb = Component("cmb")
        self.component_syncpl = Component("syncpl")
        self.component_dustmbb = Component("dustmbb")
        return

    def test_parameters(self):
        self.assertEqual(self.component_cmb.get_parameters(), [])
        self.assertEqual(self.component_syncpl.get_parameters(), ["nu_ref_s", "beta_s"])
        self.assertEqual(self.component_dustmbb.get_parameters(), ["nu_ref_d", "beta_d", "T_d"])
        return
