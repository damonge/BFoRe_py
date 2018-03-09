from unittest import TestCase
from ..bfore.components import Component, cmb, syncpl, dustmbb

class test_Component(TestCase):
    def setUp(self):
        self.component_cmb = Component(["cmb"])
        self.component_syncpl = Component(["syncpl"])
        self.component_dustmbb = Component(["dustmbb"])
        return

    def test_parameters(self):
        self.assertEqual(self.component_cmb.get_parameters(), None)
        self.assertEqual(self.component_syncpl.get_parameters(), ["betas"])
        self.assertEqual(self.component_dustmbb.get_parameters(), ["beta_d", "T_d"])
        return

    def test_call(self):
        return

class test_
