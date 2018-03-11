from __future__ import absolute_import
from unittest import TestCase
from .setup_maplike import setup_maplike


class test_MapLike(TestCase):
    def setUp(self):
        self.maplike, self.true_params = setup_maplike()
        self.data = list(self.maplike.split_data())
        return

    def test_array_shapes(self):
        for (mean, var) in self.data:
            covar_shape = self.maplike.get_amplitude_covariance(var, self.true_params).shape
            print(covar_shape)
        return
