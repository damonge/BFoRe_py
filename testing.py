from bfore.components import Component
from bfore.skymodel import SkyModel

comp_list = ["cmb", "dustmbb", "syncpl"]

for comp_name in comp_list:
    comp = Component(comp_name)
    print(comp.get_parameters())

skymodel = SkyModel(comp_list)


print(skymodel.param_names)