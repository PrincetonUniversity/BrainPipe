import importlib


class NetCollector(object):
    """
    Collects the different neural network architectures into a library
    """
    module_list = dict()

    def add_module(cls, module, identifier):
        NetCollector.module_list[identifier] = module

    def get_module(cls, identifier):
        importlib.import_module("neurotorch.nets." + identifier)

        try:
            return NetCollector.module_list[identifier]
        except KeyError:
            raise ValueError("{} could not be found in the" +
                             " NetCollector".format(identifier))
