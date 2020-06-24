#!/usr/bin/env python
__doc__ = """

Sample Specification

Nicholas Turner <nturner@cs.princeton.edu>, 2017
"""

class SampleSpec(object):
    """
    Class specifying the purpose of each volume within a dataset sample

    The three possible classes are: input, label, mask
    Links each label to a mask if it exists
    """

    def __init__(self, sample_keys):

        #Assigning keys to purposes
        (self._inputs,
        self._labels,
        self._masks ) = self._parse_sample_keys(sample_keys)

        self._mask_lookup = self._create_mask_lookup()


    def get_inputs(self):
        return self._inputs


    def get_labels(self):
        return self._labels


    def get_masks(self):
        return self._masks


    def has_mask(self, label_name):
        "Returns whether a label has a matched mask"
        assert label_name in self._mask_lookup, "{} not in lookup".format(label_name)
        return self._mask_lookup[label_name] is not None


    def get_mask_name(self, label_name):
        assert label_name in self._mask_lookup
        return self._mask_lookup[label_name]


    def get_mask_index(self, label_name):
        assert label_name in self._mask_lookup
        return self._masks.index(self._mask_lookup[label_name])


    #================================================
    # Non-interface functions
    #================================================

    def _parse_sample_keys(self, keys):
        """
        Assigns keys to purposes within (inputs, labels, masks) by
        inspecting their names

        All names containing _mask  -> masks
        All names containing _label -> labels
        Else                        -> inputs
        """
        inputs = []; labels = []; masks = []

        for k in keys:
            if   "_mask"  in k:
                masks.append(k)
            elif "_label" in k:
                labels.append(k)
            else:
                inputs.append(k)

        return sorted(inputs), sorted(labels), sorted(masks)


    def _create_mask_lookup(self):
        """
        Creates a lookup dictionary between labels and their respective masks

        Assumes labels and masks are already defined (keys already parsed)
        """

        lookup = {} #init
        for l in self._labels:

            mask_name_candidate1 = l.replace("_label","_mask")
            mask_name_candidate2 = l + "_mask"

            if   mask_name_candidate1 in self._masks:
                lookup[l] = mask_name_candidate1
            elif mask_name_candidate2 in self._masks:
                lookup[l] = mask_name_candidate2
            else:
                lookup[l] = None

        return lookup


