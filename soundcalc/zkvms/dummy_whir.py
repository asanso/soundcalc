from __future__ import annotations

from soundcalc.zkvms.whir_based_vm import WHIRBasedVM, WHIRBasedVMConfig


from ..common.fields import *


class DummyWHIRPreset:
    @staticmethod
    def default() -> "DummyWHIR":
        """
        A dummy zkVM using WHIR for testing WHIR
        """

        name = "DummyWHIR"
        hash_size_bits = 256
        log_inv_rate = 1 # rate 1/2
        num_iterations = 5
        folding_factor = 4
        field = GOLDILOCKS_2
        log_degree = 23
        constraint_degree = 1
        num_queries = [80,35,22,12,9]
        num_ood_samples = [2,2,2,2]


        cfg = WHIRBasedVMConfig(
            name=name,
            hash_size_bits=hash_size_bits,
            log_inv_rate=log_inv_rate,
            num_iterations=num_iterations,
            folding_factor=folding_factor,
            field=field,
            log_degree=log_degree,
            constraint_degree=constraint_degree,
            num_queries=num_queries,
            num_ood_samples=num_ood_samples
        )
        return WHIRBasedVM(cfg)
