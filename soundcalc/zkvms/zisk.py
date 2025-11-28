from __future__ import annotations

from soundcalc.zkvms.fri_based_vm import FRIBasedCircuit, FRIBasedVM, FRIBasedVMConfig

from ..common.fields import *


def _parse_fri_folding_arities(fri_fa_str: str) -> tuple[list[int], int]:
    """
    Parse FRI folding arity string like "23-19-15-11-8-5" into folding factors and early stop degree.

    The sequence represents log2 of domain sizes at each step.
    E.g., "23-19-15-11-8-5" means:
      - 23→19: fold by 2^(23-19) = 16
      - 19→15: fold by 2^4 = 16
      - 15→11: fold by 2^4 = 16
      - 11→8: fold by 2^3 = 8
      - 8→5: fold by 2^3 = 8
      - Final 5: early_stop_degree = 2^5 = 32
    """
    values = [int(x) for x in fri_fa_str.split('-')]
    folding_factors = []
    for i in range(len(values) - 1):
        fold_log = values[i] - values[i + 1]
        folding_factors.append(1 << fold_log)
    early_stop_degree = 1 << values[-1]
    return folding_factors, early_stop_degree


def _make_circuit(name, bits, bf, d, fixed, stage1, pols, queries, opens, fri_fa) -> FRIBasedCircuit:
    """Factory function to create a circuit from table parameters."""
    FRI_folding_factors, FRI_early_stop_degree = _parse_fri_folding_arities(fri_fa)

    return FRIBasedCircuit(FRIBasedVMConfig(
        name=name,
        trace_length=1 << bits,
        rho=1 / (1 << bf),
        AIR_max_degree=d,
        num_columns=fixed + stage1,
        batch_size=pols,
        num_queries=queries,
        max_combo=opens,
        FRI_folding_factors=FRI_folding_factors,
        FRI_early_stop_degree=FRI_early_stop_degree,
        field=GOLDILOCKS_3,
        hash_size_bits=256,
        power_batching=True,
        grinding_query_phase=0,
    ))


# Base ZisK circuits
# Columns: (name, bits, bf, d, fixed, stage1, pols, queries, opens, fri_fa)
#
# See https://github.com/ethereum/soundcalc/issues/18 for more details:
#
#
# Aligned for better reading:
#                              bits bf  d  fix  stg1  pols  qry opn  fri_fa
ZISK_BASE_CIRCUITS = [
    ("Main",                    22,  1, 3,   3,   38,   61, 128,  3, "23-19-15-11-8-5"),
    ("Rom",                     22,  1, 2,   1,    1,   18, 128,  3, "23-19-15-11-8-5"),
    ("Mem",                     22,  1, 3,   2,   13,   29, 128,  3, "23-19-15-11-8-5"),
    ("RomData",                 21,  1, 3,   2,    6,   19, 128,  3, "22-18-14-11-8-5"),
    ("InputData",               21,  1, 3,   2,    9,   27, 128,  3, "22-18-14-11-8-5"),
    ("MemAlign",                21,  1, 3,   2,   29,   59, 128,  3, "22-18-14-11-8-5"),
    ("MemAlignByte",            22,  1, 3,   1,   16,   25, 128,  3, "23-19-15-11-8-5"),
    ("MemAlignReadByte",        22,  1, 3,   1,   10,   18, 128,  3, "23-19-15-11-8-5"),
    ("MemAlignWriteByte",       22,  1, 3,   1,   14,   23, 128,  3, "23-19-15-11-8-5"),
    ("Arith",                   21,  1, 3,   1,   44,   64, 128,  3, "22-18-14-11-8-5"),
    ("Binary",                  22,  1, 3,   1,   39,   49, 128,  3, "23-19-15-11-8-5"),
    ("BinaryAdd",               22,  1, 3,   1,   10,   18, 128,  3, "23-19-15-11-8-5"),
    ("BinaryExtension",         22,  1, 3,   1,   29,   40, 128,  3, "23-19-15-11-8-5"),
    ("Add256",                  20,  1, 3,   1,   47,   69, 128,  3, "21-17-13-9-5"),
    ("ArithEq",                 20,  1, 3,   2,   39,  434, 128, 36, "21-17-13-9-5"),
    ("ArithEq384",              20,  1, 3,   2,   33,  536, 128, 54, "21-17-13-9-5"),
    ("Keccakf",                 16,  1, 3,   2, 2137, 4065, 128, 26, "17-13-9-5"),
    ("Sha256f",                 18,  1, 3,   2,  102, 1265, 128, 87, "19-15-11-8-5"),
    ("SpecifiedRanges",         20,  1, 3,  34,   33,   88, 128,  3, "21-17-13-9-5"),
    ("VirtualTable0",           20,  1, 3, 100,   16,  129, 128,  3, "21-17-13-9-5"),
    ("VirtualTable1",           20,  1, 3, 145,   16,  174, 128,  3, "21-17-13-9-5"),
]

# Recursive Proving Circuits
#                              bits bf  d  fix  stg1  pols  qry opn  fri_fa
ZISK_RECURSIVE_CIRCUITS = [
    ("ArithEq Compressor",      18,  2, 3,  45,   36,  238,  64,  4, "20-16-12-8-5"),
    ("ArithEq384 Compressor",   18,  2, 3,  45,   36,  238,  64,  4, "20-16-12-8-5"),
    ("Keccakf Compressor",      21,  2, 3,  45,   36,  238,  64,  4, "23-19-15-11-8-5"),
    ("Sha256f Compressor",      19,  2, 3,  45,   36,  238,  64,  4, "21-17-13-9-5"),
    ("VirtualTable1 Compressor",18,  2, 3,  45,   36,  238,  64,  4, "20-16-12-8-5"),
    ("Recursive1",              17,  3, 3,  45,   36,  243,  43,  4, "20-16-12-8-5"),
    ("Recursive2",              17,  3, 3,  45,   36,  243,  43,  4, "20-16-12-8-5"),
    ("Final",                   16,  4, 3,  45,   42,  249,  32,  4, "20-15-10"),
]


class ZiskPreset:
    @staticmethod
    def default() -> FRIBasedVM:
        """
        Create a ZisK VM with multiple circuits.

        For ZisK, we populate the parameters from the following data:
        https://github.com/ethereum/soundcalc/issues/18
        """
        circuits = [_make_circuit(*row) for row in ZISK_BASE_CIRCUITS]
        circuits += [_make_circuit(*row) for row in ZISK_RECURSIVE_CIRCUITS]
        return FRIBasedVM(name="ZisK", circuits=circuits)


if __name__ == "__main__":
    import unittest

    class TestParseFriFoldingArities(unittest.TestCase):
        def test_main_circuit_fri_fa(self):
            # 23-19-15-11-8-5: folds by 16,16,16,8,8 then stops at degree 32
            factors, early_stop = _parse_fri_folding_arities("23-19-15-11-8-5")
            self.assertEqual(factors, [16, 16, 16, 8, 8])
            self.assertEqual(early_stop, 32)

        def test_short_fri_fa(self):
            # 17-13-9-5: folds by 16,16,16 then stops at degree 32
            factors, early_stop = _parse_fri_folding_arities("17-13-9-5")
            self.assertEqual(factors, [16, 16, 16])
            self.assertEqual(early_stop, 32)

        def test_final_circuit_fri_fa(self):
            # 20-15-10: folds by 32,32 then stops at degree 1024
            factors, early_stop = _parse_fri_folding_arities("20-15-10")
            self.assertEqual(factors, [32, 32])
            self.assertEqual(early_stop, 1024)

    unittest.main()
