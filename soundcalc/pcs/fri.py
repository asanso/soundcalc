from __future__ import annotations

from dataclasses import dataclass
from math import log2

from soundcalc.common.fields import FieldParams
from soundcalc.common.fri import get_FRI_proof_size_bits, get_num_FRI_folding_rounds
from soundcalc.common.utils import get_bits_of_security_from_error
from soundcalc.pcs.pcs import PCS
from soundcalc.proxgaps.proxgaps_regime import ProximityGapsRegime


@dataclass(frozen=True)
class FRIConfig:
    """
    Configuration for FRI PCS.
    """

    # The output length of the hash function that is used in bits
    # Note: this concerns the hash function used for Merkle trees
    hash_size_bits: int

    # The code rate ρ
    rho: float
    # Domain size before low-degree extension (i.e. trace length)
    trace_length: int
    # Preset field parameters (contains p, ext_size, F)
    field: FieldParams
    # Number of functions appearing in the batched-FRI
    # This can be greater than `num_columns`: some zkEVMs have to use "segment polynomials" (aka "composition polynomials")
    batch_size: int
    # Boolean flag to indicate if batched-FRI is implemented using coefficients
    # r^0, r^1, ... r^{batch_size-1} (power_batching = True) or
    # 1, r_1, r_2, ... r_{batch_size - 1} (power_batching = False)
    power_batching: bool
    # Number of FRI queries
    num_queries: int

    # FRI folding factor: one factor per FRI round
    FRI_folding_factors: list[int]
    # Many zkEVMs don't FRI fold until the final poly is of degree 1. They instead stop earlier.
    # This is the degree they stop at (and it influences the number of FRI folding rounds).
    FRI_early_stop_degree: int

    # Proof of Work grinding compute during FRI query phase (expressed in bits of security)
    grinding_query_phase: int


class FRI(PCS):
    """
    FRI Polynomial Commitment Scheme.
    """

    def __init__(self, config: FRIConfig):
        self.hash_size_bits = config.hash_size_bits
        self.rho = config.rho
        self.trace_length = config.trace_length
        self.batch_size = config.batch_size
        self.power_batching = config.power_batching
        self.num_queries = config.num_queries
        self.FRI_folding_factors = config.FRI_folding_factors
        self.FRI_early_stop_degree = config.FRI_early_stop_degree
        self.grinding_query_phase = config.grinding_query_phase

        # Negative log of rate
        self.k = int(round(-log2(self.rho)))
        # Log of trace length
        self.h = int(round(log2(self.trace_length)))
        # Domain size, after low-degree extension
        self.D = self.trace_length / self.rho

        # Extract field parameters from the preset field
        # Extension field degree (e.g., ext_size = 2 for Fp²)
        self.field_extension_degree = config.field.field_extension_degree
        # Extension field size |F| = p^{ext_size}
        self.field = config.field
        self.field_size = config.field.F

        # Compute number of FRI folding rounds
        self.FRI_rounds_n = get_num_FRI_folding_rounds(
            witness_size=int(self.D),
            field_extension_degree=int(self.field_extension_degree),
            folding_factors=self.FRI_folding_factors,
            fri_early_stop_degree=int(self.FRI_early_stop_degree),
        )

    def get_pcs_security_levels(self, regime: ProximityGapsRegime) -> dict[str, int]:
        """
        Returns PCS-specific security levels for a given regime.
        """
        bits = {}

        # Compute FRI errors for batching
        bits["batching"] = get_bits_of_security_from_error(self._get_batching_error(regime))

        # Compute FRI error for folding / commit phase
        FRI_rounds = self.FRI_rounds_n
        for i in range(FRI_rounds):
            bits[f"commit round {i+1}"] = get_bits_of_security_from_error(self._get_commit_phase_error(i, regime))

        # Compute FRI error for query phase
        bits["query phase"] = get_bits_of_security_from_error(self._get_query_phase_error(regime))

        return bits

    def _get_batching_error(self, regime: ProximityGapsRegime) -> float:
        """
        Returns the error due to the batching step. This depends on whether batching is done
        with powers or with random coefficients.
        """
        rate = self.rho
        dimension = self.trace_length

        if self.power_batching:
            epsilon = regime.get_error_powers(rate, dimension, self.batch_size)
        else:
            epsilon = regime.get_error_linear(rate, dimension)

        return epsilon

    def _get_commit_phase_error(self, round: int, regime: ProximityGapsRegime) -> float:
        """
        Returns the error from a round of the commit phase.
        """
        rate = self.rho

        acc_folding_factor = 1
        for i in range(round + 1):
            acc_folding_factor *= self.FRI_folding_factors[i]

        dimension = self.trace_length / acc_folding_factor

        epsilon = regime.get_error_powers(rate, dimension, self.FRI_folding_factors[round])

        return epsilon

    def _get_query_phase_error(self, regime: ProximityGapsRegime) -> float:
        """
        Returns the error from the FRI query phase, including grinding.
        """
        rate = self.rho
        dimension = self.trace_length

        # error is (1-pp)^number of queries
        pp = regime.get_proximity_parameter(rate, dimension)
        epsilon = (1 - pp) ** self.num_queries

        # add grinding
        epsilon *= 2 ** (-self.grinding_query_phase)

        return epsilon

    def get_proof_size_bits(self) -> int:
        """
        Returns an estimate for the proof size, given in bits.
        """
        # XXX (BW): note that it is not clear that this is the
        # proof size for every zkEVM we can think of
        # XXX (BW): we should probably also add something for the OOD samples and plookup, lookup etc.
        return get_FRI_proof_size_bits(
            hash_size_bits=self.hash_size_bits,
            field_size_bits=self.field.extension_field_element_size_bits(),
            batch_size=self.batch_size,
            num_queries=self.num_queries,
            domain_size=int(self.D),
            folding_factors=self.FRI_folding_factors,
            rate=self.rho
        )

    def get_best_attack_security(self) -> int | None:
        """
        Security level based on the best known attack.

        Currently, this is based on the toy problem also known as "ethSTARK conjecture".
        It uses the simplest and probably the most optimistic soundness analysis.

        Note: this is just for historical reference, the toy problem security has no real meaning.

        This is Regime 1 from the RISC0 Python calculator
        """
        # FRI errors under the toy problem regime
        # see "Toy problem security" in §5.9.1 of the ethSTARK paper
        commit_phase_error = 1 / self.field_size
        query_phase_error_without_grinding = self.rho ** self.num_queries
        # Add bits of security from grinding (see section 6.3 in ethSTARK)
        query_phase_error_with_grinding = query_phase_error_without_grinding * 2 ** (-self.grinding_query_phase)

        final_error = commit_phase_error + query_phase_error_with_grinding
        final_level = get_bits_of_security_from_error(final_error)

        return final_level

    def get_field(self) -> FieldParams:
        return self.field

    def get_rate(self) -> float:
        return self.rho

    def get_dimension(self) -> int:
        return self.trace_length

    def get_parameter_summary(self) -> str:
        """
        Returns a description of the parameters of the PCS.
        """
        lines = []
        lines.append("")
        lines.append("```")

        params = {
            "hash_size_bits": self.hash_size_bits,
            "rho": self.rho,
            "k = -log2(rho)": self.k,
            "trace_length": self.trace_length,
            "h = log2(trace_length)": self.h,
            "domain_size D = trace_length / rho": self.D,
            "batch_size": self.batch_size,
            "power_batching": self.power_batching,
            "num_queries": self.num_queries,
            "FRI_folding_factors": self.FRI_folding_factors,
            "FRI_early_stop_degree": self.FRI_early_stop_degree,
            "FRI_rounds_n": self.FRI_rounds_n,
            "grinding_query_phase": self.grinding_query_phase,
            "field": self.field.to_string(),
            "field_extension_degree": self.field_extension_degree,
        }

        key_width = max(len(k) for k in params.keys())
        for k, v in params.items():
            lines.append(f"  {k:<{key_width}} : {v}")

        lines.append("```")
        return "\n".join(lines)
