from __future__ import annotations
from abc import ABC, abstractmethod


class Circuit(ABC):
    """
    A class modeling a single circuit within a zkVM.
    Each circuit has its own parameters and security analysis.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the circuit."""
        ...

    @abstractmethod
    def get_parameter_summary(self) -> str:
        """Returns a description of the parameters of the circuit."""
        ...

    @abstractmethod
    def get_proof_size_bits(self) -> int:
        """Returns an estimate for the proof size, given in bits."""
        ...

    @abstractmethod
    def get_security_levels(self) -> dict[str, dict[str, int]]:
        """
        Returns a dictionary that maps each regime (i.e., a way of doing security analysis)
        to a dictionary that contains the round-by-round soundness levels.

        It maps from a label that describes the regime (e.g., UDR, JBR in case of FRI) to a
        regime-specific dictionary. Any such regime-specific dictionary is as follows:

        It maps from a label that explains which round it is for to an integer.
        If this integer is, say, k, then it means the error for this round is at
        most 2^{-k}.
        """
        ...


class zkVM(ABC):
    """
    A class modeling a zkVM, which contains one or more circuits.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Returns the name of the zkVM."""
        ...

    @abstractmethod
    def get_circuits(self) -> list[Circuit]:
        """Returns the list of circuits in this zkVM."""
        ...
