from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid: bool
    V: int
    r: int
    E_total: int
    E_expected: int

    def __str__(self):
        status = "PASS" if self.is_valid else "FAIL"
        return (
            f"[{status}] E_total={self.E_total}, "
            f"expected 3*{self.V} - {self.r} - 3 = {self.E_expected}"
        )

class Validator:
    """
    Checks the Triangulated Disk Identity for Appel-Haken configurations:

        E_total = 3V - r - 3

    Parameters (all integers):
        V           -- total vertices in the configuration, including the r ring vertices
        E_internal  -- edges strictly between non-ring vertices
        E_attachment-- edges that involve at least one ring vertex (ring-to-ring and
                       ring-to-interior edges both count here)
        r           -- ring size (number of boundary vertices)

    Any fully triangulated disk must satisfy this identity (derived from Euler's
    formula for planar graphs). Any extraction that violates it must be rejected
    or sent to the HITL correction UI.
    """

