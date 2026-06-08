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
    """Checks the Triangulated Disk Identity for Appel-Haken configurations.

    Every fully triangulated disk satisfies E_total = 3V - r - 3, derived
    from Euler's formula for planar graphs. Any extraction that violates it
    must be rejected or sent to the HITL correction UI.
    """

    def check(self, V: int, E_internal: int, E_attachment: int, r: int) -> ValidationResult:
        """Check whether a configuration satisfies the Triangulated Disk Identity.

        Args:
            V: Total vertices in the configuration, including the r ring vertices.
            E_internal: Edges strictly between non-ring vertices.
            E_attachment: Edges involving at least one ring vertex (ring-to-ring
                and ring-to-interior edges both count).
            r: Ring size (number of boundary vertices).

        Returns:
            ValidationResult with is_valid=True iff E_total == 3V - r - 3.
        """
        E_total = E_internal + E_attachment
        E_expected = 3 * V - r - 3
        return ValidationResult(
            is_valid=(E_total == E_expected),
            V=V,
            r=r,
            E_total=E_total,
            E_expected=E_expected,
        )

