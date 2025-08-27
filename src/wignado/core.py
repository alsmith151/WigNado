from pathlib import Path
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator
from enum import Enum
from typing import Literal

class GenomicRegion(BaseModel):
    """A validated genomic region with chromosome, start, and end coordinates.
    
    Attributes:
        chrom: chromosome/contig name
        start: 0-based inclusive start position  
        end: 0-based exclusive end position (must be > start)
    """
    model_config = {"validate_assignment": True, "extra": "forbid"}

    chrom: str = Field(..., min_length=1)
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)

    @field_validator("end")
    @classmethod
    def validate_interval(cls, v: int, info) -> int:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError(f"End position ({v}) must be greater than start ({info.data['start']})")
        return v

    @property
    def length(self) -> int:
        """Return the length of this genomic region."""
        return self.end - self.start

    def expand_to_left(self, flanking: int) -> "GenomicRegion":
        """Expand the region to the left by a specified number of base pairs."""
        return GenomicRegion(
            chrom=self.chrom,
            start=self.start - flanking,
            end=self.end
        )

    def expand_to_right(self, flanking: int) -> "GenomicRegion":
        """Expand the region to the right by a specified number of base pairs."""
        return GenomicRegion(
            chrom=self.chrom,
            start=self.start,
            end=self.end + flanking
        )

    def expand_from_center(self, flanking: int) -> "GenomicRegion":
        """Expand the region from the center by a specified number of base pairs."""
        center = (self.start + self.end) // 2
        return GenomicRegion(
            chrom=self.chrom,
            start=center - flanking,
            end=center + flanking
        )

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start:,}-{self.end:,}"




class QueryConfig(BaseModel):
    """Behavioral toggles for query-time error handling."""
    error_on_missing_chromosome: bool = Field(
        default=False, description="Raise KeyError if chromosome absent in store"
    )
    error_on_exceeding_chromosome_bounds: bool = Field(
        default=False,
        description="Raise if requested region extends beyond chromosome length",
    )


class ReferencePoint(Enum):
    START = "start"
    CENTER = "center"
    END = "end"
    SCALE = "scale"


class RegionConfig(BaseModel):
    """Region transformation + binning configuration.

    See revised docstring in earlier patch for detailed description.
    """

    mode: Literal["reference-point", "scale-regions"] = "reference-point"
    reference_point: ReferencePoint = ReferencePoint.CENTER
    bp_before: int = Field(0, ge=0, description="Upstream bases (reference-point)")
    bp_after: int = Field(0, ge=0, description="Downstream bases (reference-point)")
    body: int = Field(0, ge=0, description="Scaled body length (scale-regions)")
    upstream: int = Field(0, ge=0, description="Unscaled upstream flank length")
    downstream: int = Field(0, ge=0, description="Unscaled downstream flank length")
    unscaled_5_prime: int = Field(0, ge=0, description="Unscaled internal 5' length")
    unscaled_3_prime: int = Field(0, ge=0, description="Unscaled internal 3' length")
    bin_size: int = Field(0, ge=0, description="Fixed bin size (0 = none)")
    n_bins: int | None = Field(
        None, ge=1, description="Explicit number of bins (reference-point only)"
    )

    @model_validator(mode="after")
    def _validate(self):  # type: ignore[override]
        if self.n_bins is not None and self.bin_size and self.bin_size > 0:
            raise ValueError("Specify only one of bin_size or n_bins, not both")
        if self.mode == "scale-regions" and self.bin_size > 0:
            for name in [
                "body",
                "upstream",
                "downstream",
                "unscaled_5_prime",
                "unscaled_3_prime",
            ]:
                v = getattr(self, name)
                if v % self.bin_size != 0:
                    raise ValueError(
                        f"{name}={v} must be a multiple of bin_size={self.bin_size} in scale-regions mode"
                    )
        return self

    @computed_field
    def window_size(self) -> int:  # type: ignore[override]
        if self.mode == "reference-point":
            return self.bp_before + self.bp_after
        return (
            self.upstream
            + self.unscaled_5_prime
            + self.body
            + self.unscaled_3_prime
            + self.downstream
        )