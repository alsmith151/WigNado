from pathlib import Path
from pydantic import BaseModel, Field, field_validator

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

    def extend(self, before: int, after: int) -> "GenomicRegion":
        """Extend the genomic region by a specified number of base pairs."""
        return GenomicRegion(
            chrom=self.chrom,
            start=self.start - before,
            end=self.end + after
        )

    def __str__(self) -> str:
        return f"{self.chrom}:{self.start:,}-{self.end:,}"
