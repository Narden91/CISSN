"""
Shared constants for the CISSN package.

Centralised here so that the structured state dimensionality is defined
exactly once and referenced everywhere else, rather than copied as three
independent class-level magic numbers.
"""

# The CISSN encoder maps every input sequence to a 5-dimensional structured
# latent state: [Level, Trend, Seasonal-cos, Seasonal-sin, Residual].
# This value is a hard architectural constraint; changing it requires
# re-designing the block-diagonal A matrix and the disentanglement loss.
STRUCTURED_STATE_DIM: int = 5
