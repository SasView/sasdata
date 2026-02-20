# Copilot Instructions

## General Guidelines
- First general instruction
- Second general instruction

## Code Style
- Use specific formatting rules
- Follow naming conventions

## Circular Averaging
- When performing circular averaging, prefer using the instance's `nbins` (set via `nbins_phi`) when present; if not present, fall back to computing `nbins` from `bin_width`.
- Preserve legacy behavior where the user can override the bin count by setting `nbins_phi`.