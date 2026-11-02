#!/usr/bin/env python3
"""Default entry point for the best validated multi-judge rubric (v6)."""
import cross_validate
import cross_validate_v6


if __name__ == "__main__":
    cross_validate_v6.configure()
    raise SystemExit(cross_validate.main())
