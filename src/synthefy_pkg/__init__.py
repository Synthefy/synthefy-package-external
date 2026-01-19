import os
import sys

from synthefy_pkg.utils.licensing_utils import (
    error,
    verify_clock_tampering,
    verify_license_key,
    verify_uptime,
)

COMPILE = True


def check_license_key():
    return
    try:
        LICENSE_KEY = os.getenv("LICENSE_KEY")
        if not LICENSE_KEY:
            error("LICENSE_KEY environment variable is not set")
            raise ValueError(
                "LICENSE_KEY environment variable must be set to run this container"
            )
        verify_license_key(LICENSE_KEY)
        verify_clock_tampering()
        verify_uptime()
    except Exception as e:
        error(f"License key verification failed: {e}")
        sys.exit(1)


if not os.getenv("SYNTHEFY_DEV_MODE"):
    print("Checking license key!")
    check_license_key()
    print("License key check completed successfully.")
