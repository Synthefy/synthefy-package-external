"""
License Key Generation and Verification Module
================================================

This module provides functionality for generating and verifying JWT-based license keys,
as well as performing container runtime checks. License keys are signed using RSA (RS256).
The private key is used for signing and must be kept secret. The public key is
embedded in the customer-distributed code for license verification.

Key Generation:
---------------
Generate an RSA key pair using OpenSSL:

1. Generate a 2048-bit RSA private key:
   $ openssl genpkey -algorithm RSA -out private_key.pem -pkeyopt rsa_keygen_bits:2048

2. Extract the corresponding public key:
   $ openssl rsa -pubout -in private_key.pem -out public_key.pem

Setting Keys:
-------------
- PRIVATE_KEY: Set this environment variable in the environment of the person who from Synthefy who generates
  the license key. It can be set in the following way:

      export PRIVATE_KEY="$(cat /path/to/private_key.pem)"

- PUBLIC_KEY: The public key (contents of public_key.pem) is embedded in this module.

Usage:
------
- Use `generate_license_key(expiration_date: str)` to generate license keys.
- Use `verify_license_key(license_key: str)` in customer containers to verify license validity.
- The module also provides functions to encrypt container startup times and to verify
  system clock tampering and uptime limits.

IMPORTANT:
----------
Any error message raised from this module instructs the user to contact Synthefy for support,
without revealing internal details of the license verification mechanism.
"""

import datetime
import inspect
import json
import os
import sys
import time
import uuid

import jwt
from cryptography.fernet import Fernet

# Module-level constants for public key and Fernet encryption key.
PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEApxMIeShNPkDJSD5jNGX0
kHQk7AUzrDs+9frk9jSIy0rs/F6hMtJlM4VLdkpmLVbuG2FmRqLDy7cmPE4n9C/t
zfeOYsfOyAe0ag2iPnew4HrNPSKT3KuWmiRGA2Zt+fs2cQFIOtRGOanCmOFKUKax
nRnigPnBoAALUNTqrBh0FdJvNVki6OdG6JJdUzW5SsyRdDj6t8dJWqvIBH88Oy4S
W4uuElqP32KWWo+ryOerubwmsB1bc4JnE+KjKLCUpoFmXHRsZLtqndKSKhb3ntLy
J5X6+wvjQlwgibUCzCyVnNWRVPwFCufSYBwqjiXjkGrgh5i2QghoYbbukLqD1J3j
sQIDAQAB
-----END PUBLIC KEY-----"""

COMPILE = True

# The Fernet key is just used for encryption of the startup time.
FERNET_KEY = b"9fn7h9G4iCHb9YCCjvvnIVuc3GHnXCiHF-A9eVal2W8="


def error(message: str) -> None:
    """
    Print an error message in red.

    Args:
        message: The error message to print
    """
    print(f"\033[91m{message}\033[0m")


def generate_license_key(expiration_date: str) -> str:
    """
    Generates a secure JWT license key with issuance and expiration times.

    The license key is signed with the RSA private key using the RS256 algorithm.
    The payload includes a unique identifier, issued-at, and expiration timestamps (in Unix time).

    :param expiration_date: Expiration date in the format 'YYYY-MM-DD'
    :return: Encoded JWT license key as a string.
    :raises EnvironmentError: If the PRIVATE_KEY environment variable is missing.
    :raises ValueError: If the expiration_date format is invalid.
    :raises RuntimeError: If license key generation fails.
    """
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    if not PRIVATE_KEY:
        error(
            "PRIVATE_KEY environment variable is not set. Aborting license generation."
        )
        raise EnvironmentError(
            "Private key not configured. Set the PRIVATE_KEY environment variable with your private key."
        )

    unique_id = str(uuid.uuid4())
    try:
        # Parse the expiration date and set time to end-of-day UTC.
        exp_datetime = datetime.datetime.strptime(expiration_date, "%Y-%m-%d")
        exp_datetime = exp_datetime.replace(
            hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc
        )
    except ValueError:
        error("Invalid expiration date format. Expected 'YYYY-MM-DD'.")
        raise ValueError(
            "Invalid expiration date format. Please contact Synthefy for support."
        )

    iat_datetime = datetime.datetime.now(datetime.timezone.utc)

    payload = {
        "uuid": unique_id,
        "iat": int(iat_datetime.timestamp()),
        "exp": int(exp_datetime.timestamp()),
    }

    try:
        license_key = jwt.encode(payload, PRIVATE_KEY, algorithm="RS256")
        return license_key
    except Exception as e:
        error("Failed to generate license key.")
        raise RuntimeError(f"License key generation failed. {e}")


def verify_license_key(license_key: str) -> bool:
    """
    Verifies the validity of a JWT license key.

    This function checks that the license key:
      - Has a valid signature using the embedded RSA public key.
      - Contains valid 'iat' (issued at) and 'exp' (expiration) claims.
      - Was issued in the past.

    :param license_key: The JWT license key as a string.
    :return: True if the license key is valid.
    :raises RuntimeError: If the license key is expired or if verification fails.
    :raises ValueError: If the license key is malformed.
    """
    try:
        decoded_payload = jwt.decode(
            license_key, PUBLIC_KEY, algorithms=["RS256"]
        )
        iat = decoded_payload.get("iat")
        if iat is None:
            error("License key is missing the 'iat' claim.")
            raise ValueError(
                "Invalid license key. Please contact Synthefy for support."
            )

        # Convert 'iat' to a datetime object.
        if isinstance(iat, (int, float)):
            iat_datetime = datetime.datetime.fromtimestamp(
                iat, tz=datetime.timezone.utc
            )
        elif isinstance(iat, str):
            try:
                iat_datetime = datetime.datetime.fromisoformat(iat)
                if iat_datetime.tzinfo is None:
                    iat_datetime = iat_datetime.replace(
                        tzinfo=datetime.timezone.utc
                    )
            except Exception:
                error("Failed to parse 'iat' claim.")
                raise ValueError(
                    "Invalid license key. Please contact Synthefy for support."
                )
        else:
            error("Invalid type for 'iat' claim.")
            raise ValueError(
                "Invalid license key. Please contact Synthefy for support."
            )

        current_time = datetime.datetime.now(datetime.timezone.utc)
        if iat_datetime > current_time:
            error("Invalid license key. 'iat' is not correct.")
            raise ValueError(
                "Invalid license key. Please contact Synthefy for support."
            )

        return True

    except jwt.ExpiredSignatureError:
        error("License key has expired. Please contact Synthefy for support.")
        sys.exit(1)
    except jwt.InvalidTokenError:
        error(
            "Invalid license key provided. Please contact Synthefy for support."
        )
        sys.exit(1)
    except ValueError as ve:
        error(f"Invalid license key: {ve}")
        sys.exit(1)
    except Exception:
        error(
            "Unexpected error occured during license verification. Please contact Synthefy for support."
        )
        sys.exit(1)


def generate_key() -> bytes:
    """
    Generates a new Fernet key for encryption purposes.

    :return: A URL-safe base64-encoded 32-byte key.
    :raises RuntimeError: If key generation fails.
    """
    try:
        return Fernet.generate_key()
    except Exception:
        error("Failed to generate encryption key.")
        raise RuntimeError(
            "Encryption key generation failed. Please contact Synthefy for support."
        )


def save_encrypted_started_time() -> None:
    """
    Encrypts the container's startup time information and saves it to a secure file.

    The saved data includes:
      - The current UTC datetime.
      - The system monotonic time.

    If the file already exists, it is not overwritten.

    :raises RuntimeError: If saving the encrypted startup time fails.
    """
    filename = "/tmp/bb01c563-dab4-43c3-9f4a-f1e9be9c1356"
    if os.path.exists(filename):
        return

    try:
        fernet = Fernet(FERNET_KEY)
        current_datetime = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        monotonic_time = time.monotonic()
        data = {"datetime": current_datetime, "monotonic": monotonic_time}
        data_bytes = json.dumps(data).encode("utf-8")
        encrypted_data = fernet.encrypt(data_bytes)
        with open(filename, "wb") as file:
            file.write(encrypted_data)
    except Exception:
        error("Failed to determine the container's startup time.")
        sys.exit(1)


def get_decrypted_started_time() -> dict:
    """
    Decrypts and retrieves the stored startup time information from the secure file.

    :return: A dictionary with keys:
             - 'datetime': The startup time as a datetime object (UTC).
             - 'monotonic': The system monotonic time as a float.
    :raises RuntimeError: If the startup time file is missing or decryption fails.
    """
    filename = "/tmp/bb01c563-dab4-43c3-9f4a-f1e9be9c1356"

    # When licensing_utils is imported during bootstrap (via python -c),
    # this function gets called before the uptime is saved, since the
    # __init__ file's checks are triggered. In this case, return the
    # current datetime to avoid failures during bootstrap.
    if (
        not os.path.exists(filename)
        and inspect.stack()[-1].filename == "<string>"
    ):
        return {
            "datetime": datetime.datetime.now(datetime.timezone.utc),
            "monotonic": time.monotonic(),
        }

    if not os.path.exists(filename):
        error(
            "Startup time record not found. Please contact Synthefy for support."
        )
        sys.exit(1)

    try:
        fernet = Fernet(FERNET_KEY)
        with open(filename, "rb") as file:
            encrypted_data = file.read()
        decrypted_bytes = fernet.decrypt(encrypted_data)
        data = json.loads(decrypted_bytes.decode("utf-8"))
        data["datetime"] = datetime.datetime.fromisoformat(
            data["datetime"]
        ).replace(tzinfo=datetime.timezone.utc)
        data["monotonic"] = float(data["monotonic"])
        return data
    except Exception:
        error(
            "Failed to retrieve the container's startup time. Please contact Synthefy for support."
        )
        sys.exit(1)

def check_dev_mode() -> bool:
    """
    Checks if the SYNTHEFY_DEV_MODE environment variable is set.
    """
    return os.getenv("SYNTHEFY_DEV_MODE", "0") == "1"

def bootstrap() -> None:
    """
    Bootstraps the licensing system by verifying the license key and initializing
    the startup time record.

    This function:
      - Checks that the LICENSE_KEY environment variable is set.
      - Verifies the provided license key.
      - Saves the encrypted container startup time if not already present.

    :raises RuntimeError: If any step of the bootstrap process fails.
    """

    # If SYNTHEFY_DEV_MODE is set, skip the license key check.
    if check_dev_mode():
        return

    LICENSE_KEY = os.getenv("LICENSE_KEY")
    if not LICENSE_KEY:
        error("LICENSE_KEY environment variable is not set.")
        print("To set the license key, run the container with:")
        print("    docker run -e LICENSE_KEY=<your_license_key> ...")
        sys.exit(1)

    try:
        verify_license_key(LICENSE_KEY)
        save_encrypted_started_time()
        print("Bootstrap completed successfully.")
    except Exception:
        error("Bootstrap Verification Failed.")
        sys.exit(1)


def verify_clock_tampering() -> None:
    """
    Verifies that the system clock has not been tampered with by comparing the elapsed
    time between the wall clock and the system's monotonic clock since startup.

    :raises RuntimeError: If clock tampering is detected or verification fails.
    """
    try:
        started_time = get_decrypted_started_time()
        current_datetime = datetime.datetime.now(datetime.timezone.utc)
        current_monotonic = time.monotonic()
        datetime_diff = abs(
            (current_datetime - started_time["datetime"]).total_seconds()
        )
        monotonic_diff = abs(current_monotonic - started_time["monotonic"])

        # Allow a 10-second tolerance between the two measurements.
        if abs(datetime_diff - monotonic_diff) > 10:
            error("System clock tampering detected.")
            sys.exit(1)
    except Exception:
        error("Clock tampering detected. Please contact Synthefy for support.")
        sys.exit(1)


def verify_uptime() -> None:
    """
    Verifies that the container's runtime does not exceed the allowed duration specified by
    the license key's validity period.

    The function:
      - Retrieves the encrypted startup time.
      - Decodes the license key to extract 'iat' and 'exp' claims.
      - Computes the maximum allowed uptime from the license validity period.
      - Compares the actual container uptime with the allowed duration.

    :raises RuntimeError: If the container's uptime exceeds the allowed duration or if any check fails.
    """
    started_time = get_decrypted_started_time()
    LICENSE_KEY = os.getenv("LICENSE_KEY")
    if not LICENSE_KEY:
        error("LICENSE_KEY environment variable is missing.")
        sys.exit(1)

    try:
        decoded_payload = jwt.decode(
            LICENSE_KEY, PUBLIC_KEY, algorithms=["RS256"]
        )
        iat_timestamp = decoded_payload.get("iat")
        exp_timestamp = decoded_payload.get("exp")
        if not iat_timestamp or not exp_timestamp:
            error("License key missing required claims.")
            raise ValueError(
                "Invalid license key. Please contact Synthefy for support."
            )

        iat_datetime = datetime.datetime.fromtimestamp(
            iat_timestamp, tz=datetime.timezone.utc
        )
        exp_datetime = datetime.datetime.fromtimestamp(
            exp_timestamp, tz=datetime.timezone.utc
        )
        max_allowed_duration = (exp_datetime - iat_datetime).total_seconds()

        current_monotonic = time.monotonic()
        uptime_seconds = current_monotonic - started_time["monotonic"]

        if uptime_seconds > max_allowed_duration:
            error(
                "License expired: container runtime exceeds permitted duration. Please contact Synthefy for support."
            )

    except jwt.ExpiredSignatureError:
        error("License key has expired. Please contact Synthefy for support.")
        sys.exit(1)
    except jwt.InvalidTokenError:
        error(
            "Invalid license key detected during uptime verification. Please contact Synthefy for support."
        )
        sys.exit(1)
    except Exception:
        error(
            "Unexpected error occured during uptime verification. Please contact Synthefy for support."
        )
        sys.exit(1)
