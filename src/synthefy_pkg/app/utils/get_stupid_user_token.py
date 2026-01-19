#!/usr/bin/env python3
"""
Script to get a fresh access token for stupid@synthefy.com user.
Creates the user if they don't exist, otherwise signs them in.
Outputs the token to GitHub Actions output file if running in CI.
"""

import os
import sys

from loguru import logger

from synthefy_pkg.app.utils.supabase_token_generator import (
    SupabaseTokenGenerator,
)


def get_stupid_user_token():
    """
    Get a fresh access token for stupid@synthefy.com user.
    Creates the user if they don't exist.

    Returns:
        str: Fresh access token
    """
    email = "stupid@synthefy.com"
    password = "stupid_password_123"  # Fixed password for consistency

    try:
        generator = SupabaseTokenGenerator()

        # Try to sign in existing user first
        try:
            logger.info(f"Attempting to sign in existing user: {email}")
            user_data = generator.sign_in_existing_user(email, password)
            logger.info(
                f"Successfully signed in existing user: {user_data['user_id']}"
            )
            return user_data["access_token"]

        except Exception as sign_in_error:
            # If sign in fails, create the user
            logger.info(f"Sign in failed: {sign_in_error}")
            logger.info(f"Creating new user: {email}")

            user_data = generator.create_dummy_user(
                email=email,
                password=password,
                user_metadata={
                    "dummy_user": True,
                    "created_by": "get_stupid_user_token_script",
                    "purpose": "testing",
                    "user_type": "stupid_user",
                },
            )

            logger.info(
                f"Successfully created new user: {user_data['user_id']}"
            )
            return user_data["access_token"]

    except Exception as e:
        logger.error(f"Error getting token for {email}: {str(e)}")
        raise


def main():
    """Main function to get and print the access token."""
    try:
        # Check environment variables
        required_vars = [
            "SUPABASE_URL",
            "SUPABASE_API_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            print(
                f"Error: Missing required environment variables: {', '.join(missing_vars)}"
            )
            print("Please set the following environment variables:")
            for var in missing_vars:
                print(f"  - {var}")
            sys.exit(1)

        # Get the access token
        access_token = get_stupid_user_token()

        # Print the token
        print("Access token for stupid@synthefy.com:")
        print(access_token)

        # Also print as bearer token format
        print("\nBearer token format:")
        print(f"Bearer {access_token}")

        # Get the path to the GITHUB_OUTPUT file
        github_output_file = os.getenv("GITHUB_OUTPUT")

        if github_output_file:
            # Append the output to the GITHUB_OUTPUT file
            # The format is "name=value"
            with open(github_output_file, "a") as f:
                f.write(f"access_token={access_token}\n")
            print("Output 'access_token' set to GitHub Actions output")
        else:
            print(
                "GITHUB_OUTPUT environment variable not found. Cannot set output."
            )

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
