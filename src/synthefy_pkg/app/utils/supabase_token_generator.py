#!/usr/bin/env python3
"""
Supabase Token Generator Utility

This module provides functions to programmatically create dummy users and obtain
fresh Supabase bearer tokens for testing purposes.
"""

import os
import uuid
from typing import Any, Dict, Optional

from gotrue._sync.gotrue_client import AuthApiError
from loguru import logger
from supabase import Client, ClientOptions, create_client


class SupabaseTokenGenerator:
    """
    Utility class for generating fresh Supabase bearer tokens for dummy users.
    """

    def __init__(self):
        """
        Initialize the token generator with Supabase credentials.

        Uses anon key for regular auth operations and service role key for admin operations.
        """
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_anon_key = os.getenv("SUPABASE_API_KEY")
        self.supabase_service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not self.supabase_url:
            raise ValueError("SUPABASE_URL must be provided")
        if not self.supabase_anon_key:
            raise ValueError("SUPABASE_API_KEY must be provided")
        if not self.supabase_service_key:
            raise ValueError("SUPABASE_SERVICE_ROLE_KEY must be provided")

        # Use anon key for regular auth operations
        self.client: Client = create_client(
            self.supabase_url,
            self.supabase_anon_key,
            options=ClientOptions(auto_refresh_token=False),
        )

        # Use service role key for admin operations
        self.admin_client: Client = create_client(
            self.supabase_url,
            self.supabase_service_key,
            options=ClientOptions(auto_refresh_token=False),
        )

        logger.info(
            f"Initialized Supabase clients for URL: {self.supabase_url}"
        )
        logger.info(
            "Using anon key for regular operations and service role key for admin operations"
        )

    def create_dummy_user(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a dummy user and return the user data with access token.

        Args:
            email: Email for the dummy user (defaults to random email)
            password: Password for the dummy user (defaults to random password)
            user_metadata: Optional metadata to attach to the user

        Returns:
            Dictionary containing user data and access token
        """
        # Generate random credentials if not provided
        if not email:
            email = f"dummy_user_{uuid.uuid4().hex[:8]}@synthefy.com"

        if not password:
            password = f"dummy_password_{uuid.uuid4().hex[:12]}"

        # Prepare user metadata
        metadata = user_metadata or {}
        metadata.update(
            {
                "dummy_user": True,
                "created_by": "supabase_token_generator",
                "purpose": "testing",
            }
        )

        try:
            logger.info(f"Creating dummy user with email: {email}")

            # Sign up the user
            response = self.client.auth.sign_up(
                {
                    "email": email,
                    "password": password,
                    "options": {"data": metadata},
                }
            )

            if not response.user:
                raise Exception("Failed to create user - no user returned")

            # For sign-up, we need to sign in to get a session
            logger.info(
                "User created successfully, now signing in to get session"
            )
            sign_in_response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            if not sign_in_response.user or not sign_in_response.session:
                raise Exception(
                    "Failed to sign in after user creation - no user or session returned"
                )

            session = sign_in_response.session

            user_data = {
                "user_id": response.user.id,
                "email": response.user.email,
                "access_token": session.access_token,
                "refresh_token": session.refresh_token,
                "expires_at": session.expires_at,
                "created_at": response.user.created_at,
                "metadata": metadata,
            }

            logger.info(
                f"Successfully created dummy user: {user_data['user_id']}"
            )
            return user_data

        except AuthApiError as e:
            logger.error(f"Supabase Auth API error: {e.message}")
            raise Exception(f"Failed to create dummy user: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error creating dummy user: {str(e)}")
            raise

    def sign_in_existing_user(
        self, email: str, password: str
    ) -> Dict[str, Any]:
        """
        Sign in an existing user and return the session data.

        Args:
            email: User's email
            password: User's password

        Returns:
            Dictionary containing user data and access token
        """
        try:
            logger.info(f"Signing in existing user: {email}")

            response = self.client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )

            if not response.user or not response.session:
                raise Exception(
                    "Failed to sign in - no user or session returned"
                )

            user_data = {
                "user_id": response.user.id,
                "email": response.user.email,
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "expires_at": response.session.expires_at,
                "created_at": response.user.created_at,
                "metadata": response.user.user_metadata or {},
            }

            logger.info(f"Successfully signed in user: {user_data['user_id']}")
            return user_data

        except AuthApiError as e:
            logger.error(f"Supabase Auth API error: {e.message}")
            raise Exception(f"Failed to sign in user: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error signing in user: {str(e)}")
            raise

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: The refresh token to use

        Returns:
            Dictionary containing new session data
        """
        try:
            logger.info("Refreshing access token")

            response = self.client.auth.refresh_session(refresh_token)

            if not response.session:
                raise Exception("Failed to refresh token - no session returned")

            session_data = {
                "access_token": response.session.access_token,
                "refresh_token": response.session.refresh_token,
                "expires_at": response.session.expires_at,
            }

            logger.info("Successfully refreshed access token")
            return session_data

        except AuthApiError as e:
            logger.error(f"Supabase Auth API error: {e.message}")
            raise Exception(f"Failed to refresh token: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error refreshing token: {str(e)}")
            raise

    def delete_user(self, access_token: str) -> bool:
        """
        Delete a user using their access token.

        Note: This requires a service role key to work properly.

        Args:
            access_token: The user's access token

        Returns:
            True if user was successfully deleted
        """
        try:
            logger.info("Deleting user")

            # When using service role key, we can decode the token to get user ID
            # without needing to validate it with regular auth methods
            import jwt

            try:
                # Decode the JWT token to get user ID
                decoded = jwt.decode(
                    access_token, options={"verify_signature": False}
                )
                user_id = decoded.get("sub")
                if not user_id:
                    raise Exception("Invalid token - no user ID found")

                logger.info(f"Attempting to delete user with ID: {user_id}")

                # Delete the user by ID using admin client
                _ = self.admin_client.auth.admin.delete_user(
                    user_id, should_soft_delete=False
                )

                logger.info("Successfully deleted user")
                return True

            except jwt.InvalidTokenError:
                raise Exception("Invalid token format")

        except AuthApiError as e:
            logger.error(f"Supabase Auth API error: {e.message}")
            logger.error(f"Error details: {e}")
            raise Exception(f"Failed to delete user: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error deleting user: {str(e)}")
            raise

    def get_bearer_token(self, access_token: str) -> str:
        """
        Format an access token as a bearer token for API requests.

        Args:
            access_token: The access token

        Returns:
            Formatted bearer token string
        """
        return f"Bearer {access_token}"

    def validate_token(self, access_token: str) -> Dict[str, Any]:
        """
        Validate an access token and return user information.

        Args:
            access_token: The access token to validate

        Returns:
            Dictionary containing user information

        Raises:
            Exception: If token is invalid or validation fails
        """
        try:
            logger.info("Validating access token")

            # Always use regular validation for user tokens
            # Admin methods are only for admin operations, not token validation
            response = self.client.auth.get_user(access_token)

            if not response or not response.user:
                raise Exception("Invalid token - no user returned")

            user_info = {
                "user_id": response.user.id,
                "email": response.user.email,
                "created_at": response.user.created_at,
                "metadata": response.user.user_metadata or {},
            }

            logger.info(f"Token is valid for user: {user_info['user_id']}")
            return user_info

        except AuthApiError as e:
            logger.error(f"Supabase Auth API error: {e.message}")
            raise Exception(f"Invalid token: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error validating token: {str(e)}")
            raise

    def sign_out(self) -> None:
        """
        Sign out the current user session.

        This method clears the current session and should be called
        when you're done using the token generator to clean up.
        """
        try:
            logger.info("Signing out user session")
            self.client.auth.sign_out()
            self.admin_client.auth.sign_out()
            logger.info("Successfully signed out")
        except Exception as e:
            logger.error(f"Error signing out: {str(e)}")
            # Don't raise the exception as sign out is cleanup


if __name__ == "__main__":
    """
    Example usage of the Supabase token generator.
    """
    import json

    # Create a single generator instance with service role key
    try:
        print("=== Creating dummy user ===")
        # Use service role key for all operations
        generator = SupabaseTokenGenerator()

        # Example 1: Create a dummy user and get token
        dummy_user = generator.create_dummy_user()
        print(f"Created dummy user: {dummy_user['email']}")

        print(f"User ID: {dummy_user['user_id']}")
        print(f"Bearer token: {dummy_user['access_token'][:50]}...")
        print(f"Token expires at: {dummy_user['expires_at']}")

        # Example 2: Validate the token
        print("\n=== Validating token ===")
        user_info = generator.validate_token(dummy_user["access_token"])
        print(f"Token is valid for user: {user_info['email']}")

        # Example 3: Refresh the token
        print("\n=== Refreshing token ===")
        new_session = generator.refresh_token(dummy_user["refresh_token"])
        print(f"New access token: {new_session['access_token'][:50]}...")
        print(f"New token expires at: {new_session['expires_at']}")

        # Example 4: Clean up (delete the user)
        print("\n=== Cleaning up ===")
        success = generator.delete_user(dummy_user["access_token"])
        if success:
            print("Dummy user deleted successfully")
        else:
            print(
                "Failed to delete user (service role key may not be available)"
            )

        # Sign out to clean up session
        print("\n=== Signing out ===")
        generator.sign_out()
        print("Successfully signed out")

    except Exception as e:
        print(f"Error: {str(e)}")
        print(
            "\nMake sure you have set the SUPABASE_URL and SUPABASE_API_KEY environment variables."
        )
        print(
            "For testing, you may also want to set SUPABASE_SERVICE_ROLE_KEY to bypass email confirmation."
        )
