#!/usr/bin/env python3
"""
Test script to demonstrate the conditional Supabase dependency functionality.
This script shows how the user_id extraction works with and without the SYNTHEFY_USE_ACCESS_TOKEN environment variable.
"""

import os
from unittest.mock import MagicMock, patch

from synthefy_pkg.app.utils.supabase_utils import (
    get_supabase_user,
    get_user_id_from_token_or_body,
    get_user_id_from_token_or_form,
)


def test_with_access_token_enabled():
    """Test when SYNTHEFY_USE_ACCESS_TOKEN=1 is set"""
    print("=== Testing with SYNTHEFY_USE_ACCESS_TOKEN=1 ===")
    
    # Mock environment variable
    with patch.dict(os.environ, {'SYNTHEFY_USE_ACCESS_TOKEN': '1'}):
        # Mock the get_supabase_user function
        with patch('synthefy_pkg.app.utils.supabase_utils.get_supabase_user') as mock_get_user:
            mock_get_user.return_value = "user_from_token_123"
            
            # Test get_user_id_from_token_or_form
            try:
                result = get_user_id_from_token_or_form(
                    authorization="Bearer valid_token_here",
                    user_id="user_from_form_456"  # This should be ignored
                )
                print(f"✓ get_user_id_from_token_or_form returned: {result}")
                assert result == "user_from_token_123"
            except Exception as e:
                print(f"✗ get_user_id_from_token_or_form failed: {e}")
            
            # Test get_user_id_from_token_or_body
            try:
                result = get_user_id_from_token_or_body(
                    authorization="Bearer valid_token_here",
                    request_body={"user_id": "user_from_body_789"}  # This should be ignored
                )
                print(f"✓ get_user_id_from_token_or_body returned: {result}")
                assert result == "user_from_token_123"
            except Exception as e:
                print(f"✗ get_user_id_from_token_or_body failed: {e}")
            
            # Test missing authorization header
            try:
                get_user_id_from_token_or_form(
                    authorization=None,
                    user_id="user_from_form_456"
                )
                print("✗ Should have failed with missing authorization header")
            except Exception as e:
                print(f"✓ Correctly failed with missing authorization: {e}")


def test_with_access_token_disabled():
    """Test when SYNTHEFY_USE_ACCESS_TOKEN is not set or set to 0"""
    print("\n=== Testing with SYNTHEFY_USE_ACCESS_TOKEN=0 ===")
    
    # Mock environment variable
    with patch.dict(os.environ, {'SYNTHEFY_USE_ACCESS_TOKEN': '0'}):
        # Test get_user_id_from_token_or_form
        try:
            result = get_user_id_from_token_or_form(
                authorization="Bearer valid_token_here",  # This should be ignored
                user_id="user_from_form_456"
            )
            print(f"✓ get_user_id_from_token_or_form returned: {result}")
            assert result == "user_from_form_456"
        except Exception as e:
            print(f"✗ get_user_id_from_token_or_form failed: {e}")
        
        # Test get_user_id_from_token_or_body
        try:
            result = get_user_id_from_token_or_body(
                authorization="Bearer valid_token_here",  # This should be ignored
                request_body={"user_id": "user_from_body_789"}
            )
            print(f"✓ get_user_id_from_token_or_body returned: {result}")
            assert result == "user_from_body_789"
        except Exception as e:
            print(f"✗ get_user_id_from_token_or_body failed: {e}")
        
        # Test missing user_id in form
        try:
            get_user_id_from_token_or_form(
                authorization="Bearer valid_token_here",
                user_id=None
            )
            print("✗ Should have failed with missing user_id")
        except Exception as e:
            print(f"✓ Correctly failed with missing user_id: {e}")
        
        # Test missing user_id in body
        try:
            get_user_id_from_token_or_body(
                authorization="Bearer valid_token_here",
                request_body={"other_field": "value"}
            )
            print("✗ Should have failed with missing user_id in body")
        except Exception as e:
            print(f"✓ Correctly failed with missing user_id in body: {e}")


def test_environment_variable_not_set():
    """Test when SYNTHEFY_USE_ACCESS_TOKEN environment variable is not set"""
    print("\n=== Testing with SYNTHEFY_USE_ACCESS_TOKEN not set ===")
    
    # Remove the environment variable
    if 'SYNTHEFY_USE_ACCESS_TOKEN' in os.environ:
        del os.environ['SYNTHEFY_USE_ACCESS_TOKEN']
    
    # Test get_user_id_from_token_or_form
    try:
        result = get_user_id_from_token_or_form(
            authorization="Bearer valid_token_here",  # This should be ignored
            user_id="user_from_form_456"
        )
        print(f"✓ get_user_id_from_token_or_form returned: {result}")
        assert result == "user_from_form_456"
    except Exception as e:
        print(f"✗ get_user_id_from_token_or_form failed: {e}")
    
    # Test get_user_id_from_token_or_body
    try:
        result = get_user_id_from_token_or_body(
            authorization="Bearer valid_token_here",  # This should be ignored
            request_body={"user_id": "user_from_body_789"}
        )
        print(f"✓ get_user_id_from_token_or_body returned: {result}")
        assert result == "user_from_body_789"
    except Exception as e:
        print(f"✗ get_user_id_from_token_or_body failed: {e}")


if __name__ == "__main__":
    print("Testing Supabase Conditional Dependency")
    print("=" * 50)
    
    test_with_access_token_enabled()
    test_with_access_token_disabled()
    test_environment_variable_not_set()
    
    print("\n" + "=" * 50)
    print("Test completed!") 