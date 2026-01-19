import secrets
from hashlib import sha256
from typing import Type

from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from synthefy_pkg.app.models.user_api_keys import UserAPIKeys


def generate_api_key() -> str:
    """Generates a 64-character API key."""
    logger.info("Generating a new API key.")
    return secrets.token_hex(32)


def hash_api_key(api_key: str) -> str:
    """Hashes the API key using SHA-256."""
    logger.debug("Hashing the provided API key.")
    return sha256(api_key.encode()).hexdigest()


def save_api_key(db: Session, user_id: str, name: str, api_key: str) -> tuple[str, str, int]:
    """
    Saves the API key to the database.

    Returns:
        Tuple containing the original API key, name, and the generated key ID.
    """
    logger.info("Saving a new API key for user_id: {}", user_id)
    try:
        hashed_key = hash_api_key(api_key)
        new_key = UserAPIKeys(user_id=user_id, name=name, hashed_key=hashed_key)
        db.add(new_key)
        db.commit()
        db.refresh(new_key)  # To get the ID from the database.
        logger.info("API key saved successfully with ID: {}", new_key.id)
        return api_key, name, new_key.id
    except SQLAlchemyError as e:
        logger.error("Failed to save API key for user_id: {}. Error: {}", user_id, str(e))
        db.rollback()
        raise RuntimeError("Error saving API key to the database.") from e


def validate_api_key(db: Session, api_key: str) -> str | None:
    """
    Validates if the provided API key exists in the database.

    Returns:
        True if the key exists, False otherwise.
    """
    logger.debug("Validating the API key.")
    try:
        hashed_key = hash_api_key(api_key)
        user = db.query(UserAPIKeys).filter_by(hashed_key=hashed_key).first()
        if user:
            logger.info("API key is valid for user_id: {}", user.user_id)
            return user.user_id
        logger.info("API key validation result: Key not found.")
        return None
    except SQLAlchemyError as e:
        logger.error("Failed to validate API key. Error: {}", str(e))
        raise RuntimeError("Error validating API key.") from e


def delete_api_key(db: Session, user_id: str, api_key_id: int) -> bool:
    """
    Deletes the API key entry for the specified user ID and API key ID.

    Returns:
        True if the key was successfully deleted, False otherwise.
    """
    logger.info("Deleting API key with ID: {} for user_id: {}", api_key_id, user_id)
    try:
        key_entry = db.query(UserAPIKeys).filter_by(user_id=user_id, id=api_key_id).first()
        if key_entry:
            db.delete(key_entry)
            db.commit()
            logger.info("API key with ID: {} successfully deleted.", api_key_id)
            return True
        logger.warning("API key with ID: {} not found for user_id: {}", api_key_id, user_id)
        return False
    except SQLAlchemyError as e:
        logger.error("Failed to delete API key with ID: {}. Error: {}", api_key_id, str(e))
        db.rollback()
        raise RuntimeError("Error deleting API key.") from e


def get_api_keys(db: Session, user_id: str) -> list[Type[UserAPIKeys]]:
    """
    Retrieves all API keys for the specified user ID.

    Returns:
        A list of UserAPIKeys objects.
    """
    logger.info("Fetching all API keys for user_id: {}", user_id)
    try:
        keys = db.query(UserAPIKeys).filter_by(user_id=user_id).all()
        logger.info("Retrieved {} API keys for user_id: {}", len(keys), user_id)
        return keys
    except SQLAlchemyError as e:
        logger.error("Failed to fetch API keys for user_id: {}. Error: {}", user_id, str(e))
        raise RuntimeError("Error fetching API keys.") from e
