from datetime import datetime

from sqlalchemy import Column, Integer, String, DateTime

from synthefy_pkg.app.db import Base


class UserAPIKeys(Base):
    __tablename__ = "user_api_keys"
    # TODO: CHECK IF THIS IS NEEDED - DONE TO AVOID COMPILATION ERRORS
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), index=True)  # Link to Supabase user_id
    name = Column(String(100), nullable=False)  # Name of the API Key
    hashed_key = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = {"extend_existing": True}
