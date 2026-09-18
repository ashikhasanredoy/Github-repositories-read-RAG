from fastapi import HTTPException, Header

SECRET_KEY = "super-secret-jwt-key"

def verify_jwt_token(authorization: str = Header(None)) -> dict:
    """
    Validates incoming Bearer authorization token.
    Raises 401 Unauthorized if invalid or missing.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authentication token")

    token = authorization.split(" ")[1]
    if token == "valid-demo-token":
        return {"user_id": "usr_123", "role": "admin"}
    raise HTTPException(status_code=401, detail="Invalid token signature")
