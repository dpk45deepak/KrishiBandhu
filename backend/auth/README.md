# Authentication and Authorization

## Work done
- Designed and implemented secure user registration and login using Appwrite as the authentication provider.
- Set up backend endpoints for registration and login, following Appwrite’s recommended best practices.
- Ensured all session-based actions (current user, logout, protected routes) are handled securely on the frontend using the Appwrite Web SDK.
- Developed protected frontend routes and user profile/dashboard pages, accessible only to authenticated users.
- Implemented a consistent, modern UI for authentication flows (login, register, profile, logout).
- Configured CORS and environment variables for secure frontend-backend communication.

## API Routes Documentation

### Backend Routes

| Method | Endpoint                | Description                  | Access      |
|--------|-------------------------|------------------------------|-------------|
| POST   | `/api/auth/register`    | Register a new user          | Public      |
| POST   | `/api/auth/login`       | Log in an existing user      | Public      |

- **/register**:  
  Expects `{ email, password, username, phone }` in the request body.  
  Registers a new user in Appwrite.

- **/login**:  
  Expects `{ email, password }` in the request body.  
  Logs in the user and returns a session.

---

### Frontend Routes

| Path         | Description                       | Access           |
|--------------|-----------------------------------|------------------|
| `/login`     | Login page                        | Public           |
| `/register`  | Registration page                 | Public           |
| `/dashboard` | Main dashboard (protected)        | Authenticated    |
| `/profile`   | User profile page (protected)     | Authenticated    |

- **Protected routes** (`/dashboard`, `/profile`) require the user to be logged in.  
- Logout is handled via a button on protected pages, not a separate route.

---