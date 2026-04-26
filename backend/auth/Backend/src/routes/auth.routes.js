import express from 'express';
import * as sdk from "node-appwrite";
import authService from '../controllers/auth.controllers.js';
import { ApiError } from '../utils/ApiError.js';
import { ApiResponse } from '../utils/ApiResponse.js';
//import { requireAuth } from '../middlewares/auth.middlewares.js';


const router = express.Router();

router.route('/register').post(async (req, res, next) => {
  try {
    console.log(req.body)
    const { email, password, phone, username } = req.body;
    const result = await authService.createAccount({ email, password, phone, username });
    return res
      .status(201)
      .json(new ApiResponse(201, "User registered successfully", { sessionId: result.$id, ...result }));
  } catch (error) {
    console.log(error)
    throw new ApiError(500, "Registration failed");
  }
});

router.route('/login').post(async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const result = await authService.login({ email, password });
    return res
      .status(200)
      .json(new ApiResponse(200, "User logged in successfully", { sessionId: result.$id, ...result }));
    //.json(new ApiResponse(200, "User logged in successfully", result));
  } catch (error) {
    console.log(error);
    throw new ApiError(500, "Login failed");
  }
});

// router.route('/current-user').get(async (req, res, next) => {
//   try {
//     const sessionCookie = req.headers.cookie;
//     console.log("Headers received:", req.headers);
//     if (!sessionCookie) throw new ApiError("No session cookie provided");

//     const userClient = new sdk.Client()
//       .setEndpoint(process.env.APPWRITE_HOST_URL)
//       .setProject(process.env.APPWRITE_PROJECT_ID);

//     userClient.headers['X-Fallback-Cookies'] = sessionCookie;
//     const accountWithSession = new sdk.Account(userClient);

//     const user = await accountWithSession.get();
//     res.status(200).json({ user });
//   } catch (error) {
//     console.log(error);
//     throw new ApiError(500, "Failed to fetch current user");
//   }
// });

// router.post('/logout', async (req, res, next) => {
//   try {
//     const sessionId = req.body?.sessionId || req.headers['x-appwrite-session'];
//     if (!sessionId) {
//       throw new ApiError(400, "Session ID is required");
//     }
//     await authService.logout(sessionId);
//     res.status(200).json(new ApiResponse(200, "User logged out successfully"));
//   } catch (error) {
//     console.log(error);
//     throw new ApiError(500, "Logout failed");
//   }
// });

// router.route('/protected').get(requireAuth, (req,res) => {
//   res.json({ message: "You are authenticated!", user: req.user });
// })


export default router;