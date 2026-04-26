import { account, client } from "../services/appwrite.js";
import * as sdk from "node-appwrite";
import { ApiError } from '../utils/ApiError.js';

class AuthService {
    async createAccount({ email, password, phone, username }) {
        try {
            const users = new sdk.Users(client);
            const user = await users.create(sdk.ID.unique(), email, phone, password, username);
            if (user) {
                return this.login({ email, password });
            } else {
                return user;
            }
        } catch (error) {
            throw new ApiError(500, "Failed to create account", [error.message]);
        }
    }

    async login({ email, password }) {
        try {
            return await account.createEmailPasswordSession(email, password);
        } catch (error) {
            throw new ApiError(500, "Failed to login", [error.message]);
        }
    }

    // async getCurrentUser(sessionId) {
    //     try {
    //         const userClient = new sdk.Client()
    //             .setEndpoint(process.env.APPWRITE_HOST_URL)
    //             .setProject(process.env.APPWRITE_PROJECT_ID);
    //         userClient.headers['X-Appwrite-Session'] = sessionId;
    //         const accountWithSession = new sdk.Account(userClient);
    //         const user = await accountWithSession.get();
    //         return user || null;
    //     } catch (error) {
    //         throw new ApiError(500, "Failed to fetch user", [error.message]);
    //     }
    // }

    // async logout(sessionId) {
    //     try {
    //         console.log("Using session ID:", sessionId);
    //         const userClient = new sdk.Client()
    //             .setEndpoint(process.env.APPWRITE_HOST_URL)
    //             .setProject(process.env.APPWRITE_PROJECT_ID);
    //         userClient.headers['X-Appwrite-Session'] = sessionId;
    //         const accountWithSession = new sdk.Account(userClient);
    //         await accountWithSession.deleteSession('current');
    //     } catch (error) {
    //         throw new ApiError(500, "Failed to logout", [error.message]);
    //     }
    // }
}

const authService = new AuthService();
export default authService;