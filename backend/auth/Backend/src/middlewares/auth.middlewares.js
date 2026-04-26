// import * as sdk from "node-appwrite";

// export async function requireAuth(req, res, next) {
//   const sessionId = req.headers['x-appwrite-session'];
//   if (!sessionId) {
//     return res.status(401).json({ message: "Unauthorized: No session ID provided" });
//   }

//   try {
//     const userClient = new sdk.Client()
//       .setEndpoint(process.env.APPWRITE_HOST_URL)
//       .setProject(process.env.APPWRITE_PROJECT_ID);
//     userClient.headers['X-Appwrite-Session'] = sessionId;
//     const accountWithSession = new sdk.Account(userClient);
//     const user = await accountWithSession.get();
//     req.user = user;
//     next();
//   } catch (error) {
//     return res.status(401).json({ message: "Unauthorized: Invalid session", error: error.message });
//   }
// }