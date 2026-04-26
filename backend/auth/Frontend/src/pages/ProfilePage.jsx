import { useEffect, useState } from "react";
import { User, Mail, Phone, Loader2, AlertCircle } from "lucide-react";
import { account } from "../api/appwrite.js";
import LogoutButton from "../components/LogoutButton.jsx";

export default function ProfilePage() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    account.get()
      .then(setUser)
      .catch(() => setError("Failed to load user info"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-slate-800/60 backdrop-blur-xl rounded-3xl border border-slate-700/50 shadow-2xl p-8">
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-emerald-500 rounded-2xl flex items-center justify-center">
                <User className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-white">Profile</h1>
            </div>
            <p className="text-slate-400 text-base">Your account details</p>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 text-emerald-400 animate-spin" />
              <span className="ml-3 text-slate-300">Loading...</span>
            </div>
          ) : error ? (
            <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-800/50 rounded-2xl backdrop-blur-sm">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <User className="w-5 h-5 text-emerald-400" />
                <span className="text-white font-semibold">{user.name || "No username"}</span>
              </div>
              <div className="flex items-center gap-3">
                <Mail className="w-5 h-5 text-emerald-400" />
                <span className="text-white">{user.email}</span>
              </div>
              {user.phone && (
                <div className="flex items-center gap-3">
                  <Phone className="w-5 h-5 text-emerald-400" />
                  <span className="text-white">{user.phone}</span>
                </div>
              )}
              <div className="pt-6">
                <LogoutButton />
              </div>
            </div>
          )}
        </div>
        <div className="mt-8 flex justify-center">
          <div className="w-16 h-1 bg-emerald-500 rounded-full"></div>
        </div>
      </div>
    </div>
  );
}