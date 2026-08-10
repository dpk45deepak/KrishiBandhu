// frontend/src/pages/Login.tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useLogin } from "../lib/hooks/useAuth";
import { Sprout, LogIn } from "lucide-react";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const login = useLogin();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login.mutateAsync({ username, password });
      navigate("/");
    } catch {}
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-agri-50 to-earth-50">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-agri-100 mb-4">
            <Sprout className="w-8 h-8 text-agri-600" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900">AgriMind AI</h1>
          <p className="text-gray-500 mt-2">
            Integrated Agricultural Intelligence Platform
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-8">
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="input-field"
                placeholder="Enter your username"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="input-field"
                placeholder="Enter your password"
                required
              />
            </div>

            <button
              type="submit"
              disabled={login.isPending}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {login.isPending ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <LogIn className="w-4 h-4" />
              )}
              {login.isPending ? "Signing in..." : "Sign in"}
            </button>
          </form>

          {login.isError && (
            <p className="text-red-500 text-sm text-center mt-4">
              Invalid credentials. Please try again.
            </p>
          )}

          <p className="text-xs text-gray-400 text-center mt-6">
            Default admin: admin / admin123
          </p>
        </div>
      </div>
    </div>
  );
}
