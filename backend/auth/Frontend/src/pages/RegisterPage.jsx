import { useState } from "react";
import { User, Lock, Phone, Mail, ArrowRight, AlertCircle } from "lucide-react";
import { useNavigate } from "react-router-dom";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:3000";

export default function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [username, setUsername] = useState("");
  const [phone, setPhone] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);
    try {
      const res = await axios.post(
        `${API_BASE}/api/auth/register`,
        { email, password, username, phone },
        { headers: { "Content-Type": "application/json" }, timeout: 15000 }
      );
      navigate("/login");
    } catch (err) {
      const message =
        err?.response?.data?.message ||
        err?.response?.data ||
        err?.message ||
        "Registration failed";
      setError("Registration failed: " + message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Main Card */}
        <div className="bg-slate-800/60 backdrop-blur-xl rounded-3xl border border-slate-700/50 shadow-2xl p-8">
          {/* Header Section */}
          <div className="mb-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-12 h-12 bg-emerald-500 rounded-2xl flex items-center justify-center">
                <User className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-white">Register</h1>
            </div>
            <p className="text-slate-400 text-base">Create your account to get started</p>
          </div>

          {/* Form Section */}
          <form className="space-y-5" onSubmit={handleSubmit}>
            {/* Username Field */}
            <div>
              <div className="relative">
                <input
                  type="text"
                  placeholder="Username"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  required
                  className="w-full bg-slate-700/50 border border-slate-600/50 rounded-2xl px-4 py-4 pr-12 text-white placeholder-slate-400 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all duration-300 outline-none"
                />
                <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                  <User className="w-5 h-5 text-slate-400" />
                </div>
              </div>
            </div>

            {/* Email Field */}
            <div>
              <div className="relative">
                <input
                  type="email"
                  placeholder="Email address"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  className="w-full bg-slate-700/50 border border-slate-600/50 rounded-2xl px-4 py-4 pr-12 text-white placeholder-slate-400 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all duration-300 outline-none"
                />
                <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                  <Mail className="w-5 h-5 text-slate-400" />
                </div>
              </div>
            </div>

            {/* Phone Field */}
            <div>
              <div className="relative">
                <input
                  type="tel"
                  placeholder="Phone"
                  value={phone}
                  onChange={e => setPhone(e.target.value)}
                  required
                  className="w-full bg-slate-700/50 border border-slate-600/50 rounded-2xl px-4 py-4 pr-12 text-white placeholder-slate-400 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all duration-300 outline-none"
                />
                <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                  <Phone className="w-5 h-5 text-slate-400" />
                </div>
              </div>
            </div>

            {/* Password Field */}
            <div>
              <div className="relative">
                <input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  className="w-full bg-slate-700/50 border border-slate-600/50 rounded-2xl px-4 py-4 pr-12 text-white placeholder-slate-400 focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/20 transition-all duration-300 outline-none"
                />
                <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                  <Lock className="w-5 h-5 text-slate-400" />
                </div>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="flex items-center gap-3 p-4 bg-red-900/30 border border-red-800/50 rounded-2xl backdrop-blur-sm">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 flex items-center justify-center gap-3 group transform hover:scale-[1.02] active:scale-[0.98] mt-8"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  <span>Registering...</span>
                </>
              ) : (
                <>
                  <span>Register</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </>
              )}
            </button>
          </form>
        </div>

        {/* Bottom Accent Line */}
        <div className="mt-8 flex justify-center">
          <div className="w-16 h-1 bg-emerald-500 rounded-full"></div>
        </div>
      </div>
    </div>
  );
}