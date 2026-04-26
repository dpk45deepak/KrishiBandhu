import { LogOut } from "lucide-react";
import { account } from "../api/appwrite";
import { useNavigate } from "react-router-dom";
import { useState } from "react";

export default function LogoutButton() {
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleLogout = async () => {
    setLoading(true);
    try {
      await account.deleteSession("current");
    } catch (err) {
      alert("Logout failed: " + (err.message || "Unknown error"));
    }
    setLoading(false);
    navigate("/login");
  };

  return (
    <button
      onClick={handleLogout}
      disabled={loading}
      className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white font-semibold py-3 px-6 rounded-2xl shadow-md transition-all duration-300 disabled:opacity-60 disabled:cursor-not-allowed"
    >
      <LogOut className="w-5 h-5" />
      {loading ? "Logging out..." : "Logout"}
    </button>
  );
}