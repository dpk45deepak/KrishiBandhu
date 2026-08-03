// frontend/src/components/Layout.tsx
import { useState } from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Database,
  GitBranch,
  Brain,
  Layers,
  Zap,
  FileText,
  Activity,
  ChevronLeft,
  ChevronRight,
  LogOut,
  User,
  Sprout,
} from "lucide-react";
import { useAuthStore } from "../lib/hooks/useAuth";
import clsx from "clsx";

const navigation = [
  { name: "Dashboard", href: "/", icon: LayoutDashboard },
  { name: "Datasets", href: "/datasets", icon: Database },
  { name: "Pipelines", href: "/pipelines", icon: GitBranch },
  { name: "Models", href: "/models", icon: Brain },
  { name: "Feature Store", href: "/features", icon: Layers },
  { name: "Inference", href: "/inference", icon: Zap },
  { name: "Reports", href: "/reports", icon: FileText },
  { name: "Monitoring", href: "/monitoring", icon: Activity },
];

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside
        className={clsx(
          "bg-white border-r border-gray-200 flex flex-col transition-all duration-300",
          collapsed ? "w-16" : "w-64",
        )}
      >
        {/* Logo */}
        <div className="flex items-center gap-2 px-4 h-16 border-b border-gray-200">
          <Sprout className="w-8 h-8 text-agri-600 flex-shrink-0" />
          {!collapsed && (
            <span className="text-xl font-bold text-agri-800">AgriMind</span>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => (
            <NavLink
              key={item.name}
              to={item.href}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-agri-50 text-agri-700"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900",
                )
              }
            >
              <item.icon className="w-5 h-5 flex-shrink-0" />
              {!collapsed && <span>{item.name}</span>}
            </NavLink>
          ))}
        </nav>

        {/* User section */}
        <div className="border-t border-gray-200 p-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-agri-100 flex items-center justify-center flex-shrink-0">
              <User className="w-4 h-4 text-agri-600" />
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {user?.full_name || user?.username}
                </p>
                <p className="text-xs text-gray-500">{user?.role}</p>
              </div>
            )}
            <button
              onClick={handleLogout}
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Collapse button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="absolute bottom-20 left-0 right-0 mx-auto w-6 h-6 rounded-full bg-white border border-gray-200 
                     flex items-center justify-center text-gray-400 hover:text-gray-600"
          style={{ marginLeft: collapsed ? "2.5rem" : "15.5rem" }}
        >
          {collapsed ? (
            <ChevronRight className="w-3 h-3" />
          ) : (
            <ChevronLeft className="w-3 h-3" />
          )}
        </button>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto bg-gray-50">
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
