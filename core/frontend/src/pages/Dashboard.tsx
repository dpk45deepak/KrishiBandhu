// frontend/src/pages/Dashboard.tsx
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import {
  Database,
  GitBranch,
  Brain,
  Zap,
  TrendingUp,
  AlertTriangle,
  Clock,
  CheckCircle2,
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

export default function Dashboard() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: () => apiClient.get("/health").then((r) => r.data),
    refetchInterval: 30000,
  });

  const { data: metrics } = useQuery({
    queryKey: ["systemMetrics"],
    queryFn: () => apiClient.get("/monitoring/system").then((r) => r.data),
    refetchInterval: 15000,
  });

  const stats = [
    {
      name: "Datasets",
      value: metrics?.feature_groups || 0,
      icon: Database,
      color: "text-blue-600",
      bg: "bg-blue-100",
    },
    {
      name: "Active Pipelines",
      value: metrics?.active_pipelines || 0,
      icon: GitBranch,
      color: "text-purple-600",
      bg: "bg-purple-100",
    },
    {
      name: "Models Deployed",
      value: metrics?.models_deployed || 0,
      icon: Brain,
      color: "text-agri-600",
      bg: "bg-agri-100",
    },
    {
      name: "Predictions Today",
      value: metrics?.total_predictions_today?.toLocaleString() || "0",
      icon: Zap,
      color: "text-amber-600",
      bg: "bg-amber-100",
    },
  ];

  const mockChartData = [
    { time: "00:00", requests: 45, latency: 120 },
    { time: "04:00", requests: 30, latency: 100 },
    { time: "08:00", requests: 80, latency: 140 },
    { time: "12:00", requests: 120, latency: 160 },
    { time: "16:00", requests: 95, latency: 130 },
    { time: "20:00", requests: 60, latency: 110 },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">Platform overview and key metrics</p>
      </div>

      {/* Health status */}
      {health && (
        <div
          className={`
          flex items-center gap-3 p-4 rounded-xl border
          ${
            health.status === "healthy"
              ? "bg-green-50 border-green-200 text-green-800"
              : "bg-yellow-50 border-yellow-200 text-yellow-800"
          }
        `}
        >
          <div
            className={`
            w-3 h-3 rounded-full 
            ${health.status === "healthy" ? "bg-green-500" : "bg-yellow-500"}
            animate-pulse
          `}
          />
          <span className="font-medium">
            System {health.status.toUpperCase()}
          </span>
          <span className="text-sm opacity-75">
            v{health.version} | Uptime:{" "}
            {Math.floor(health.uptime_seconds / 3600)}h{" "}
            {Math.floor((health.uptime_seconds % 3600) / 60)}m
          </span>
        </div>
      )}

      {/* Stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat) => (
          <div key={stat.name} className="stat-card">
            <div className={`p-3 rounded-xl ${stat.bg}`}>
              <stat.icon className={`w-6 h-6 ${stat.color}`} />
            </div>
            <div>
              <p className="text-sm text-gray-500">{stat.name}</p>
              <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
            </div>
          </div>
        ))}
      </div>

      {/* System resources */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card">
            <h3 className="text-sm font-medium text-gray-500">CPU Usage</h3>
            <div className="mt-2">
              <div className="flex justify-between text-sm">
                <span className="font-bold text-2xl">
                  {metrics.cpu_percent?.toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${metrics.cpu_percent}%` }}
                />
              </div>
            </div>
          </div>
          <div className="card">
            <h3 className="text-sm font-medium text-gray-500">Memory Usage</h3>
            <div className="mt-2">
              <div className="flex justify-between text-sm">
                <span className="font-bold text-2xl">
                  {metrics.memory_percent?.toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-agri-600 h-2 rounded-full transition-all"
                  style={{ width: `${metrics.memory_percent}%` }}
                />
              </div>
            </div>
          </div>
          <div className="card">
            <h3 className="text-sm font-medium text-gray-500">Avg Latency</h3>
            <p className="text-2xl font-bold mt-2">
              {metrics.avg_latency_ms?.toFixed(0)} ms
            </p>
            <p className="text-xs text-gray-500 mt-1">
              {metrics.api_requests_per_minute?.toFixed(0)} req/min
            </p>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-4">API Requests</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={mockChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="requests"
                stroke="#22c55e"
                fill="#22c55e20"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-4">Latency (ms)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mockChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Bar dataKey="latency" fill="#eab308" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent activity */}
      <div className="card">
        <h3 className="font-medium text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-3">
          {[
            {
              icon: CheckCircle2,
              color: "text-green-500",
              text: 'Pipeline "data-prep" completed successfully',
              time: "2 min ago",
            },
            {
              icon: Brain,
              color: "text-purple-500",
              text: 'Model "crop-yield-v2" training finished',
              time: "15 min ago",
            },
            {
              icon: AlertTriangle,
              color: "text-yellow-500",
              text: "Data quality warning: missing values in soil_data",
              time: "1 hour ago",
            },
            {
              icon: Database,
              color: "text-blue-500",
              text: 'Dataset "satellite-imagery" uploaded',
              time: "3 hours ago",
            },
          ].map((activity, i) => (
            <div key={i} className="flex items-center gap-3 py-2">
              <activity.icon className={`w-4 h-4 ${activity.color}`} />
              <span className="text-sm text-gray-600 flex-1">
                {activity.text}
              </span>
              <Clock className="w-3 h-3 text-gray-400" />
              <span className="text-xs text-gray-400">{activity.time}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
