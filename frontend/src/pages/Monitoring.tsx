// frontend/src/pages/Monitoring.tsx
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import {
  Activity,
  Cpu,
  HardDrive,
  AlertTriangle,
  Bell,
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
import clsx from "clsx";

export default function Monitoring() {
  const { data: metrics, isLoading } = useQuery({
    queryKey: ["systemMetrics"],
    queryFn: () => apiClient.get("/monitoring/system").then((r) => r.data),
    refetchInterval: 10000,
  });

  const { data: alerts } = useQuery({
    queryKey: ["alerts"],
    queryFn: () => apiClient.get("/monitoring/alerts").then((r) => r.data),
    refetchInterval: 15000,
  });

  const resourceData = [
    { name: "CPU", value: metrics?.cpu_percent || 0, color: "#3b82f6" },
    { name: "Memory", value: metrics?.memory_percent || 0, color: "#22c55e" },
    { name: "Disk", value: metrics?.disk_percent || 0, color: "#eab308" },
  ];

  const mockLatencyData = Array.from({ length: 24 }, (_, i) => ({
    time: `${i}:00`,
    p50: 80 + Math.random() * 40,
    p95: 120 + Math.random() * 80,
    p99: 200 + Math.random() * 150,
  }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Monitoring</h1>
        <p className="text-gray-500 mt-1">System metrics and alerts</p>
      </div>

      {/* Resource gauges */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {resourceData.map((res) => (
          <div key={res.name} className="card">
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-medium text-gray-700">{res.name} Usage</h3>
              {res.name === "CPU" && <Cpu className="w-5 h-5 text-blue-500" />}
              {res.name === "Memory" && (
                <HardDrive className="w-5 h-5 text-green-500" />
              )}
              {res.name === "Disk" && (
                <HardDrive className="w-5 h-5 text-amber-500" />
              )}
            </div>
            <div className="relative pt-1">
              <div className="flex justify-between mb-1">
                <span className="text-3xl font-bold">
                  {res.value.toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className="h-3 rounded-full transition-all duration-500"
                  style={{
                    width: `${Math.min(res.value, 100)}%`,
                    backgroundColor: res.color,
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="font-medium mb-4">Latency Percentiles</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={mockLatencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="p50"
                stroke="#22c55e"
                fill="#22c55e10"
                strokeWidth={2}
                name="P50"
              />
              <Area
                type="monotone"
                dataKey="p95"
                stroke="#eab308"
                fill="#eab30810"
                strokeWidth={2}
                name="P95"
              />
              <Area
                type="monotone"
                dataKey="p99"
                stroke="#ef4444"
                fill="#ef444410"
                strokeWidth={2}
                name="P99"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 className="font-medium mb-4">Request Throughput</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={mockLatencyData.slice(-12)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="time" fontSize={12} />
              <YAxis fontSize={12} />
              <Tooltip />
              <Bar
                dataKey="p50"
                fill="#22c55e"
                radius={[4, 4, 0, 0]}
                name="Requests/min"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Active alerts */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-medium flex items-center gap-2">
            <Bell className="w-5 h-5" />
            Active Alerts
          </h3>
          {alerts?.length > 0 && (
            <span className="badge badge-error">{alerts.length} active</span>
          )}
        </div>

        {alerts?.length > 0 ? (
          <div className="space-y-2">
            {alerts.map((alert: any) => (
              <div
                key={alert.id}
                className={clsx(
                  "flex items-start gap-3 p-3 rounded-lg border",
                  alert.severity === "critical"
                    ? "bg-red-50 border-red-200"
                    : alert.severity === "warning"
                      ? "bg-yellow-50 border-yellow-200"
                      : "bg-blue-50 border-blue-200",
                )}
              >
                <AlertTriangle
                  className={clsx(
                    "w-5 h-5 flex-shrink-0 mt-0.5",
                    alert.severity === "critical"
                      ? "text-red-500"
                      : alert.severity === "warning"
                        ? "text-yellow-500"
                        : "text-blue-500",
                  )}
                />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-sm">{alert.rule_name}</p>
                    <span
                      className={clsx(
                        "badge text-xs",
                        alert.severity === "critical"
                          ? "badge-error"
                          : alert.severity === "warning"
                            ? "badge-warning"
                            : "badge-info",
                      )}
                    >
                      {alert.severity}
                    </span>
                  </div>
                  <p className="text-sm mt-1">{alert.message}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    {new Date(alert.triggered_at).toLocaleString()} • Value:{" "}
                    {alert.current_value?.toFixed(2)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <CheckCircle2 className="w-8 h-8 text-green-400 mx-auto mb-2" />
            <p className="text-sm text-gray-500">No active alerts</p>
          </div>
        )}
      </div>

      {/* Platform stats */}
      {metrics && (
        <div className="card">
          <h3 className="font-medium mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Platform Stats
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-blue-600">
                {metrics.api_requests_per_minute?.toFixed(0)}
              </p>
              <p className="text-xs text-gray-500">Req/min</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-green-600">
                {metrics.active_pipelines}
              </p>
              <p className="text-xs text-gray-500">Active Pipelines</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-purple-600">
                {metrics.models_deployed}
              </p>
              <p className="text-xs text-gray-500">Models Deployed</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-amber-600">
                {metrics.total_predictions_today?.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500">Predictions Today</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
