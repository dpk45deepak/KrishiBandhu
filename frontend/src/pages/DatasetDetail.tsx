// frontend/src/pages/DatasetDetail.tsx
import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import {
  ArrowLeft,
  Upload,
  BarChart3,
  CheckCircle2,
  Brush,
  Download,
  Trash2,
  RefreshCw,
  AlertTriangle,
  FileText,
  Clock,
  ChevronDown,
  ChevronRight,
  XCircle,
} from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

export default function DatasetDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<
    "overview" | "profile" | "validation" | "versions"
  >("overview");
  const [expandedCol, setExpandedCol] = useState<string | null>(null);

  const { data: dataset, isLoading } = useQuery({
    queryKey: ["dataset", id],
    queryFn: () => apiClient.get(`/datasets/${id}`).then((r) => r.data),
    enabled: !!id,
  });

  const profileMutation = useMutation({
    mutationFn: () => apiClient.post(`/datasets/${id}/profile`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      toast.success("Profile generated");
    },
  });

  const validateMutation = useMutation({
    mutationFn: () => apiClient.post(`/datasets/${id}/validate`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      toast.success("Validation complete");
    },
  });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      return apiClient.post(`/datasets/${id}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dataset", id] });
      toast.success("File uploaded");
    },
  });

  const deleteMutation = useMutation({
    mutationFn: () => apiClient.delete(`/datasets/${id}`),
    onSuccess: () => {
      toast.success("Dataset deleted");
      navigate("/datasets");
    },
  });

  if (isLoading) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="h-8 bg-gray-200 rounded w-1/3" />
        <div className="h-4 bg-gray-200 rounded w-1/2" />
        <div className="h-64 bg-gray-200 rounded" />
      </div>
    );
  }

  if (!dataset) {
    return (
      <div className="text-center py-12">
        <XCircle className="w-12 h-12 text-gray-300 mx-auto mb-3" />
        <h3 className="text-lg font-medium">Dataset not found</h3>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate("/datasets")}
            className="p-2 rounded-lg hover:bg-gray-100"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{dataset.name}</h1>
            <p className="text-gray-500">{dataset.description}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => profileMutation.mutate()}
            disabled={profileMutation.isPending}
            className="btn-secondary flex items-center gap-2"
          >
            <BarChart3 className="w-4 h-4" />
            Profile
          </button>
          <button
            onClick={() => validateMutation.mutate()}
            disabled={validateMutation.isPending}
            className="btn-secondary flex items-center gap-2"
          >
            <CheckCircle2 className="w-4 h-4" />
            Validate
          </button>
          <label className="btn-primary flex items-center gap-2 cursor-pointer">
            <Upload className="w-4 h-4" />
            Upload
            <input
              type="file"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadMutation.mutate(file);
              }}
            />
          </label>
        </div>
      </div>

      {/* Stats bar */}
      {dataset.current_version && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card text-center">
            <p className="text-3xl font-bold text-agri-600">
              {dataset.current_version.row_count?.toLocaleString()}
            </p>
            <p className="text-sm text-gray-500">Rows</p>
          </div>
          <div className="card text-center">
            <p className="text-3xl font-bold text-blue-600">
              {dataset.current_version.column_count}
            </p>
            <p className="text-sm text-gray-500">Columns</p>
          </div>
          <div className="card text-center">
            <p className="text-3xl font-bold text-purple-600">
              v{dataset.current_version.version_number}
            </p>
            <p className="text-sm text-gray-500">Version</p>
          </div>
          <div className="card text-center">
            <p className="text-3xl font-bold text-amber-600">
              {(
                (dataset.current_version.size_bytes || 0) /
                (1024 * 1024)
              ).toFixed(1)}{" "}
              MB
            </p>
            <p className="text-sm text-gray-500">Size</p>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 bg-gray-100 p-1 rounded-lg w-fit">
        {(["overview", "profile", "validation", "versions"] as const).map(
          (tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                "px-4 py-2 rounded-md text-sm font-medium transition-colors capitalize",
                activeTab === tab
                  ? "bg-white text-gray-900 shadow-sm"
                  : "text-gray-500 hover:text-gray-700",
              )}
            >
              {tab}
            </button>
          ),
        )}
      </div>

      {/* Tab content */}
      {activeTab === "overview" && (
        <div className="space-y-4">
          <div className="card">
            <h3 className="font-medium mb-3">Dataset Info</h3>
            <dl className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-gray-500">Format</dt>
                <dd className="font-medium">{dataset.format?.toUpperCase()}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Status</dt>
                <dd>
                  <span
                    className={clsx(
                      "badge",
                      dataset.status === "ready"
                        ? "badge-success"
                        : "badge-warning",
                    )}
                  >
                    {dataset.status}
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">Created</dt>
                <dd className="font-medium">
                  {new Date(dataset.created_at).toLocaleString()}
                </dd>
              </div>
              <div>
                <dt className="text-gray-500">Tags</dt>
                <dd className="flex gap-1 flex-wrap">
                  {dataset.tags?.map((tag: string) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 rounded text-xs bg-gray-100"
                    >
                      {tag}
                    </span>
                  ))}
                </dd>
              </div>
            </dl>
          </div>

          {dataset.metadata && Object.keys(dataset.metadata).length > 0 && (
            <div className="card">
              <h3 className="font-medium mb-3">Metadata</h3>
              <pre className="text-sm bg-gray-50 p-3 rounded-lg overflow-auto">
                {JSON.stringify(dataset.metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}

      {activeTab === "profile" && (
        <div className="space-y-4">
          {dataset.profile ? (
            <>
              <div className="card">
                <h3 className="font-medium mb-4">Column Profiles</h3>
                <div className="space-y-2">
                  {dataset.profile.columns?.map((col: any) => (
                    <div key={col.name} className="border rounded-lg">
                      <button
                        onClick={() =>
                          setExpandedCol(
                            expandedCol === col.name ? null : col.name,
                          )
                        }
                        className="w-full flex items-center justify-between p-3 hover:bg-gray-50"
                      >
                        <div className="flex items-center gap-3">
                          <span className="font-medium">{col.name}</span>
                          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
                            {col.dtype}
                          </span>
                          {col.null_percentage > 0 && (
                            <span className="text-xs text-amber-600 flex items-center gap-1">
                              <AlertTriangle className="w-3 h-3" />
                              {col.null_percentage.toFixed(1)}% missing
                            </span>
                          )}
                        </div>
                        {expandedCol === col.name ? (
                          <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                      </button>
                      {expandedCol === col.name && (
                        <div className="px-3 pb-3 border-t">
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3 text-sm">
                            <div>
                              <span className="text-gray-500">Count:</span>
                              <span className="ml-1 font-medium">
                                {col.count?.toLocaleString()}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-500">Unique:</span>
                              <span className="ml-1 font-medium">
                                {col.unique_count?.toLocaleString()}
                              </span>
                            </div>
                            {col.mean !== null && (
                              <>
                                <div>
                                  <span className="text-gray-500">Mean:</span>
                                  <span className="ml-1 font-medium">
                                    {col.mean?.toFixed(2)}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-500">Std:</span>
                                  <span className="ml-1 font-medium">
                                    {col.std?.toFixed(2)}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-500">Min:</span>
                                  <span className="ml-1 font-medium">
                                    {col.min}
                                  </span>
                                </div>
                                <div>
                                  <span className="text-gray-500">Max:</span>
                                  <span className="ml-1 font-medium">
                                    {col.max}
                                  </span>
                                </div>
                              </>
                            )}
                          </div>
                          {col.top_values && col.top_values.length > 0 && (
                            <div className="mt-3">
                              <p className="text-xs text-gray-500 mb-1">
                                Top Values
                              </p>
                              <div className="h-24">
                                <ResponsiveContainer width="100%" height="100%">
                                  <BarChart data={col.top_values.slice(0, 10)}>
                                    <XAxis dataKey="value" fontSize={10} />
                                    <Tooltip />
                                    <Bar
                                      dataKey="count"
                                      fill="#22c55e"
                                      radius={[2, 2, 0, 0]}
                                    />
                                  </BarChart>
                                </ResponsiveContainer>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {dataset.profile.correlations && (
                <div className="card">
                  <h3 className="font-medium mb-4">Correlations</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr>
                          <th className="p-2"></th>
                          {Object.keys(dataset.profile.correlations)
                            .slice(0, 10)
                            .map((col) => (
                              <th key={col} className="p-2 font-medium text-xs">
                                {col}
                              </th>
                            ))}
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(dataset.profile.correlations)
                          .slice(0, 10)
                          .map(([row, vals]: [string, any]) => (
                            <tr key={row}>
                              <td className="p-2 font-medium text-xs">{row}</td>
                              {Object.values(vals)
                                .slice(0, 10)
                                .map((val: any, i: number) => (
                                  <td
                                    key={i}
                                    className="p-2 text-center"
                                    style={{
                                      backgroundColor: `rgba(34, 197, 94, ${Math.abs(val)})`,
                                      color:
                                        Math.abs(val) > 0.5
                                          ? "white"
                                          : "inherit",
                                    }}
                                  >
                                    {val.toFixed(2)}
                                  </td>
                                ))}
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <BarChart3 className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <h3 className="font-medium">No profile yet</h3>
              <p className="text-gray-500 text-sm mt-1">
                Click "Profile" to generate statistics
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === "validation" && (
        <div className="card">
          <h3 className="font-medium mb-4">Validation</h3>
          <p className="text-gray-500 text-sm">
            Run validation to check data quality
          </p>
        </div>
      )}

      {activeTab === "versions" && (
        <div className="card">
          <h3 className="font-medium mb-4">Version History</h3>
          <div className="space-y-3">
            {dataset.current_version && (
              <div className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                <div className="w-10 h-10 rounded-full bg-agri-100 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-agri-600" />
                </div>
                <div className="flex-1">
                  <p className="font-medium">
                    Version {dataset.current_version.version_number}
                  </p>
                  <p className="text-sm text-gray-500">
                    {dataset.current_version.row_count?.toLocaleString()} rows •{" "}
                    {new Date(
                      dataset.current_version.created_at || dataset.created_at,
                    ).toLocaleString()}
                  </p>
                </div>
                <span className="badge badge-success">Current</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Danger zone */}
      <div className="card border-red-200 bg-red-50">
        <h3 className="font-medium text-red-800 mb-2">Danger Zone</h3>
        <p className="text-sm text-red-600 mb-3">
          Deleting this dataset will remove all versions and data permanently.
        </p>
        <button
          onClick={() => {
            if (confirm("Are you sure you want to delete this dataset?")) {
              deleteMutation.mutate();
            }
          }}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-600 text-white hover:bg-red-700 transition-colors text-sm"
        >
          <Trash2 className="w-4 h-4" />
          Delete Dataset
        </button>
      </div>
    </div>
  );
}
