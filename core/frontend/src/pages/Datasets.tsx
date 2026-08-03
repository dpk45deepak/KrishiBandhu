// frontend/src/pages/Datasets.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { apiClient } from "../lib/api";
import {
  Plus,
  Upload,
  Search,
  Filter,
  Trash2,
  Database,
  Clock,
  MoreVertical,
  Download,
  BarChart3,
} from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";

interface Dataset {
  id: string;
  name: string;
  description: string;
  status: string;
  format: string;
  tags: string[];
  current_version?: {
    version_number: number;
    row_count: number;
    column_count: number;
    size_bytes: number;
  };
  created_at: string;
}

export default function Datasets() {
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [showCreate, setShowCreate] = useState(false);
  const queryClient = useQueryClient();

  const { data: datasets, isLoading } = useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiClient.get("/datasets").then((r) => r.data as Dataset[]),
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.delete(`/datasets/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      toast.success("Dataset deleted");
    },
  });

  const filtered = datasets?.filter((ds) => {
    if (search && !ds.name.toLowerCase().includes(search.toLowerCase()))
      return false;
    if (statusFilter && ds.status !== statusFilter) return false;
    return true;
  });

  const statusBadge = (status: string) => {
    const styles: Record<string, string> = {
      ready: "badge-success",
      error: "badge-error",
      uploading: "badge-info",
      scanning: "badge-info",
      profiling: "badge-info",
      validating: "badge-warning",
      cleaning: "badge-warning",
    };
    return clsx("badge", styles[status] || "badge-info");
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-500 mt-1">
            Manage your agricultural datasets
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Dataset
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search datasets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="input-field pl-10"
          />
        </div>
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="input-field w-40"
        >
          <option value="">All Status</option>
          <option value="ready">Ready</option>
          <option value="uploading">Uploading</option>
          <option value="error">Error</option>
        </select>
      </div>

      {/* Dataset grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 bg-gray-200 rounded w-3/4 mb-3" />
              <div className="h-3 bg-gray-200 rounded w-full mb-2" />
              <div className="h-3 bg-gray-200 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered?.map((dataset) => (
            <Link
              key={dataset.id}
              to={`/datasets/${dataset.id}`}
              className="card hover:shadow-md transition-shadow group"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-agri-600" />
                  <h3 className="font-semibold text-gray-900">
                    {dataset.name}
                  </h3>
                </div>
                <span className={statusBadge(dataset.status)}>
                  {dataset.status}
                </span>
              </div>

              {dataset.description && (
                <p className="text-sm text-gray-500 mb-3 line-clamp-2">
                  {dataset.description}
                </p>
              )}

              {dataset.current_version && (
                <div className="flex gap-4 text-xs text-gray-500 mb-3">
                  <span>
                    {dataset.current_version.row_count?.toLocaleString()} rows
                  </span>
                  <span>{dataset.current_version.column_count} cols</span>
                  <span>v{dataset.current_version.version_number}</span>
                </div>
              )}

              <div className="flex items-center justify-between">
                <div className="flex gap-1">
                  {dataset.tags?.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="flex items-center gap-1 text-gray-400">
                  <Clock className="w-3 h-3" />
                  <span className="text-xs">
                    {new Date(dataset.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* Empty state */}
      {!isLoading && filtered?.length === 0 && (
        <div className="text-center py-12">
          <Database className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <h3 className="text-lg font-medium text-gray-900">
            No datasets found
          </h3>
          <p className="text-gray-500 mt-1">
            {search
              ? "Try adjusting your search"
              : "Upload your first dataset to get started"}
          </p>
        </div>
      )}

      {/* Create dataset modal */}
      {showCreate && (
        <CreateDatasetModal onClose={() => setShowCreate(false)} />
      )}
    </div>
  );
}

function CreateDatasetModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [format, setFormat] = useState("csv");
  const queryClient = useQueryClient();

  const createMutation = useMutation({
    mutationFn: (data: { name: string; description: string; format: string }) =>
      apiClient.post("/datasets", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
      toast.success("Dataset created");
      onClose();
    },
  });

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-md">
        <h2 className="text-lg font-semibold mb-4">Create Dataset</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            createMutation.mutate({ name, description, format });
          }}
          className="space-y-4"
        >
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input-field"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input-field"
              rows={3}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Format
            </label>
            <select
              value={format}
              onChange={(e) => setFormat(e.target.value)}
              className="input-field"
            >
              <option value="csv">CSV</option>
              <option value="parquet">Parquet</option>
              <option value="json">JSON</option>
              <option value="geojson">GeoJSON</option>
              <option value="tiff">TIFF</option>
            </select>
          </div>
          <div className="flex gap-3 justify-end pt-2">
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="btn-primary"
            >
              {createMutation.isPending ? "Creating..." : "Create"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
