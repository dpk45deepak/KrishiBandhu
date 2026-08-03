// frontend/src/pages/Models.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { apiClient } from "../lib/api";
import { Brain, Plus, TrendingUp, Zap, Clock, BarChart3 } from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";

interface Model {
  id: string;
  name: string;
  model_type: string;
  status: string;
  description: string;
  current_version?: {
    version: number;
    metrics: Record<string, number>;
    deployed: boolean;
  };
  created_at: string;
}

export default function Models() {
  const [showRegister, setShowRegister] = useState(false);
  const queryClient = useQueryClient();

  const { data: models, isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: () => apiClient.get("/ml/models").then((r) => r.data as Model[]),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Registry</h1>
          <p className="text-gray-500 mt-1">Manage your ML models</p>
        </div>
        <button
          onClick={() => setShowRegister(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          Register Model
        </button>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-5 bg-gray-200 rounded w-1/2 mb-3" />
              <div className="h-4 bg-gray-200 rounded w-full mb-2" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {models?.map((model) => (
            <Link
              key={model.id}
              to={`/models/${model.id}`}
              className="card hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-agri-600" />
                  <h3 className="font-semibold">{model.name}</h3>
                </div>
                <span
                  className={clsx(
                    "badge",
                    model.status === "deployed"
                      ? "badge-success"
                      : model.status === "failed"
                        ? "badge-error"
                        : model.status === "ready"
                          ? "badge-info"
                          : "badge-warning",
                  )}
                >
                  {model.status}
                </span>
              </div>

              <p className="text-sm text-gray-500 mb-3 line-clamp-2">
                {model.description}
              </p>

              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <BarChart3 className="w-3 h-3" />
                  {model.model_type}
                </span>
                {model.current_version && (
                  <>
                    <span>v{model.current_version.version}</span>
                    {model.current_version.deployed && (
                      <span className="badge badge-success text-xs">
                        Deployed
                      </span>
                    )}
                  </>
                )}
                <span className="flex items-center gap-1 ml-auto">
                  <Clock className="w-3 h-3" />
                  {new Date(model.created_at).toLocaleDateString()}
                </span>
              </div>

              {model.current_version?.metrics && (
                <div className="mt-3 pt-3 border-t grid grid-cols-2 gap-2">
                  {Object.entries(model.current_version.metrics)
                    .slice(0, 2)
                    .map(([key, val]) => (
                      <div key={key} className="text-center">
                        <p className="text-lg font-bold text-agri-600">
                          {typeof val === "number" ? val.toFixed(3) : val}
                        </p>
                        <p className="text-xs text-gray-500 capitalize">
                          {key.replace("_", " ")}
                        </p>
                      </div>
                    ))}
                </div>
              )}
            </Link>
          ))}
        </div>
      )}

      {!isLoading && models?.length === 0 && (
        <div className="text-center py-12">
          <Brain className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <h3 className="font-medium">No models registered</h3>
          <p className="text-gray-500 text-sm mt-1">
            Register your first model to get started
          </p>
        </div>
      )}

      {showRegister && (
        <RegisterModelModal onClose={() => setShowRegister(false)} />
      )}
    </div>
  );
}

function RegisterModelModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [modelType, setModelType] = useState("classification");
  const [description, setDescription] = useState("");
  const queryClient = useQueryClient();

  const registerMutation = useMutation({
    mutationFn: (data: any) => apiClient.post("/ml/models", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      toast.success("Model registered");
      onClose();
    },
  });

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-md">
        <h2 className="text-lg font-semibold mb-4">Register Model</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            registerMutation.mutate({
              name,
              model_type: modelType,
              description,
            });
          }}
          className="space-y-4"
        >
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input-field"
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Type</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="input-field"
            >
              <option value="classification">Classification</option>
              <option value="regression">Regression</option>
              <option value="clustering">Clustering</option>
              <option value="time_series">Time Series</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input-field"
              rows={3}
            />
          </div>
          <div className="flex gap-3 justify-end">
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={registerMutation.isPending}
              className="btn-primary"
            >
              Register
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
