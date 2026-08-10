// frontend/src/pages/ModelDetail.tsx
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { ArrowLeft, Brain, BarChart3, Zap, TrendingUp } from "lucide-react";
import toast from "react-hot-toast";

export default function ModelDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: model } = useQuery({
    queryKey: ["model", id],
    queryFn: () => apiClient.get(`/ml/models/${id}`).then((r) => r.data),
    enabled: !!id,
  });

  const deployMutation = useMutation({
    mutationFn: () => apiClient.post(`/ml/models/${id}/deploy`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["model", id] });
      toast.success("Model deployed");
    },
  });

  if (!model) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate("/models")}
          className="p-2 rounded-lg hover:bg-gray-100"
        >
          <ArrowLeft className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-2xl font-bold">{model.name}</h1>
          <p className="text-gray-500">{model.description}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card text-center">
          <Brain className="w-8 h-8 text-agri-600 mx-auto mb-2" />
          <p className="text-2xl font-bold">{model.model_type}</p>
          <p className="text-sm text-gray-500">Model Type</p>
        </div>
        <div className="card text-center">
          <BarChart3 className="w-8 h-8 text-blue-600 mx-auto mb-2" />
          <p className="text-2xl font-bold">{model.versions?.length || 0}</p>
          <p className="text-sm text-gray-500">Versions</p>
        </div>
        <div className="card text-center">
          <Zap className="w-8 h-8 text-amber-600 mx-auto mb-2" />
          <p className="text-2xl font-bold">
            {model.current_version?.deployed ? "Deployed" : "Not Deployed"}
          </p>
          <p className="text-sm text-gray-500">Status</p>
        </div>
      </div>

      {model.current_version?.metrics && (
        <div className="card">
          <h3 className="font-medium mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Metrics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(model.current_version.metrics).map(([key, val]) => (
              <div key={key} className="text-center p-3 bg-gray-50 rounded-lg">
                <p className="text-xl font-bold text-agri-600">
                  {typeof val === "number" ? val.toFixed(4) : String(val)}
                </p>
                <p className="text-xs text-gray-500 capitalize">
                  {key.replace(/_/g, " ")}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <h3 className="font-medium mb-4">Version History</h3>
        <div className="space-y-3">
          {model.versions?.map((v: any) => (
            <div
              key={v.version}
              className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg"
            >
              <span className="text-lg font-bold text-gray-400">
                v{v.version}
              </span>
              <div className="flex-1">
                <div className="flex gap-2 flex-wrap">
                  {Object.entries(v.metrics || {})
                    .slice(0, 3)
                    .map(([k, val]) => (
                      <span
                        key={k}
                        className="text-xs bg-white px-2 py-0.5 rounded border"
                      >
                        {k}:{" "}
                        {typeof val === "number" ? val.toFixed(3) : String(val)}
                      </span>
                    ))}
                </div>
              </div>
              <span className="text-xs text-gray-500">
                {new Date(v.created_at).toLocaleDateString()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {!model.current_version?.deployed && (
        <button
          onClick={() => deployMutation.mutate()}
          disabled={deployMutation.isPending}
          className="btn-primary"
        >
          {deployMutation.isPending ? "Deploying..." : "Deploy Model"}
        </button>
      )}
    </div>
  );
}
