// frontend/src/pages/Inference.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { Zap, Play, Plus, Clock, Activity, BarChart3 } from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";

interface Endpoint {
  id: string;
  name: string;
  model_id: string;
  model_version: number;
  endpoint_path: string;
  status: string;
  request_count: number;
  error_count: number;
  avg_latency_ms: number;
  deployed_at: string;
}

export default function Inference() {
  const [showCreate, setShowCreate] = useState(false);
  const [testEndpoint, setTestEndpoint] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const { data: endpoints, isLoading } = useQuery({
    queryKey: ["endpoints"],
    queryFn: () =>
      apiClient.get("/inference/endpoints").then((r) => r.data as Endpoint[]),
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Inference Endpoints
          </h1>
          <p className="text-gray-500 mt-1">Deployed model serving endpoints</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Endpoint
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          {[...Array(2)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-5 bg-gray-200 rounded w-1/4 mb-2" />
              <div className="h-4 bg-gray-200 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {endpoints?.map((ep) => (
            <div key={ep.id} className="card">
              <div className="flex items-start justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-amber-500" />
                    <h3 className="font-semibold">{ep.name}</h3>
                    <span
                      className={clsx(
                        "badge",
                        ep.status === "active"
                          ? "badge-success"
                          : "badge-error",
                      )}
                    >
                      {ep.status}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1 font-mono">
                    {ep.endpoint_path}
                  </p>
                </div>
                <button
                  onClick={() => setTestEndpoint(ep.name)}
                  className="btn-secondary text-sm flex items-center gap-1"
                >
                  <Play className="w-3 h-3" />
                  Test
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">
                    {ep.request_count?.toLocaleString()}
                  </p>
                  <p className="text-xs text-gray-500">Requests</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-amber-600">
                    {ep.avg_latency_ms?.toFixed(0)} ms
                  </p>
                  <p className="text-xs text-gray-500">Avg Latency</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-red-600">
                    {ep.error_count}
                  </p>
                  <p className="text-xs text-gray-500">Errors</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-agri-600">
                    {ep.request_count > 0
                      ? (
                          ((ep.request_count - ep.error_count) /
                            ep.request_count) *
                          100
                        ).toFixed(1)
                      : 100}
                    %
                  </p>
                  <p className="text-xs text-gray-500">Success Rate</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && endpoints?.length === 0 && (
        <div className="text-center py-12">
          <Zap className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <h3 className="font-medium">No endpoints deployed</h3>
          <p className="text-gray-500 text-sm mt-1">
            Deploy a model to create an inference endpoint
          </p>
        </div>
      )}

      {testEndpoint && (
        <TestEndpointModal
          endpoint={testEndpoint}
          onClose={() => setTestEndpoint(null)}
        />
      )}
      {showCreate && (
        <CreateEndpointModal onClose={() => setShowCreate(false)} />
      )}
    </div>
  );
}

function TestEndpointModal({
  endpoint,
  onClose,
}: {
  endpoint: string;
  onClose: () => void;
}) {
  const [input, setInput] = useState('{"feature1": 1.0, "feature2": 2.0}');
  const [result, setResult] = useState<any>(null);

  const testMutation = useMutation({
    mutationFn: (instances: any[]) =>
      apiClient.post(`/inference/endpoints/${endpoint}/predict`, { instances }),
    onSuccess: (data) => setResult(data.data),
  });

  const handleTest = () => {
    try {
      const parsed = JSON.parse(input);
      testMutation.mutate(Array.isArray(parsed) ? parsed : [parsed]);
    } catch {
      toast.error("Invalid JSON input");
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-lg">
        <h2 className="font-semibold mb-4">Test Endpoint: {endpoint}</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Input (JSON)
            </label>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="input-field font-mono text-sm"
              rows={6}
            />
          </div>
          <button
            onClick={handleTest}
            disabled={testMutation.isPending}
            className="btn-primary"
          >
            {testMutation.isPending ? "Predicting..." : "Predict"}
          </button>
          {result && (
            <div>
              <label className="block text-sm font-medium mb-1">Result</label>
              <pre className="bg-gray-50 p-3 rounded-lg text-sm overflow-auto max-h-48">
                {JSON.stringify(result, null, 2)}
              </pre>
              <p className="text-xs text-gray-500 mt-1">
                Latency: {result.prediction_time_ms?.toFixed(1)}ms
              </p>
            </div>
          )}
        </div>
        <button onClick={onClose} className="btn-secondary mt-4 w-full">
          Close
        </button>
      </div>
    </div>
  );
}

function CreateEndpointModal({ onClose }: { onClose: () => void }) {
  const [modelId, setModelId] = useState("");
  const [version, setVersion] = useState(1);
  const [name, setName] = useState("");
  const queryClient = useQueryClient();
  const { data: models } = useQuery({
    queryKey: ["models"],
    queryFn: () => apiClient.get("/ml/models").then((r) => r.data),
  });

  const createMutation = useMutation({
    mutationFn: () =>
      apiClient.post("/inference/endpoints", null, {
        params: { model_id: modelId, model_version: version, name },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["endpoints"] });
      toast.success("Endpoint created");
      onClose();
    },
  });

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-md">
        <h2 className="font-semibold mb-4">Create Endpoint</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            createMutation.mutate();
          }}
          className="space-y-4"
        >
          <div>
            <label className="block text-sm font-medium mb-1">Model</label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              className="input-field"
              required
            >
              <option value="">Select model...</option>
              {models?.map((m: any) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Version</label>
            <input
              type="number"
              value={version}
              onChange={(e) => setVersion(Number(e.target.value))}
              className="input-field"
              min={1}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Endpoint Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input-field"
              required
            />
          </div>
          <div className="flex gap-3 justify-end">
            <button type="button" onClick={onClose} className="btn-secondary">
              Cancel
            </button>
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="btn-primary"
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
