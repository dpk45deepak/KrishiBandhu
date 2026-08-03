// frontend/src/pages/Pipelines.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { apiClient } from "../lib/api";
import {
  Play,
  Plus,
  GitBranch,
  Clock,
  CheckCircle2,
  XCircle,
  RotateCw,
  Timer,
} from "lucide-react";
import toast from "react-hot-toast";
import clsx from "clsx";

interface PipelineRun {
  id: string;
  status: string;
  started_at: string;
  completed_at: string;
  duration_seconds: number;
  stages: { name: string; status: string; duration_seconds: number }[];
}

interface Pipeline {
  id: string;
  name: string;
  description: string;
  status: string;
  config: { stages: { stage_type: string; depends_on: string[] }[] };
  latest_run?: PipelineRun;
  run_count: number;
  success_rate: number;
  created_at: string;
}

export default function Pipelines() {
  const [showCreate, setShowCreate] = useState(false);
  const queryClient = useQueryClient();

  const { data: pipelines, isLoading } = useQuery({
    queryKey: ["pipelines"],
    queryFn: () => apiClient.get("/pipeline").then((r) => r.data as Pipeline[]),
  });

  const runMutation = useMutation({
    mutationFn: (id: string) => apiClient.post(`/pipeline/${id}/run`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      toast.success("Pipeline started");
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Pipelines</h1>
          <p className="text-gray-500 mt-1">Data processing and ML pipelines</p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="btn-primary flex items-center gap-2"
        >
          <Plus className="w-4 h-4" />
          New Pipeline
        </button>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-5 bg-gray-200 rounded w-1/4 mb-2" />
              <div className="h-4 bg-gray-200 rounded w-3/4" />
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {pipelines?.map((pipeline) => (
            <div key={pipeline.id} className="card">
              <div className="flex items-start justify-between">
                <Link to={`/pipelines/${pipeline.id}`} className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <GitBranch className="w-5 h-5 text-purple-600" />
                    <h3 className="font-semibold text-gray-900 hover:text-agri-600 transition-colors">
                      {pipeline.name}
                    </h3>
                    <span
                      className={clsx(
                        "badge",
                        pipeline.status === "active"
                          ? "badge-success"
                          : "badge-warning",
                      )}
                    >
                      {pipeline.status}
                    </span>
                  </div>
                  {pipeline.description && (
                    <p className="text-sm text-gray-500 mb-3">
                      {pipeline.description}
                    </p>
                  )}
                </Link>
                <button
                  onClick={() => runMutation.mutate(pipeline.id)}
                  disabled={runMutation.isPending}
                  className="btn-primary flex items-center gap-2 text-sm"
                >
                  <Play className="w-4 h-4" />
                  Run
                </button>
              </div>

              {/* Pipeline stages visualization */}
              <div className="flex items-center gap-2 mt-3">
                {pipeline.config.stages?.map((stage, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="px-3 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
                      {stage.stage_type}
                    </span>
                    {i < (pipeline.config.stages?.length || 0) - 1 && (
                      <div className="w-4 h-px bg-gray-300" />
                    )}
                  </div>
                ))}
              </div>

              {/* Stats */}
              <div className="flex items-center gap-6 mt-4 text-sm text-gray-500">
                <span className="flex items-center gap-1">
                  <RotateCw className="w-3 h-3" />
                  {pipeline.run_count} runs
                </span>
                <span className="flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-green-500" />
                  {pipeline.success_rate?.toFixed(0)}% success
                </span>
                {pipeline.latest_run && (
                  <>
                    <span className="flex items-center gap-1">
                      <Timer className="w-3 h-3" />
                      {pipeline.latest_run.duration_seconds?.toFixed(1)}s
                    </span>
                    <span
                      className={clsx(
                        "badge",
                        pipeline.latest_run.status === "completed"
                          ? "badge-success"
                          : pipeline.latest_run.status === "failed"
                            ? "badge-error"
                            : "badge-info",
                      )}
                    >
                      {pipeline.latest_run.status}
                    </span>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && pipelines?.length === 0 && (
        <div className="text-center py-12">
          <GitBranch className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <h3 className="font-medium">No pipelines yet</h3>
          <p className="text-gray-500 text-sm mt-1">
            Create a pipeline to automate your workflows
          </p>
        </div>
      )}

      {showCreate && (
        <CreatePipelineModal onClose={() => setShowCreate(false)} />
      )}
    </div>
  );
}

function CreatePipelineModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [stages, setStages] = useState<string[]>([
    "scan",
    "profile",
    "validate",
  ]);
  const queryClient = useQueryClient();

  const stageOptions = [
    "scan",
    "profile",
    "validate",
    "clean",
    "standardize",
    "feature_engineer",
    "feature_store",
    "train",
    "evaluate",
    "tune",
    "explain",
    "report",
  ];

  const createMutation = useMutation({
    mutationFn: (data: any) => apiClient.post("/pipeline", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipelines"] });
      toast.success("Pipeline created");
      onClose();
    },
  });

  const toggleStage = (stage: string) => {
    setStages((prev) =>
      prev.includes(stage) ? prev.filter((s) => s !== stage) : [...prev, stage],
    );
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-6 w-full max-w-lg">
        <h2 className="text-lg font-semibold mb-4">Create Pipeline</h2>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            createMutation.mutate({
              name,
              config: {
                stages: stages.map((s, i) => ({
                  stage_type: s,
                  depends_on: i > 0 ? [stages[i - 1]] : [],
                })),
              },
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
            <label className="block text-sm font-medium mb-2">Stages</label>
            <div className="flex flex-wrap gap-2">
              {stageOptions.map((stage) => (
                <button
                  key={stage}
                  type="button"
                  onClick={() => toggleStage(stage)}
                  className={clsx(
                    "px-3 py-1.5 rounded-full text-sm font-medium transition-colors",
                    stages.includes(stage)
                      ? "bg-agri-100 text-agri-700 border border-agri-300"
                      : "bg-gray-100 text-gray-500 border border-gray-200 hover:bg-gray-200",
                  )}
                >
                  {stage.replace("_", " ")}
                </button>
              ))}
            </div>
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
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
