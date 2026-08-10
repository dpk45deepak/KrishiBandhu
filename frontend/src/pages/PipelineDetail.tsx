// frontend/src/pages/PipelineDetail.tsx
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../lib/api';
import {
  ArrowLeft, Play, RotateCw, CheckCircle2, XCircle, Clock, Timer, AlertTriangle,
  ChevronRight,
} from 'lucide-react';
import toast from 'react-hot-toast';
import clsx from 'clsx';

export default function PipelineDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const { data: pipeline } = useQuery({
    queryKey: ['pipeline', id],
    queryFn: () => apiClient.get(`/pipeline/${id}`).then((r) => r.data),
    enabled: !!id,
  });

  const { data: runs } = useQuery({
    queryKey: ['pipelineRuns', id],
    queryFn: () => apiClient.get(`/pipeline/${id}/runs`).then((r) => r.data),
    enabled: !!id,
  });

  const runMutation = useMutation({
    mutationFn: () => apiClient.post(`/pipeline/${id}/run`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pipelineRuns', id] });
      queryClient.invalidateQueries({ queryKey: ['pipeline', id] });
      toast.success('Pipeline started');
    },
  });

  if (!pipeline) return null;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button onClick={() => navigate('/pipelines')} className="p-2 rounded-lg hover:bg-gray-100">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div>
            <h1 className="text-2xl font-bold">{pipeline.name}</h1>
            <p className="text-gray-500">{pipeline.description}</p>
          </div>
        </div>
        <button onClick={() => runMutation.mutate()} className="btn-primary flex items-center gap-2">
          <Play className="w-4 h-4" />
          Run Pipeline
        </button>
      </div>

      {/* Pipeline DAG visualization */}
      <div className="card">
        <h3 className="font-medium mb-4">Pipeline Stages</h3>
        <div className="flex flex-wrap items-center gap-2">
          {pipeline.config?.stages?.map((stage: any, i: number) => (
            <div key={i} className="flex items-center gap-2">
              <div className="px-4 py-2 rounded-lg bg-gray-100 border border-gray-200">
                <p className="font-medium text-sm capitalize">{stage.stage_type.replace('_', ' ')}</p>
                {stage.depends_on?.length > 0 && (
                  <p className="text-xs text-gray-500 mt-0.5">
                    ← {stage.depends_on.join(', ')}
                  </p>
                )}
              </div>
              {i < (pipeline.config.stages?.length || 0) - 1 && (
                <ChevronRight className="w-4 h-4 text-gray-400" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Runs history */}
      <div className="card">
        <h3 className="font-medium mb-4">Run History</h3>
        <div className="space-y-3">
          {runs?.map((run: any) => (
            <div key={run.id} className="flex items-center gap-4 p-3 rounded-lg border">
              <div className={clsx(
                'w-3 h-3 rounded-full',
                run.status === 'completed' ? 'bg-green-500' :
                run.status === 'failed' ? 'bg-red-500' :
                run.status === 'running' ? 'bg-blue-500 animate-pulse' : 'bg-gray-300'
              )} />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-medium">{run.id?.slice(0, 8)}</span>
                  <span className={clsx(
                    'badge',
                    run.status === 'completed' ? 'badge-success' :
                    run.status === 'failed' ? 'badge-error' : 'badge-info'
                  )}>
                    {run.status}
                  </span>
                </div>
                <div className="flex gap-4 text-xs text-gray-500 mt-1">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date(run.started_at).toLocaleString()}
                  </span>
                  {run.duration_seconds && (
                    <span className="flex items-center gap-1">
                      <Timer className="w-3 h-3" />
                      {run.duration_seconds.toFixed(1)}s
                    </span>
                  )}
                </div>
                {run.stages && (
                  <div className="flex gap-1 mt-2">
                    {run.stages.map((stage: any, si: number) => (
                      <div
                        key={si}
                        className={clsx(
                          'w-2 h-2 rounded-full',
                          stage.status === 'completed' ? 'bg-green-500' :
                          stage.status === 'failed' ? 'bg-red-500' :
                          stage.status === 'running' ? 'bg-blue-500' : 'bg-gray-200'
                        )}
                        title={`${stage.name}: ${stage.status}`}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          {(!runs || runs.length === 0) && (
            <p className="text-gray-500 text-sm text-center py-4">No runs yet</p>
          )}
        </div>
      </div>
    </div>
  );
}