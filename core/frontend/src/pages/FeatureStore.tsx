// frontend/src/pages/FeatureStore.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";
import { Layers, Database, Search, Clock, BarChart3 } from "lucide-react";
import clsx from "clsx";

interface FeatureGroup {
  id: string;
  name: string;
  description: string;
  feature_count: number;
  entity_key: string;
  version: number;
  status: string;
  row_count?: number;
  created_at: string;
}

export default function FeatureStore() {
  const [search, setSearch] = useState("");

  const { data: groups, isLoading } = useQuery({
    queryKey: ["featureGroups"],
    queryFn: () =>
      apiClient.get("/features/groups").then((r) => r.data as FeatureGroup[]),
  });

  const filtered = groups?.filter(
    (g) => !search || g.name.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Feature Store</h1>
        <p className="text-gray-500 mt-1">
          Centralized feature management for ML
        </p>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          placeholder="Search feature groups..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="input-field pl-10"
        />
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-5 bg-gray-200 rounded w-1/3 mb-3" />
              <div className="h-4 bg-gray-200 rounded w-full" />
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filtered?.map((group) => (
            <div
              key={group.id}
              className="card hover:shadow-md transition-shadow"
            >
              <div className="flex items-center gap-2 mb-2">
                <Layers className="w-5 h-5 text-blue-600" />
                <h3 className="font-semibold">{group.name}</h3>
                <span
                  className={clsx(
                    "badge",
                    group.status === "active"
                      ? "badge-success"
                      : "badge-warning",
                  )}
                >
                  {group.status}
                </span>
              </div>
              <p className="text-sm text-gray-500 mb-3">{group.description}</p>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Database className="w-3 h-3" />
                  {group.feature_count} features
                </span>
                <span>Entity: {group.entity_key}</span>
                {group.row_count && (
                  <span>{group.row_count?.toLocaleString()} rows</span>
                )}
                <span className="flex items-center gap-1 ml-auto">
                  <Clock className="w-3 h-3" />
                  {new Date(group.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && filtered?.length === 0 && (
        <div className="text-center py-12">
          <Layers className="w-12 h-12 text-gray-300 mx-auto mb-3" />
          <h3 className="font-medium">No feature groups</h3>
          <p className="text-gray-500 text-sm mt-1">
            Feature groups will appear here after pipeline execution
          </p>
        </div>
      )}
    </div>
  );
}
