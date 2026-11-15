"use client";

import dynamic from "next/dynamic";
import { Skeleton } from "@/components/common/Skeleton";

const IndiaMap = dynamic(() => import("./IndiaMap"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-[520px] bg-surface rounded-xl">
      <Skeleton className="h-8 w-32" />
    </div>
  ),
});

export default IndiaMap;
