"use client";
import React from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export const ShimmerButton = ({
  shimmerColor = "#ffffff",
  shimmerSize = "0.05em",
  borderRadius = "100px",
  shimmerDuration = "3s",
  background = "rgba(0, 0, 0, 1)",
  className,
  children,
  ...props
}) => {
  return (
    <motion.button
      style={{
        "--spread": "90deg",
        "--shimmer-color": shimmerColor,
        "--radius": borderRadius,
        "--speed": shimmerDuration,
        "--cut": shimmerSize,
        "--bg": background,
      }}
      className={cn(
        "group relative z-0 flex cursor-pointer items-center justify-center overflow-hidden whitespace-nowrap border border-white/10 px-6 py-3 text-white [background:var(--bg)] [border-radius:var(--radius)] dark:[border:1px_solid_rgba(255,255,255,.1)] dark:[box-shadow:0_-20px_80px_-20px_#ffffff1f_inset]",
        className
      )}
      initial={{ scale: 1 }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", stiffness: 400, damping: 17 }}
      {...props}
    >
      {/* spark container */}
      <div className="absolute inset-0 overflow-visible [container-type:size]">
        {/* spark */}
        <div className="absolute inset-0 h-full w-full [background-image:conic-gradient(from_0deg,transparent_0_340deg,var(--shimmer-color)_360deg)] [mask:radial-gradient(ellipse_at_center,transparent_20%,black_50%)] dark:[background-image:conic-gradient(from_0deg,transparent_0_340deg,var(--shimmer-color)_360deg)] before:absolute before:inset-0 before:h-full before:w-full before:animate-spin before:duration-[var(--speed)] before:[background-image:conic-gradient(from_0deg,transparent_0_340deg,var(--shimmer-color)_360deg)] before:[mask:radial-gradient(ellipse_at_center,transparent_20%,black_50%)]"></div>
      </div>

      {/* backdrop */}
      <div className="absolute -z-30 h-full w-full blur-md [background-image:conic-gradient(from_0deg,transparent_0_340deg,var(--shimmer-color)_360deg)] [mask:radial-gradient(ellipse_at_center,transparent_20%,black_50%)] dark:[background-image:conic-gradient(from_0deg,transparent_0_340deg,var(--shimmer-color)_360deg)]"></div>
      {children}

      {/* Highlight */}
      <div className="absolute inset-[calc(var(--cut)*-1)] rounded-[calc(var(--radius)+var(--cut))] bg-gradient-to-b from-transparent via-white/5 to-transparent p-[calc(var(--cut))] opacity-50 transition-opacity duration-500 group-hover:opacity-80"></div>
    </motion.button>
  );
};