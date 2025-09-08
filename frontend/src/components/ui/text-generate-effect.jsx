"use client";
import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { cn } from "../../lib/utils";

export const TextGenerateEffect = ({
  words,
  className,
  filter = true,
  duration = 1,
}) => {
  const [displayedText, setDisplayedText] = useState("");
  const [currentIndex, setCurrentIndex] = useState(0);
  const intervalRef = useRef(null);

  const wordsArray = words.split(" ");

  useEffect(() => {
    if (currentIndex < wordsArray.length) {
      intervalRef.current = setInterval(() => {
        setDisplayedText(
          (prev) => prev + (prev ? " " : "") + wordsArray[currentIndex]
        );
        setCurrentIndex((prev) => prev + 1);
      }, duration * 100);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [currentIndex, wordsArray, duration]);

  // Reset when words change
  useEffect(() => {
    setDisplayedText("");
    setCurrentIndex(0);
  }, [words]);

  const renderWords = () => {
    return (
      <motion.div className={className}>
        {displayedText.split(" ").map((word, idx) => {
          return (
            <motion.span
              key={word + idx}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{
                duration: 0.25,
                ease: "easeInOut",
                delay: idx * 0.1,
              }}
              className={cn(
                "text-black dark:text-white opacity-0",
                filter && "filter blur-sm"
              )}
            >
              {word}{" "}
            </motion.span>
          );
        })}
      </motion.div>
    );
  };

  return <div className={cn("font-bold", className)}>{renderWords()}</div>;
};