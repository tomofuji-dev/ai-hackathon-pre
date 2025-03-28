import { useEffect, useState } from "react";
import { Box, Typography } from "@mui/material";

interface ScoreAnimationProps {
  score: number;
}

export const ScoreAnimation = ({ score }: ScoreAnimationProps) => {
  const [currentScore, setCurrentScore] = useState(0);
  const [animationComplete, setAnimationComplete] = useState(false);

  useEffect(() => {
    // Reset animation state when score changes
    setCurrentScore(0);
    setAnimationComplete(false);
    
    // Animate the score counting up
    const duration = 1500; // Animation duration in ms
    const interval = 20; // Update interval in ms
    const steps = duration / interval;
    const increment = score / steps;
    let current = 0;
    let timer: number;
    
    const updateScore = () => {
      current += increment;
      if (current >= score) {
        current = score;
        setCurrentScore(score);
        setAnimationComplete(true);
        clearInterval(timer);
      } else {
        setCurrentScore(Math.floor(current));
      }
    };
    
    timer = window.setInterval(updateScore, interval);
    
    return () => {
      clearInterval(timer);
    };
  }, [score]);

  // Determine color based on score
  const getScoreColor = () => {
    if (score >= 80) return "#00E676"; // Green for high scores
    if (score >= 50) return "#FFC107"; // Amber for medium scores
    return "#F44336"; // Red for low scores
  };

  // Get appropriate message based on score
  const getMessage = () => {
    if (score >= 80) return "素晴らしい！";
    if (score >= 50) return "良い調子！";
    return "もう一度！";
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        position: "relative",
        py: 4,
      }}
    >
      <Typography
        variant="h1"
        sx={{
          fontSize: { xs: "5rem", sm: "8rem" },
          fontWeight: "bold",
          color: getScoreColor(),
          textShadow: `0 0 10px ${getScoreColor()}80`,
          animation: animationComplete
            ? "pulse 1s infinite alternate"
            : "none",
          "@keyframes pulse": {
            "0%": {
              transform: "scale(1)",
              opacity: 1,
            },
            "100%": {
              transform: "scale(1.05)",
              opacity: 0.9,
            },
          },
        }}
      >
        {currentScore}
      </Typography>
      
      {animationComplete && (
        <Typography
          variant="h4"
          sx={{
            mt: 2,
            fontWeight: "bold",
            color: getScoreColor(),
            animation: "fadeIn 0.5s",
            "@keyframes fadeIn": {
              "0%": {
                opacity: 0,
                transform: "translateY(20px)",
              },
              "100%": {
                opacity: 1,
                transform: "translateY(0)",
              },
            },
          }}
        >
          {getMessage()}
        </Typography>
      )}
    </Box>
  );
};
