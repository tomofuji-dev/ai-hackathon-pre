import { useEffect, useRef } from "react";
import { Box } from "@mui/material";

interface AudioVisualizerProps {
  isRecording: boolean;
}

export const AudioVisualizer = ({ isRecording }: AudioVisualizerProps) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!isRecording) {
      // Draw idle state - a flat line with small waves
      drawIdleState(ctx, width, height);
      return;
    }

    // Start animation for recording state
    let startTime = Date.now();
    
    const animate = () => {
      if (!ctx) return;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      // Calculate time difference for animation
      const elapsed = Date.now() - startTime;
      
      // Draw active waveform
      drawActiveWaveform(ctx, width, height, elapsed);
      
      // Continue animation
      animationRef.current = requestAnimationFrame(animate);
    };
    
    animate();
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRecording]);

  // Draw idle state visualization
  const drawIdleState = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number
  ) => {
    const centerY = height / 2;
    
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    
    // Draw a flat line with small waves
    for (let x = 0; x < width; x += 1) {
      const y = centerY + Math.sin(x * 0.05) * 2;
      ctx.lineTo(x, y);
    }
    
    ctx.strokeStyle = "#6200EA33"; // Primary color with transparency
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw a thinner line on top for aesthetics
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    for (let x = 0; x < width; x += 1) {
      const y = centerY + Math.sin(x * 0.05) * 1;
      ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "#6200EA";
    ctx.lineWidth = 1;
    ctx.stroke();
  };

  // Draw active waveform visualization
  const drawActiveWaveform = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    elapsed: number
  ) => {
    const centerY = height / 2;
    const amplitude = 30; // Maximum wave height
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, centerY - amplitude, 0, centerY + amplitude);
    gradient.addColorStop(0, "#6200EA");   // Primary color at top
    gradient.addColorStop(0.5, "#00E676"); // Secondary color in middle
    gradient.addColorStop(1, "#6200EA");   // Primary color at bottom
    
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    
    // Draw a more complex waveform
    for (let x = 0; x < width; x += 1) {
      // Combine multiple sine waves for more interesting visualization
      const y = centerY + 
        Math.sin(x * 0.02 + elapsed * 0.005) * amplitude * 0.5 +
        Math.sin(x * 0.03 - elapsed * 0.003) * amplitude * 0.3 +
        Math.sin(x * 0.01 + elapsed * 0.002) * amplitude * 0.2;
      
      ctx.lineTo(x, y);
    }
    
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Add glow effect
    ctx.shadowColor = "#6200EA";
    ctx.shadowBlur = 10;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Reset shadow for next frame
    ctx.shadowBlur = 0;
  };

  return (
    <Box
      sx={{
        width: "100%",
        height: "100%",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <canvas
        ref={canvasRef}
        width={400}
        height={200}
        style={{
          width: "100%",
          maxWidth: "400px",
          height: "auto",
        }}
      />
    </Box>
  );
};
