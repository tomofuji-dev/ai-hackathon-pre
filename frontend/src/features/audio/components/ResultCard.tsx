import { Box, Card, CardContent, Typography, Divider, Chip, LinearProgress } from "@mui/material";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import SchoolIcon from "@mui/icons-material/School";

interface ResultCardProps {
  score: number;
  label: string;
  scoreColor: string;
  message: string;
  levelMessage: string;
}

export const ResultCard = ({ score, label, scoreColor, message, levelMessage }: ResultCardProps) => {
  // Calculate progress for the progress bar
  const progressValue = Math.min(100, score);
  
  return (
    <Card 
      elevation={4}
      sx={{ 
        borderRadius: 2,
        overflow: "hidden",
        position: "relative",
      }}
    >
      {/* Score indicator at the top */}
      <Box 
        sx={{ 
          bgcolor: scoreColor,
          py: 1,
          px: 2,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Typography variant="h6" sx={{ color: "#fff", fontWeight: "bold" }}>
          スコア
        </Typography>
        <Typography variant="h5" sx={{ color: "#fff", fontWeight: "bold" }}>
          {score}点
        </Typography>
      </Box>
      
      <CardContent sx={{ p: 3 }}>
        {/* Label */}
        <Box sx={{ mb: 3, textAlign: "center" }}>
          <Chip 
            label={`「${label}」`} 
            color="primary" 
            sx={{ 
              fontSize: "1.2rem", 
              py: 2.5, 
              px: 1,
              fontWeight: "bold",
            }} 
          />
        </Box>
        
        {/* Progress bar */}
        <Box sx={{ mb: 3 }}>
          <LinearProgress 
            variant="determinate" 
            value={progressValue} 
            sx={{ 
              height: 10, 
              borderRadius: 5,
              bgcolor: "rgba(0,0,0,0.1)",
              "& .MuiLinearProgress-bar": {
                bgcolor: scoreColor,
                borderRadius: 5,
              }
            }} 
          />
          <Box sx={{ display: "flex", justifyContent: "space-between", mt: 0.5 }}>
            <Typography variant="caption" color="text.secondary">0</Typography>
            <Typography variant="caption" color="text.secondary">100</Typography>
          </Box>
        </Box>
        
        <Divider sx={{ my: 2 }} />
        
        {/* Feedback message */}
        <Box sx={{ display: "flex", alignItems: "flex-start", mb: 2 }}>
          <EmojiEventsIcon sx={{ color: scoreColor, mr: 1, fontSize: 28 }} />
          <Typography variant="body1">
            {message}
          </Typography>
        </Box>
        
        {/* Level progress message */}
        <Box sx={{ display: "flex", alignItems: "flex-start" }}>
          <SchoolIcon sx={{ color: "primary.main", mr: 1, fontSize: 28 }} />
          <Typography variant="body1">
            {levelMessage}
          </Typography>
        </Box>
        
        {/* Achievement badge for high scores */}
        {score >= 80 && (
          <Box 
            sx={{ 
              position: "absolute", 
              top: -15, 
              right: -15, 
              bgcolor: "secondary.main",
              color: "#fff",
              width: 70,
              height: 70,
              borderRadius: "50%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
              boxShadow: 3,
              transform: "rotate(15deg)",
              zIndex: 1,
            }}
          >
            <EmojiEventsIcon sx={{ fontSize: 30 }} />
            <Typography variant="caption" sx={{ fontSize: "0.6rem", fontWeight: "bold" }}>
              EXCELLENT
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
