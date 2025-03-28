import { Box, Typography, Button, Paper, Grid, Card, CardContent, CardMedia } from "@mui/material";
import { Link as RouterLink } from "react-router-dom";
import { useGame } from "@/context/GameContext";
import MicIcon from "@mui/icons-material/Mic";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import SchoolIcon from "@mui/icons-material/School";

export const HomePage = () => {
  const { resetGame, scores, totalScore, level } = useGame();

  const handleStartNewGame = () => {
    resetGame();
  };

  return (
    <Box sx={{ textAlign: "center", py: 4 }}>
      <Typography
        variant="h2"
        component="h1"
        gutterBottom
        sx={{
          fontWeight: "bold",
          background: "linear-gradient(45deg, #6200EA 30%, #00E676 90%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          mb: 4,
        }}
      >
        Voice Master Challenge
      </Typography>

      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Test your voice and earn points!
      </Typography>

      <Paper
        elevation={3}
        sx={{
          p: 4,
          mb: 6,
          maxWidth: 800,
          mx: "auto",
          borderRadius: 2,
          background: "rgba(30, 30, 30, 0.8)",
        }}
      >
        <Typography variant="h6" gutterBottom>
          How to Play:
        </Typography>
        <Typography variant="body1" paragraph align="left">
          1. Click the "Record" button and speak clearly into your microphone.
        </Typography>
        <Typography variant="body1" paragraph align="left">
          2. Stop the recording when you're done.
        </Typography>
        <Typography variant="body1" paragraph align="left">
          3. Submit your recording to see how well you did!
        </Typography>
        <Typography variant="body1" paragraph align="left">
          4. Earn points based on your performance and level up!
        </Typography>
      </Paper>

      <Grid container spacing={4} sx={{ mb: 6 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <CardMedia
                component={() => <MicIcon sx={{ fontSize: 80, color: "primary.main", mb: 2 }} />}
              />
              <Typography variant="h6" component="div">
                Record Your Voice
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Speak clearly and confidently to get the best score
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <CardMedia
                component={() => <SchoolIcon sx={{ fontSize: 80, color: "primary.main", mb: 2 }} />}
              />
              <Typography variant="h6" component="div">
                Get Feedback
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Receive instant analysis of your voice performance
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <CardMedia
                component={() => <EmojiEventsIcon sx={{ fontSize: 80, color: "primary.main", mb: 2 }} />}
              />
              <Typography variant="h6" component="div">
                Earn Points & Level Up
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Improve your skills and climb the ranks
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {scores.length > 0 && (
        <Paper
          elevation={3}
          sx={{
            p: 3,
            mb: 4,
            maxWidth: 600,
            mx: "auto",
            borderRadius: 2,
            background: "rgba(30, 30, 30, 0.8)",
          }}
        >
          <Typography variant="h6" gutterBottom>
            Your Stats
          </Typography>
          <Box sx={{ display: "flex", justifyContent: "space-around", mb: 2 }}>
            <Box>
              <Typography variant="h4" color="secondary.main">
                {totalScore}
              </Typography>
              <Typography variant="body2">Total Score</Typography>
            </Box>
            <Box>
              <Typography variant="h4" color="primary.main">
                {level}
              </Typography>
              <Typography variant="body2">Current Level</Typography>
            </Box>
            <Box>
              <Typography variant="h4" color="info.main">
                {scores.length}
              </Typography>
              <Typography variant="body2">Attempts</Typography>
            </Box>
          </Box>
        </Paper>
      )}

      <Box sx={{ display: "flex", justifyContent: "center", gap: 2, mt: 4 }}>
        <Button
          component={RouterLink}
          to="/game"
          variant="contained"
          color="primary"
          size="large"
          sx={{
            py: 1.5,
            px: 4,
            fontSize: "1.2rem",
            transition: "transform 0.2s",
            "&:hover": {
              transform: "scale(1.05)",
            },
          }}
        >
          {scores.length > 0 ? "Continue Playing" : "Start Game"}
        </Button>
        {scores.length > 0 && (
          <Button
            variant="outlined"
            color="secondary"
            size="large"
            onClick={handleStartNewGame}
            sx={{
              py: 1.5,
              px: 4,
            }}
          >
            New Game
          </Button>
        )}
      </Box>
    </Box>
  );
};
